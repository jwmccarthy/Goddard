import unittest

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch as th

from jarl.data.batch import TensorBatch

import difo
from difo import (
    AGENT_CONT_DIM,
    EDGE_CONT_DIM,
    GOAL_CENTER_Y,
    GOAL_CENTER_Z,
    PAIR_TARGET_DIM,
    ContactTrackingEnv,
    DIFOReward,
    DIFOUpdate,
    DIFOTransitionCapture,
    ExpertTransitionBuffer,
    GaussianDiffusion,
    GraphDIFO,
    InteractionGraph,
    InteractionGraphEncoder,
    bounded_distance_gate,
    build_policy,
    build_transition,
    build_transition_graph,
    entities_from_observation,
    entities_from_replay_state,
    gather_pairs,
    global_target,
    global_target_dim,
    observation_width,
    pair_target,
    perturb_graph,
    position_scale,
    reward_scales,
    TransitionDenoiser,
)


N_CARS = 2
OBS_WIDTH = observation_width(N_CARS)


class AllValidActionCodec:
    action_shape = (7,)

    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones((*state.shape[:-1], 18), dtype=th.bool, device=state.device)


def random_observation(count: int, scale: float = 0.2) -> th.Tensor:
    observation = th.zeros(count, OBS_WIDTH)
    observation[:, 0:9] = scale * th.randn(count, 9)
    observation[:, 9 : 9 + 21 * N_CARS] = (
        0.1 * scale * th.randn(count, 21 * N_CARS)
    )
    observation[:, 9 + 21 * N_CARS + 68] = -0.3
    observation[:, 9 + 21 * N_CARS + 71] = 0.3
    return observation


def make_replay_buffer(
    segment_lengths: tuple[int, ...] = (20, 30, 10),
    delta_rows: tuple[int, ...] = (1, 2),
) -> ExpertTransitionBuffer:
    lengths = th.tensor(segment_lengths)
    offsets = th.cat((th.zeros(1, dtype=th.long), lengths.cumsum(0)))
    states = 0.05 * th.randn(int(lengths.sum()), 71)
    states[:, 24] = 0.5
    states[:, 25] = 1.0
    return ExpertTransitionBuffer(
        SimpleNamespace(
            _replays=states,
            _offsets=offsets,
            _sampling_probabilities=th.ones(len(segment_lengths)),
            frame_skip=4,
        ),
        delta_rows=delta_rows,
        seed=0,
    )


class BoundedDistanceGateTest(unittest.TestCase):
    def test_matches_specification(self):
        sigma, power = 500.0, 4.0
        self.assertAlmostEqual(
            float(bounded_distance_gate(th.tensor([0.0]), sigma, power)), 1.0
        )
        self.assertAlmostEqual(
            float(bounded_distance_gate(th.tensor([sigma]), sigma, power)), 0.5
        )
        far = float(bounded_distance_gate(th.tensor([5000.0]), sigma, power))
        self.assertLess(far, 1e-3)
        distances = th.linspace(0.0, 4000.0, 50)
        gates = bounded_distance_gate(distances, sigma, power)
        self.assertTrue(bool((gates[1:] <= gates[:-1]).all()))

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            bounded_distance_gate(th.ones(1), 0.0, 4.0)
        with self.assertRaises(ValueError):
            bounded_distance_gate(th.ones(1), 500.0, 0.0)


class EntityGraphTest(unittest.TestCase):
    def test_observation_features_are_ball_relative(self):
        observation = random_observation(3)
        entities = entities_from_observation(observation, N_CARS)
        self.assertEqual(entities.car_position.shape, (3, N_CARS, 3))
        graph, gates = build_transition_graph(
            entities, entities, 500.0, 4.0
        )
        relative = entities.car_position - entities.ball_position[:, None, :]
        self.assertTrue(
            th.allclose(graph.node_cont[:, :N_CARS, 0:3], relative)
        )
        self.assertEqual(graph.node_cont.shape[-1], AGENT_CONT_DIM)
        self.assertEqual(gates.shape, (3, N_CARS))

    def test_transition_gate_uses_both_endpoints(self):
        observation = random_observation(1)
        observation[0, 0:3] = th.tensor([0.0, 0.0, 0.0])
        observation[0, 9:12] = th.tensor([0.0, 1000.0 / 6000.0, 0.0])
        next_observation = observation.clone()
        next_observation[0, 9:12] = th.tensor([0.0, 100.0 / 6000.0, 0.0])
        entities = entities_from_observation(observation, N_CARS)
        next_entities = entities_from_observation(next_observation, N_CARS)
        _, gates = build_transition_graph(
            entities, next_entities, 500.0, 4.0
        )
        self.assertGreater(float(gates[0, 0]), 0.9)

    def test_contact_sets_gate_to_one(self):
        observation = random_observation(1)
        observation[0, 9:12] = th.tensor([0.0, 3000.0 / 6000.0, 0.0])
        entities = entities_from_observation(
            observation, N_CARS, ego_contact=th.tensor([True])
        )
        _, gates = build_transition_graph(entities, entities, 500.0, 4.0)
        self.assertEqual(float(gates[0, 0]), 1.0)

    def test_gate_biases_attention_away_from_far_agents(self):
        th.manual_seed(0)
        observation = random_observation(1)
        observation[0, 9:12] = th.tensor([0.0, 0.0, 0.0])
        observation[0, 30:33] = th.tensor([0.0, 5000.0 / 6000.0, 0.0])
        entities = entities_from_observation(observation, N_CARS)
        graph, _ = build_transition_graph(entities, entities, 500.0, 4.0)
        encoder = InteractionGraph(feature_dim=16, layers=1)
        far_graph = graph
        mutated = graph.node_cont.clone()
        mutated[:, 1] = th.randn_like(mutated[:, 1])
        mutated_graph = graph.__class__(
            node_cont=mutated,
            node_discrete=graph.node_discrete,
            node_type=graph.node_type,
            edge_feats=graph.edge_feats,
            edge_gate=graph.edge_gate,
            edge_mask=graph.edge_mask,
        )
        with th.no_grad():
            original = encoder(
                far_graph.node_cont,
                far_graph.node_discrete,
                far_graph.node_type,
                far_graph.edge_feats,
                far_graph.edge_gate,
                far_graph.edge_mask,
            )
            mutated_out = encoder(
                mutated_graph.node_cont,
                mutated_graph.node_discrete,
                mutated_graph.node_type,
                mutated_graph.edge_feats,
                mutated_graph.edge_gate,
                mutated_graph.edge_mask,
            )
        ball_delta = (original[:, N_CARS] - mutated_out[:, N_CARS]).abs().max()
        self.assertLess(float(ball_delta), 1e-4)


class PairGraphTest(unittest.TestCase):
    def test_pair_ordering_and_target_duplication(self):
        observation = random_observation(2)
        next_observation = observation + 0.01
        entities = entities_from_observation(observation, N_CARS)
        next_entities = entities_from_observation(next_observation, N_CARS)
        graph, gates, agent_delta, ball_delta = build_transition(
            entities, next_entities, 500.0, 4.0
        )
        pairs = gather_pairs(graph)
        self.assertEqual(
            pairs.node_cont.shape, (2 * N_CARS, N_CARS + 1, AGENT_CONT_DIM)
        )
        self.assertEqual(pairs.edge_feats.shape[-1], EDGE_CONT_DIM)
        for agent in range(N_CARS):
            row = agent
            self.assertTrue(
                th.allclose(
                    pairs.node_cont[row, 0], graph.node_cont[0, agent]
                )
            )
            self.assertTrue(
                th.allclose(
                    pairs.node_cont[row, 1], graph.node_cont[0, N_CARS]
                )
            )
        targets = pair_target(agent_delta, ball_delta)
        self.assertEqual(targets.shape, (2, N_CARS, PAIR_TARGET_DIM))
        self.assertTrue(th.allclose(targets[..., :3], targets[..., 32:35]))
        self.assertTrue(th.allclose(targets[..., 3:6], targets[..., 35:38]))
        self.assertEqual(
            global_target(agent_delta, ball_delta).shape[1],
            global_target_dim(N_CARS),
        )
        self.assertEqual(gates.shape, (2, N_CARS))


class GaussianDiffusionTest(unittest.TestCase):
    def test_schedule_is_monotone(self):
        diffusion = GaussianDiffusion(25)
        alpha = diffusion.alphas_cumprod
        self.assertEqual(alpha.shape, (25,))
        self.assertTrue(bool((alpha[1:] <= alpha[:-1]).all()))
        self.assertGreater(float(alpha[-1]), 0.0)
        self.assertLess(float(alpha[-1]), 1.0)

    def test_add_noise_shapes_and_endpoints(self):
        diffusion = GaussianDiffusion(10)
        clean = th.randn(4, 5)
        noise = th.randn(4, 5)
        timesteps = th.tensor([0, 3, 9, 5])
        noisy = diffusion.add_noise(clean, noise, timesteps)
        self.assertEqual(noisy.shape, clean.shape)
        exact = (
            diffusion.sqrt_alphas_cumprod[0] * clean
            + diffusion.sqrt_one_minus_alphas_cumprod[0] * noise
        )
        near_clean = diffusion.add_noise(
            clean, noise, th.zeros(4, dtype=th.long)
        )
        self.assertTrue(th.allclose(near_clean, exact, atol=1e-5))
        sampled = diffusion.sample_timesteps(64)
        self.assertEqual(sampled.shape, (64,))
        self.assertGreaterEqual(int(sampled.min()), 0)
        self.assertLess(int(sampled.max()), 10)


class ExpertTransitionBufferTest(unittest.TestCase):
    def test_samples_within_segments_and_matches_delta(self):
        buffer = make_replay_buffer()
        entities, next_entities, delta_t = buffer.sample(64, 2)
        self.assertEqual(entities.car_position.shape, (64, N_CARS, 3))
        self.assertTrue(th.allclose(delta_t, th.full((64, 1), 2 * 4 / 120.0)))
        scale = position_scale(entities.ball_position.device)
        goal_center = th.tensor((0.0, GOAL_CENTER_Y, GOAL_CENTER_Z))
        ball_world = entities.ball_position * scale
        expected = (
            goal_center * th.tensor((-1.0, 1.0, 1.0)) - ball_world
        ) / (2.0 * scale)
        self.assertTrue(th.allclose(entities.own_goal_relative, expected))

    def test_raises_when_no_segment_is_long_enough(self):
        buffer = make_replay_buffer(segment_lengths=(2, 3))
        with self.assertRaises(RuntimeError):
            buffer.sample(4, 5)

    def test_replay_entities_match_observation_layout(self):
        rows = 0.05 * th.randn(2, 71)
        rows[:, 25] = 1.0
        entities = entities_from_replay_state(
            rows, th.tensor([True, False])
        )
        self.assertTrue(bool(entities.contact[0, 0]))
        self.assertFalse(bool(entities.contact[0, 1]))
        self.assertTrue(bool(entities.car_on_ground[0, 0]))
        self.assertEqual(entities.ball_position.shape, (2, 3))


def make_graph_data(count: int = 8, seed: int = 0):
    generator = th.Generator().manual_seed(seed)
    observation = random_observation(count)
    next_observation = observation + 0.05 * th.randn(count, OBS_WIDTH, generator=generator)
    entities = entities_from_observation(observation, N_CARS)
    next_entities = entities_from_observation(next_observation, N_CARS)
    return build_transition(entities, next_entities, 500.0, 4.0)


class GraphDIFOTest(unittest.TestCase):
    def _modules(self, target_dim: int, condition_dim: int, **kwargs):
        return GraphDIFO(
            InteractionGraph(feature_dim=16, layers=1),
            target_dim=target_dim,
            diffusion=GaussianDiffusion(10),
            condition_dim=condition_dim,
            hidden=16,
            **kwargs,
        )

    def test_zero_lambda_reward_is_log_two(self):
        graph, _, agent_delta, ball_delta = make_graph_data(4)
        model = GraphDIFO(
            InteractionGraph(feature_dim=16, layers=1),
            target_dim=global_target_dim(N_CARS),
            diffusion=GaussianDiffusion(10),
            condition_dim=16,
            hidden=16,
            include_ego=False,
            lam=0.0,
        )
        target = global_target(agent_delta, ball_delta)
        reward = model.reward(graph, target, th.full((4, 1), 1 / 30))
        self.assertTrue(th.allclose(reward, th.full((4,), th.log(th.tensor(2.0)))))

    def test_training_separates_expert_from_agent(self):
        graph, _, agent_delta, ball_delta = make_graph_data(32, seed=1)
        target = global_target(agent_delta, ball_delta)
        expert_target = th.zeros_like(target[:16])
        agent_target = th.full_like(target[16:], 3.0)
        is_expert = th.cat((th.ones(16), th.zeros(16)))
        model = GraphDIFO(
            InteractionGraph(feature_dim=16, layers=1),
            target_dim=global_target_dim(N_CARS),
            diffusion=GaussianDiffusion(10),
            condition_dim=16,
            hidden=16,
            include_ego=False,
        )
        optimizer = th.optim.Adam(model.parameters(), lr=1e-2)
        loss = None
        for _ in range(60):
            loss, _ = model.training_loss(
                graph,
                th.cat((expert_target, agent_target)),
                is_expert,
                th.full((32, 1), 1 / 30),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss, final_metrics = model.training_loss(
            graph,
            th.cat((expert_target, agent_target)),
            is_expert,
            th.full((32, 1), 1 / 30),
        )
        self.assertGreater(float(final_metrics["expert_accuracy"]), 0.9)
        self.assertGreater(float(final_metrics["agent_accuracy"]), 0.9)
        self.assertLess(float(final_loss.detach()), float(loss.detach()))

    def test_gate_zero_disables_pair_rows(self):
        pairs = None
        graph, gates, agent_delta, ball_delta = make_graph_data(4)
        pairs = gather_pairs(graph)
        targets = pair_target(agent_delta, ball_delta).reshape(-1, PAIR_TARGET_DIM)
        model = self._modules(PAIR_TARGET_DIM, 32, include_ego=True)
        is_expert = th.tensor([1.0, 0.0] * 4)
        zero_gate = th.zeros(4 * N_CARS)
        gated_loss, _ = model.training_loss(
            pairs, targets, is_expert, th.full((4 * N_CARS, 1), 1 / 30),
            gate=zero_gate,
        )
        self.assertEqual(float(gated_loss.detach()), 0.0)


class TransitionDenoiserTest(unittest.TestCase):
    def test_forward_with_independent_embedding_dimensions(self):
        denoiser = TransitionDenoiser(
            target_dim=5,
            condition_dim=7,
            hidden=32,
            time_dim=8,
            label_dim=4,
            delta_dim=6,
        )
        output = denoiser(
            noisy=th.randn(3, 5),
            timesteps=th.tensor([0, 4, 9]),
            labels=th.tensor([1, 0, 1]),
            delta_t=th.full((3, 1), 1 / 30),
            condition=th.randn(3, 7),
        )
        self.assertEqual(output.shape, (3, 5))

    def test_graph_difo_trains_with_hidden_wider_than_label(self):
        graph, _, agent_delta, ball_delta = make_graph_data(4)
        model = GraphDIFO(
            InteractionGraph(feature_dim=32, layers=1),
            target_dim=global_target_dim(N_CARS),
            diffusion=GaussianDiffusion(10),
            condition_dim=32,
            hidden=32,
            include_ego=False,
        )
        loss, metrics = model.training_loss(
            graph,
            global_target(agent_delta, ball_delta),
            th.tensor([1.0, 0.0, 1.0, 0.0]),
            th.full((4, 1), 1 / 30),
        )
        self.assertTrue(bool(th.isfinite(loss)))
        self.assertIn("expert_accuracy", metrics)


class DIFOUpdateAndRewardTest(unittest.TestCase):
    def _rollout(self, steps: int = 6, envs: int = 3, done: bool = False):
        return TensorBatch(
            {
                "observation": random_observation(steps * envs).reshape(
                    steps, envs, OBS_WIDTH
                ),
                "next_obs": random_observation(steps * envs).reshape(
                    steps, envs, OBS_WIDTH
                ),
                "difo_touch_self": th.zeros(steps, envs, dtype=th.bool),
                "terminated": th.full((steps, envs), done, dtype=th.bool),
                "truncated": th.zeros(steps, envs, dtype=th.bool),
                "learner_mask": th.ones(steps, envs, dtype=th.bool),
                "reward": th.ones(steps, envs),
                "action": th.zeros(steps, envs, 1),
                "old_log_prob": th.zeros(steps, envs),
            }
        )

    def _models(self):
        global_difo = GraphDIFO(
            InteractionGraph(16, EDGE_CONT_DIM, 1),
            target_dim=global_target_dim(N_CARS),
            diffusion=GaussianDiffusion(10),
            condition_dim=16,
            hidden=16,
            include_ego=False,
        )
        pair_difo = GraphDIFO(
            InteractionGraph(16, EDGE_CONT_DIM, 1),
            target_dim=PAIR_TARGET_DIM,
            diffusion=GaussianDiffusion(10),
            condition_dim=32,
            hidden=16,
            include_ego=True,
        )
        return global_difo, pair_difo

    def test_update_returns_metrics_and_rollout(self):
        global_difo, pair_difo = self._models()
        expert = make_replay_buffer()
        optimizer = th.optim.Adam(
            list(global_difo.parameters()) + list(pair_difo.parameters()),
            lr=1e-3,
        )
        stage = DIFOUpdate(
            expert,
            global_difo,
            pair_difo,
            optimizer,
            batch_size=2,
            n_cars=N_CARS,
            delta_rows=(1,),
            frame_skip=4,
            perturbation=0.01,
        )
        rollout = self._rollout()
        result, metrics = stage.run(rollout)
        self.assertIs(result, rollout)
        values = metrics["DIFO"]
        self.assertGreater(values["minibatches"], 0)
        self.assertLess(values["global_loss"], 100.0)
        self.assertLess(values["pair_loss"], 100.0)
        self.assertTrue(0.0 <= values["global_expert_accuracy"] <= 1.0)

    def test_update_skips_when_nothing_is_learnable(self):
        global_difo, pair_difo = self._models()
        expert = make_replay_buffer()
        optimizer = th.optim.Adam(
            list(global_difo.parameters()) + list(pair_difo.parameters())
        )
        stage = DIFOUpdate(
            expert,
            global_difo,
            pair_difo,
            optimizer,
            batch_size=2,
            n_cars=N_CARS,
        )
        rollout = self._rollout()
        rollout = rollout.replace_fields(
            learner_mask=th.zeros(6, 3, dtype=th.bool)
        )
        _, metrics = stage.run(rollout)
        self.assertEqual(metrics["DIFO"]["minibatches"], 0.0)

    def test_reward_is_added_and_masked_on_done(self):
        global_difo, pair_difo = self._models()
        transform = DIFOReward(
            global_difo,
            pair_difo,
            beta=0.5,
            sigma=500.0,
            gate_power=4.0,
            n_cars=N_CARS,
            frame_skip=4,
        )
        rollout = self._rollout(done=False)
        out = transform(rollout, None)
        environment = rollout["reward"]
        intrinsic = out["reward"] - environment
        self.assertGreater(float(intrinsic.abs().mean()), 0.0)
        metrics = transform.metrics()["DIFOReward"]
        self.assertIn("interacting_fraction", metrics)

        done_rollout = self._rollout(done=True)
        done_out = transform(done_rollout, None)
        self.assertTrue(
            th.allclose(done_out["reward"], done_rollout["reward"])
        )

    def test_reward_scales_by_mode(self):
        scales = reward_scales
        self.assertEqual(
            scales("nexto", 10.0, 0.7, 0.1, 1.0), (10.0, 0.7, 0.1, 1.0)
        )
        self.assertEqual(
            scales("goals", 10.0, 0.7, 0.1, 1.0), (10.0, 0.0, 0.0, 0.0)
        )
        self.assertEqual(
            scales("imitation", 10.0, 0.7, 0.1, 1.0), (0.0, 0.0, 0.0, 0.0)
        )
        with self.assertRaises(ValueError):
            scales("unknown", 10.0, 0.7, 0.1, 1.0)


class CaptureTest(unittest.TestCase):
    def test_capture_reads_contact_and_defaults_to_zero(self):
        class FakeContext:
            observation = th.zeros(4, OBS_WIDTH)

        class FakeEnv:
            last_car_ball_touches = None

        capture = DIFOTransitionCapture(FakeEnv())
        record = capture(FakeContext())
        self.assertEqual(record["difo_touch_self"].shape, (4,))
        self.assertFalse(bool(record["difo_touch_self"].any()))

        FakeEnv.last_car_ball_touches = th.tensor(
            [[True, False], [False, True]]
        )
        capture = DIFOTransitionCapture(FakeEnv())
        record = capture(FakeContext())
        self.assertTrue(
            th.equal(
                record["difo_touch_self"],
                th.tensor([True, False, False, True]),
            )
        )


class GraphEncoderTest(unittest.TestCase):
    def test_forward_shapes(self):
        encoder = InteractionGraphEncoder(
            N_CARS, feature_size=16, layers=1, include_ego=True
        )
        features = encoder(random_observation(5))
        self.assertEqual(features.shape, (5, 32))
        nested = encoder(random_observation(5).reshape(1, 5, OBS_WIDTH))
        self.assertEqual(nested.shape, (1, 5, 32))
        single = encoder(random_observation(1)[0])
        self.assertEqual(single.shape, (32,))

        pooled = InteractionGraphEncoder(
            N_CARS, feature_size=16, layers=1, include_ego=False
        )
        self.assertEqual(pooled(random_observation(5)).shape, (5, 16))


class PerturbationTest(unittest.TestCase):
    def test_perturbation_leaves_discrete_fields_alone(self):
        graph, _, agent_delta, ball_delta = make_graph_data(4)
        generator = th.Generator().manual_seed(0)
        perturbed = perturb_graph(graph, 0.01, generator)
        self.assertTrue(th.equal(perturbed.node_discrete, graph.node_discrete))
        self.assertTrue(th.equal(perturbed.edge_mask, graph.edge_mask))
        difference = (perturbed.node_cont - graph.node_cont).abs().mean()
        self.assertGreater(float(difference), 0.0)


class ContactTrackingEnvTest(unittest.TestCase):
    def _env(self):
        contacts = th.tensor([[True, False], [False, True]])

        class FakeInner:
            def get_transition_state(self):
                return "capsule"

        class FakeCarl:
            _env = FakeInner()
            n_envs = 4
            n_sim = 2
            n_cars = 2
            device = th.device("cpu")
            single_observation_space = gym.spaces.Box(
                -1.0, 1.0, (OBS_WIDTH,), dtype="float32"
            )
            observation_space = single_observation_space
            single_action_space = gym.spaces.MultiDiscrete([3, 3, 3, 2, 2, 3, 2])
            action_space = single_action_space
            action_codec = None

            def _carl_state(self, capsule):
                return SimpleNamespace(car_ball_touches=contacts)

            def step(self, action):
                return (
                    th.zeros(4, OBS_WIDTH),
                    th.zeros(4),
                    th.zeros(4, dtype=th.bool),
                    th.zeros(4, dtype=th.bool),
                    {},
                )

            def close(self):
                return None

        return ContactTrackingEnv(FakeCarl())

    def test_records_contacts_from_the_last_transition(self):
        env = self._env()
        self.assertIsNone(env.last_car_ball_touches)
        env.step(th.zeros(4, 7))
        self.assertTrue(
            th.equal(
                env.last_car_ball_touches,
                th.tensor([[True, False], [False, True]]),
            )
        )
        capture = DIFOTransitionCapture(env)
        record = capture(SimpleNamespace(observation=th.zeros(4, OBS_WIDTH)))
        self.assertTrue(
            th.equal(
                record["difo_touch_self"],
                th.tensor([True, False, False, True]),
            )
        )


class RawPolicyTest(unittest.TestCase):
    def _env(self):
        class FakeEnv:
            single_observation_space = gym.spaces.Box(
                -1.0, 1.0, (OBS_WIDTH,), dtype="float32"
            )
            single_action_space = gym.spaces.MultiDiscrete([3, 3, 3, 2, 2, 3, 2])
            device = "cpu"
            action_codec = AllValidActionCodec()

        return FakeEnv()

    def test_builds_and_samples_factorized_actions(self):
        env = self._env()
        policy = build_policy(env, 16, [16], {"n_cars": N_CARS, "layers": 1})
        observation = random_observation(6)
        output = policy.act(observation)
        self.assertEqual(output.action.shape, (6, 7))
        self.assertEqual(output.log_prob.shape, (6,))
        evaluation = policy.evaluate_actions(observation, output.action)
        self.assertEqual(evaluation.log_prob.shape, (6,))
        self.assertEqual(evaluation.entropy.shape, (6,))


class NoPulseDependencyTest(unittest.TestCase):
    def test_source_has_no_pulse_or_distill_imports(self):
        source = Path(difo.__file__).read_text()
        self.assertNotIn("from pulse", source)
        self.assertNotIn("import pulse", source)
        self.assertNotIn("from distill", source)
        self.assertNotIn("import distill", source)
        self.assertNotIn("FrozenPulseController", source)
        self.assertNotIn("PulseLatentEnv", source)


if __name__ == "__main__":
    unittest.main()
