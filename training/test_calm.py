"""CALM cadence, conditioning, gradient ownership, boundaries and persistence."""

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from calm import (
    CalmAgent,
    CalmDataset,
    ConditionalDiscriminator,
    ReferenceConditioning,
    ReferenceContext,
    ReferencePositionIndex,
    add_calm_arguments,
    calm_ppo_loss,
    calm_rewards,
    collect_calm,
    conditional_discriminator_loss,
    position_features,
    resolve_calm_defaults,
    update_calm_discriminator,
)
from replays.layout import team_replay_row_size
from training.checkpoints import load_checkpoint, save_checkpoint
from training.discriminator import balanced_window_batch
from training.imitation import EpisodeImitationStatistics, TransitionRollout
from training.ppo import ppo_loss, update_ppo
from training.rollout import Rollout, compute_gae


def arguments(*extra):
    parser = argparse.ArgumentParser()
    add_calm_arguments(parser)
    args = parser.parse_args([
        "--n-sim", "2", "--hidden-size", "16", "--layers", "1",
        "--calm-latent-size", "4", "--calm-encoder-hidden", "16",
        "--frame-embedding", "8", "--discriminator-hidden", "16",
        "--rollout", "6", "--minibatch-size", "8", "--update-epochs", "1",
        "--discriminator-batch-size", "8", "--discriminator-update-microbatch-size", "4",
        "--discriminator-epochs", "1", "--calm-regularization-batch-size", "4",
        "--replay-reset-fraction", "0", *extra,
    ])
    resolve_calm_defaults(args)
    args.goal_reward_weight = 0.
    return args


class Codec:
    def mask(self, observations):
        # Appending a latent to simulator observations would break action masking.
        assert observations.shape[-1] == 64
        result = torch.ones((len(observations), 14), dtype=torch.bool, device=observations.device)
        result[:, 1] = False
        return result


def agent_for(args):
    return CalmAgent(64, [2] * 7, Codec(), args.hidden_size, args.layers, args=args, n_cars=2)


def write_replay(path, count=160, offset=0):
    rows = np.zeros((count, team_replay_row_size(1)), dtype=np.float32)
    rows[:, :51] = np.random.default_rng(31).normal(0, .1, (count, 51))
    rows[:, 0] = offset + np.arange(count) * .001
    rows[:, -2:] = 0
    np.save(path, rows)
    return rows


class FakeEnv:
    device = torch.device("cpu")
    n_envs = 4
    _team_sign = torch.tensor([[1., -1.]])

    def __init__(self):
        self.events = SimpleNamespace(score_delta=torch.zeros(2))
        self.steps = 0

    def step(self, actions):
        self.steps += 1
        observation = torch.full((4, 64), self.steps * .01)
        final = observation.clone()
        terminated = torch.tensor([self.steps == 2, False, False, False])
        truncated = torch.tensor([False, self.steps == 3, False, False])
        observation[terminated | truncated] = 0
        return observation, torch.zeros(4), terminated, truncated, {"final_obs": final}


class CalmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_two_second_cadence_and_reference_boundaries(self):
        for frameskip, length in ((4, 61), (8, 31), (12, 21)):
            with self.subTest(frameskip=frameskip), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "one.npy"
                rows = write_replay(path)
                rows[75, -1] = 1  # Correction at a skipped phase is still excluded.
                np.save(path, rows)
                write_replay(Path(directory) / "two.npy", offset=10)
                path.with_suffix(".json").write_text(json.dumps({"frameskip": 4, "goal_terminal": True}))
                args = arguments("--frameskip", str(frameskip), "--replay-dir", directory)
                dataset = CalmDataset(args, torch.device("cpu"))
                self.assertEqual(args.calm_window_length, length)
                windows = dataset.references(dataset.train_ends)
                torch.testing.assert_close(windows[:, -1, 0] - windows[:, 0, 0], torch.full((len(windows),), .06))
                self.assertFalse(((dataset.train_ends >= 75) & (dataset.train_ends <= 135)).any())
                self.assertNotIn(159, dataset.train_ends.tolist())
                first, near, independent = dataset.regularization_samples(64)
                self.assertLessEqual((first[:, 0, 0] - near[:, 0, 0]).abs().max(), frameskip / 4 * .001 + 1e-6)
                self.assertTrue(torch.isfinite(independent).all())
                ends = dataset.sample_ends(64)
                local = dataset.local_windows(ends, 5, 1)
                reference = dataset.references(ends)
                self.assertTrue((local[:, 0, 0] >= reference[:, 0, 0]).all())
                self.assertTrue((local[:, -1, 0] <= reference[:, -1, 0]).all())
        with self.assertRaisesRegex(ValueError, "integer number"):
            arguments("--frameskip", "28")
        with self.assertRaisesRegex(ValueError, "fit inside"):
            arguments("--discriminator-window-length", "32")

    def test_encoder_gradient_ownership_and_conditioned_action_masks(self):
        args = arguments()
        agent = agent_for(args)
        reference = torch.randn(8, args.calm_window_length, 51)
        observation = torch.randn(8, 64)
        latent = agent.actor.encoder(reference)
        torch.testing.assert_close(latent.norm(dim=-1), torch.ones(8))
        with torch.no_grad():
            actions, old_log, _, old_values = agent.get_action_and_value(observation, reference=reference)
        self.assertTrue((actions[:, 0] == 0).all())
        loss, _ = ppo_loss(agent, observation, actions, old_log, torch.arange(8).float(),
                           old_values + .5, old_values, .2, torch.zeros(7), 0., False, conditioning=reference)
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in agent.actor.encoder.parameters()), 0)
        agent.zero_grad(set_to_none=True)
        agent.get_value(observation, reference=reference).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in agent.actor.encoder.parameters()))
        agent.zero_grad(set_to_none=True)
        model = ConditionalDiscriminator(args, 2)
        generated, expert = torch.randn(8, 5, 51), torch.randn(8, 5, 51)
        generated[0, :4] = float("nan")
        loss, penalty = conditional_discriminator_loss(model, generated, expert, agent.actor.encoder(reference), 5.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(penalty.item(), 0)
        self.assertTrue(all(p.grad is None for p in agent.actor.encoder.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(torch.equal(model(expert, latent.detach()), model(expert, -latent.detach())))

    def test_compilable_penalty_matches_autograd_loss_and_parameter_gradients(self):
        for flags in ([], ["--discriminator-relative"], ["--discriminator-no-flags", "--discriminator-no-boost"]):
            for coefficient in (0., 5.):
                with self.subTest(flags=flags, coefficient=coefficient):
                    args = arguments(*flags)
                    model = ConditionalDiscriminator(args, 2).double()
                    baseline = copy.deepcopy(model)
                    generated, expert = torch.randn(4, 5, 51, dtype=torch.float64), torch.randn(4, 5, 51, dtype=torch.float64)
                    generated[0, :4] = float("nan")
                    expert[1, :2] = float("nan")
                    latent = F.normalize(torch.randn(4, args.calm_latent_size, dtype=torch.float64), dim=-1)
                    negative = -latent
                    actual, actual_penalty = conditional_discriminator_loss(model, generated, expert, latent, coefficient, negative)
                    fake_windows, real_windows = balanced_window_batch(generated, expert).chunk(2)
                    real_windows = real_windows.detach().requires_grad_(True)
                    condition = latent.detach().clone().requires_grad_(True)
                    real, fake, mismatch = baseline(real_windows, condition), baseline(fake_windows, condition), baseline(real_windows, negative)
                    expected = F.softplus(real).mean() + .5 * (F.softplus(-fake).mean() + F.softplus(-mismatch).mean())
                    penalty = expected.new_zeros(())
                    if coefficient:
                        gradients = torch.autograd.grad(real.sum(), (real_windows, condition), create_graph=True)
                        penalty = sum(g.flatten(1).square().sum(-1).mean() for g in gradients)
                        expected = expected + coefficient * penalty
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(actual_penalty, penalty, rtol=1e-10, atol=1e-10)
                    actual.backward()
                    expected.backward()
                    for a, b in zip(model.parameters(), baseline.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)

    def test_feature_cache_keeps_encoder_trainable_and_matches_uncached_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            write_replay(Path(directory) / "one.npy")
            args = arguments("--replay-dir", directory, "--discriminator-relative")
            experts = CalmDataset(args, torch.device("cpu"))
            ends = experts.sample_ends(3).repeat(4).reshape(3, 4)
            encoder = agent_for(args).actor.encoder
            baseline = copy.deepcopy(encoder)
            for max_bytes in (0, 512 * 1024**2):
                cache = ReferenceConditioning(experts, ends, args, max_bytes=max_bytes)
                indices = torch.randperm(ends.numel())[:8]
                features, _ = cache(indices)
                encoder.zero_grad(set_to_none=True)
                baseline.zero_grad(set_to_none=True)
                actual = encoder(features)
                expected = baseline(experts.references(ends.flatten()[indices]))
                torch.testing.assert_close(actual, expected)
                actual[:, 0].sum().backward()
                expected[:, 0].sum().backward()
                for a, b in zip(encoder.parameters(), baseline.parameters()):
                    torch.testing.assert_close(a.grad, b.grad)
                with torch.no_grad():
                    encoder.network[-1].bias.add_(.1)
                    baseline.network[-1].bias.add_(.1)
                self.assertFalse(torch.equal(encoder(features), actual))
                torch.testing.assert_close(encoder(features), baseline(experts.references(ends.flatten()[indices])))

    def test_position_matching_uses_reference_start_and_retains_rng_control(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = write_replay(Path(directory) / "one.npy")
            # Three distinct ego+ball configurations at starts 0, 10 and 20.
            # Their reference END states are deliberately shuffled.
            rows[[0, 10, 20], :3] = np.array([[0, 0, 0], [.5, .5, .5], [-.5, -.5, .1]])
            rows[[0, 10, 20], 9:12] = np.array([[.5, .2, .1], [-.4, .6, .2], [0, 0, 0]])
            rows[[60, 70, 80], :51] = rows[[20, 0, 10], :51]
            np.save(Path(directory) / "one.npy", rows)
            args = arguments("--replay-dir", directory, "--calm-match-topk", "1", "--calm-match-backend", "sampled")
            experts = CalmDataset(args, torch.device("cpu"))
            observations = torch.from_numpy(rows[[10, 0, 20], :51].copy())
            candidates = torch.tensor([60, 70, 80])
            with patch.object(experts, "sample_ends", return_value=candidates):
                torch.testing.assert_close(experts.select_ends(observations, args), torch.tensor([70, 60, 80]))
                args.calm_match_topk = 3
                state = experts.state_dict()
                first = experts.select_ends(observations, args)
                experts.load_state_dict(state)
                torch.testing.assert_close(first, experts.select_ends(observations, args))
                args.calm_reference_sampling = "uniform"
                torch.testing.assert_close(experts.select_ends(observations, args), candidates)

    def test_precomputed_index_matches_global_brute_force_and_handles_ties(self):
        torch.manual_seed(73)
        frames = torch.randn(300, 51)
        ends = torch.arange(60, 300, 2)
        index = ReferencePositionIndex(frames, ends, -60, workers=1)
        queries = torch.cat((frames[ends[:3] - 60], torch.randn(9, 51) * 10))
        points = position_features(frames[ends - 60]).double()
        distances = (position_features(queries).double()[:, None] - points[None]).square().sum(-1)
        for k in (1, 16, len(ends) + 10):
            actual = index.nearest_ends(queries, k)
            expected = ends[distances.topk(min(k, len(ends)), largest=False).indices]
            torch.testing.assert_close(actual, expected)
        self.assertEqual(index.nearest_ends(queries[:0], 16).shape, (0, 16))
        # Identical configurations still return distinct, eligible windows.
        frames.zero_()
        tied = ReferencePositionIndex(frames, ends, -60, workers=1).nearest_ends(queries[:1] * 0, 16)
        self.assertEqual(tied.unique().numel(), 16)
        self.assertTrue(torch.isin(tied, ends).all())

    def test_indexed_selection_uses_all_eligible_windows_and_restores_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            write_replay(Path(directory) / "one.npy")
            args = arguments("--replay-dir", directory, "--calm-match-candidates", "16")
            experts = CalmDataset(args, torch.device("cpu"))
            self.assertEqual(len(experts.position_index.ends), len(experts.train_ends))
            queries = experts.frames[experts.train_ends[::10] + experts.offsets[0]]
            expected = experts.position_index.nearest_ends(queries, args.calm_match_topk)
            state = experts.state_dict()
            with patch.object(experts, "sample_ends", side_effect=AssertionError("sampled search used")):
                first = experts.select_ends(queries, args)
                experts.load_state_dict(state)
                torch.testing.assert_close(first, experts.select_ends(queries, args))
            self.assertTrue((first[:, None] == expected).any(-1).all())
            args.calm_match_topk = 1
            torch.testing.assert_close(experts.select_ends(queries, args), experts.train_ends[::10])

    def test_rollout_switches_padding_timeouts_and_complete_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            write_replay(Path(directory) / "one.npy")
            args = arguments("--replay-dir", directory, "--calm-min-hold-seconds", ".06666666666666667",
                             "--calm-max-hold-seconds", ".06666666666666667")
            experts = CalmDataset(args, torch.device("cpu"))
            agent = agent_for(args)
            optimizer = torch.optim.Adam(agent.parameters(), lr=.001)
            env = FakeEnv()
            observation = torch.zeros(4, 64)
            rollout = Rollout.allocate(6, 4, 64, 7, env.device)
            scenes = TransitionRollout(observation, 6, 51, 5)
            context = ReferenceContext(experts, 4, args, torch.compile(agent.actor.encoder, backend="eager"))
            context.resample(agent, torch.ones(4, dtype=torch.bool), observation)
            ends = torch.empty((6, 4), dtype=torch.long)
            latents = torch.empty(6, 4, 4)
            observation = collect_calm(env, agent, agent.get_action_and_value, rollout, observation,
                                       args, scenes, context, ends, latents)
            self.assertEqual(rollout.dones.sum().item(), 2)
            self.assertTrue(scenes.dones.all())  # Every decision changed reference.
            windows = scenes.transitions(torch.arange(24))
            self.assertTrue(torch.isnan(windows[:, :-2]).all())
            self.assertTrue(torch.isfinite(windows[:, -2:]).all())
            torch.testing.assert_close(latents.flatten(0, 1), agent.actor.encoder(experts.references(ends.flatten())))
            final = torch.full((4, 64), .03)
            expected_timeout = agent.get_value(final, latent=latents[2])[1]
            torch.testing.assert_close(rollout.timeout_values[2, 1], expected_timeout)
            model = ConditionalDiscriminator(args, 2)
            target = copy.deepcopy(model).requires_grad_(False)
            discriminator_optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            metrics = update_calm_discriminator(model, discriminator_optimizer, experts, scenes, rollout,
                                                ends, latents, args, target)
            self.assertGreater(metrics["discriminator/minibatches"], 0)
            calm_rewards(target, scenes, rollout, latents, args, EpisodeImitationStatistics(4, env.device))
            self.assertEqual(rollout.rewards[1, 0].item(), 0.)
            with torch.no_grad():
                value = agent.get_value(observation, latent=context.latents)
            compute_gae(rollout, value, .99, .95)
            old_encoder = [p.detach().clone() for p in agent.actor.encoder.parameters()]

            def loss(*values, conditioning):
                return calm_ppo_loss(*values, conditioning=conditioning, regularization_weight=.1)

            metrics = update_ppo(agent, optimizer, loss, rollout, args,
                                 conditioning=lambda indices: (experts.references(ends.flatten()[indices]),
                                                               experts.regularization_samples(4)))
            self.assertTrue(all(math_value == math_value for math_value in metrics.values()))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old_encoder, agent.actor.encoder.parameters())))

    def test_checkpoint_restores_encoder_optimizer_rng_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments()
            args.calm_dataset_signature = "fixture"
            agent = agent_for(args)
            optimizer = torch.optim.Adam(agent.parameters(), lr=.001)
            reference = torch.randn(4, args.calm_window_length, 51)
            agent.actor.encoder(reference)[:, 0].sum().backward()
            optimizer.step()
            expected = agent.actor.encoder(reference).detach().clone()
            path = Path(directory) / "calm.pt"
            save_checkpoint(path, "calm", args, agent, optimizer, 24)
            expected_random = torch.rand(4)
            for parameter in agent.parameters():
                parameter.data.zero_()
            payload = load_checkpoint(path, "calm", args, agent, optimizer)
            self.assertEqual(payload["step"], 24)
            torch.testing.assert_close(torch.rand(4), expected_random)
            torch.testing.assert_close(agent.actor.encoder(reference), expected)
            self.assertTrue(optimizer.state)
            resumed = arguments("--resume", str(path))
            self.assertEqual(resumed.calm_window_length, 31)
            self.assertEqual(resumed.calm_match_backend, "indexed")
            legacy = torch.load(path, weights_only=False)
            legacy["config"].pop("calm_match_backend")
            torch.save(legacy, path)
            self.assertEqual(arguments("--resume", str(path)).calm_match_backend, "sampled")
            with self.assertWarnsRegex(UserWarning, "reference matching backend changed"):
                load_checkpoint(path, "calm", args, agent, optimizer)
            save_checkpoint(path, "calm", args, agent, optimizer, 24)
            args.calm_reference_sampling = "uniform"
            with self.assertRaisesRegex(ValueError, "calm_reference_sampling"):
                load_checkpoint(path, "calm", args, agent, optimizer)
            args.calm_reference_sampling = "state-matched"
            args.calm_window_seconds = 3
            with self.assertRaisesRegex(ValueError, "CALM configuration differs"):
                load_checkpoint(path, "calm", args, agent, optimizer)


if __name__ == "__main__":
    unittest.main()
