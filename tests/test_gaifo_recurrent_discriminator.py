"""Chronological global discriminator context stays causal through training and rewards."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from gaifo import (
    BALL_NEAR_DISTANCE, BLUE_START, GLOBAL_DISCRIMINATOR_WEIGHT, ORANGE_START,
    POSITION_SCALE, SPECIALIST_DISCRIMINATOR_WEIGHT,
    AdaptiveDiscriminatorUpdate, ConfidentExpertResetTransform, ExpertSceneDataset,
    FactorizedSceneDiscriminator, GeneratedContextTimeline, GlobalContextLoss,
    GlobalContextMinibatches,
    SceneDiscriminator, SceneDiscriminatorLoss, SceneDiscriminatorReward,
    generated_context_frames, nearest_ball_distance, simulation_episode_ends,
    train_discriminator_minibatch,
)
from jarl.data import TensorBatch


def write_experts(
    folder: Path, length: int = 12, reject_discontinuities: bool = False,
) -> ExpertSceneDataset:
    for index in range(2):
        rows = np.zeros((length, 161), np.float32)
        rows[:, 0] = (index + 1) / 10 + np.arange(length) / 1_000
        rows[:, 2] = 91.25 / POSITION_SCALE[2]
        for car in (BLUE_START, ORANGE_START):
            rows[:, car + 2] = 17 / POSITION_SCALE[2]
            rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
        if reject_discontinuities:
            rows[length // 2, -1] = 1
        np.save(folder / f"{index}-period.npy", rows)
    return ExpertSceneDataset(
        folder, trajectory_length=2, heldout_size=3,
        reject_discontinuities=reject_discontinuities,
    )


class AccumulatingGlobal(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = th.nn.Parameter(th.tensor(1.0))

    def score_sequence(self, scenes, reset, initial_state=None):
        state = (th.zeros(1, scenes.shape[1], 1) if initial_state is None
                 else initial_state.clone())
        output = []
        for frame, fresh in zip(scenes, reset):
            state = state.masked_fill(fresh[None, :, None], 0) + frame[None, :, :1] * self.gain
            output.append(state[0, :, 0])
        return th.stack(output), state


class AccumulatingFactorized(th.nn.Module):
    factorized = True
    recurrent_global = True

    def __init__(self):
        super().__init__()
        self.global_discriminator = AccumulatingGlobal()
        self.context_version = 0

    def specialist_logits(self, windows):
        return windows.new_zeros((len(windows), 2))


def stepwise_context(model, scenes, ages):
    features = model._encode(scenes)
    state = features.new_zeros((1, len(scenes), model.gru.hidden_size))
    for step in range(scenes.shape[1]):
        _, updated = model.gru(features[:, step:step + 1], state)
        state = th.where((ages >= scenes.shape[1] - step)[None, :, None], updated, state)
    return model.head(state[0]).squeeze(-1)


def stepwise_sequence(model, scenes, reset, initial_state):
    features = model._encode(scenes.transpose(0, 1))
    state = initial_state
    logits = []
    for step in range(len(scenes)):
        state = state.masked_fill(reset[step][None, :, None], 0)
        output, state = model.gru(features[:, step:step + 1], state)
        logits.append(model.head(output[:, 0]).squeeze(-1))
    return th.stack(logits), state


class RecurrentDiscriminatorTests(unittest.TestCase):
    def test_long_context_carries_burnin_state_and_backpropagates_beyond_16_frames(self):
        th.manual_seed(31)
        model = SceneDiscriminator(8, 8, 12, recurrent_global=True)
        scenes = th.randn(3, 96, 51, requires_grad=True)
        ages = th.tensor([96, 77, 9])
        logits = model.score_context(scenes, ages, bptt_length=64)
        th.testing.assert_close(logits, model.score_context(scenes, ages))
        gradients = th.autograd.grad(logits.sum(), scenes)[0]
        th.testing.assert_close(gradients[:2, :32], th.zeros_like(gradients[:2, :32]))
        th.testing.assert_close(gradients[2, :87], th.zeros_like(gradients[2, :87]))
        self.assertGreater(gradients[0, 32].abs().sum().item(), 0)
        self.assertGreater(gradients[1, 32].abs().sum().item(), 0)

    def test_cross_rollout_training_contexts_and_expert_gaps_are_causal(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = write_experts(
                Path(directory), length=256, reject_discontinuities=True,
            )
            for heldout in (False, True):
                starts = (expert.heldout_window_starts if heldout
                          else expert.train_window_starts)
                gaps = (starts[1:] != starts[:-1] + 1).nonzero(as_tuple=True)[0]
                self.assertEqual(len(gaps), 1)
                first_after_gap = starts[gaps[0] + 1]
                context, age = expert.context_frames(
                    th.tensor([[int(first_after_gap), 0]]), 96,
                    heldout=heldout, return_age=True,
                )
                self.assertEqual(age.tolist(), [1])
                th.testing.assert_close(context[0], context[0, :1].expand_as(context[0]))
                late, age = expert.context_frames(
                    th.tensor([[int(starts[-1]), 0]]), 96,
                    heldout=heldout, return_age=True,
                )
                self.assertEqual(age.tolist(), [96])
                self.assertGreater((late[0, -1, 0] - late[0, 0, 0]).item(), .09)

            def windows(start, steps):
                sample = th.zeros(steps, 4, 2, 51)
                times = th.arange(start, start + steps)[:, None, None]
                actors = th.arange(4)[None, :, None] / 10
                sample[..., 0] = .11 + (times + actors) / 1_000
                sample[..., 2] = 91.25 / POSITION_SCALE[2]
                for car in (BLUE_START, ORANGE_START):
                    sample[..., car + 2] = 17 / POSITION_SCALE[2]
                    sample[..., car + 9] = 1
                    sample[..., car + 14] = 1
                    sample[..., car + 16] = 1
                return sample

            previous = windows(0, 72)
            current = windows(72, 40)
            old_ends = th.zeros(72, 4, dtype=th.bool)
            old_ends[70, 3] = True
            timeline = GeneratedContextTimeline(
                current.flatten(0, 1), 4, th.zeros(40, 4, dtype=th.bool),
                previous[:, :, -1], simulation_episode_ends(old_ends),
            )
            chosen = th.tensor([39 * 4, 39 * 4 + 2])
            context, ages = timeline.contexts(chosen, 96)
            self.assertEqual(ages.tolist(), [96, 41])
            th.testing.assert_close(context[0, :, 0], .11 + th.arange(16, 112) / 1_000)
            th.testing.assert_close(context[1, -41:, 0],
                                    .11 + (th.arange(71, 112) + .2) / 1_000)
            th.testing.assert_close(context[1, :55, 0], context[1, :1, 0].expand(55))

            sampler = GlobalContextMinibatches(
                expert, batch_size=8, epochs=1, noise_std=0,
                context_length=96, stride=1,
            )
            sample = next(sampler.sample_contexts(
                current.flatten(0, 1), chosen, 4, timeline=timeline,
            ))
            n = len(sample["is_agent"]) // 2
            self.assertEqual(n, 2)
            th.testing.assert_close(sample["age"][:n], sample["age"][n:])
            self.assertTrue((sample["age"][:n] <= 96).all())
            self.assertTrue(th.isin(sample["window"][:n, -1, 0], context[:, -1, 0]).all())
            self.assertEqual(sampler.batch_size, 8)

    def test_update_retains_cross_rollout_history_when_unscheduled(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = write_experts(Path(directory), length=64)
            model = SceneDiscriminator(8, 8, 8, recurrent_global=True)
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=2, epochs=1,
                noise_std=0, heldout_size=3, accuracy_target=0,
                history_add_size=0, history_mix_fraction=0, max_grad_norm=1,
                discriminator=model, optimizer=th.optim.SGD(model.parameters(), lr=0),
                loss=SceneDiscriminatorLoss(model), update_interval=3,
                microbatch_size=2, context_length=12, context_stride=1,
                context_bptt_length=4,
            )

            def rollout(start, end=False):
                windows = th.zeros(5, 4, 2, 51)
                windows[:, :, :, 0] = .11 + th.arange(start, start + 5)[:, None, None] / 100
                windows[..., 2] = 91.25 / POSITION_SCALE[2]
                for car in (BLUE_START, ORANGE_START):
                    windows[..., car + 2] = 17 / POSITION_SCALE[2]
                    windows[..., car + 9] = 1
                    windows[..., car + 14] = 1
                    windows[..., car + 16] = 1
                terminated = th.zeros(5, 4, dtype=th.bool)
                terminated[3, 1] = end
                return TensorBatch({
                    "scene_window": windows,
                    "scene_window_valid": th.ones(5, 4, dtype=th.bool),
                    "terminated": terminated,
                })

            _, first = update.run(rollout(0))
            self.assertEqual(first["Discriminator"]["updated"], 1)
            _, second = update.run(rollout(5, end=True))
            self.assertEqual(second["Discriminator"]["scheduled"], 0)
            self.assertEqual(len(update._recent_context_frames), 10)

            third = rollout(10)
            timeline = GeneratedContextTimeline(
                third["scene_window"].flatten(0, 1), 4, third["terminated"],
                update._recent_context_frames, update._recent_context_ends,
            )
            context, ages = timeline.contexts(th.tensor([0, 1, 2]), 12)
            self.assertEqual(ages.tolist(), [2, 2, 11])
            th.testing.assert_close(context[2, -11:, 0], .11 + th.arange(11) / 100)
            _, third_metrics = update.run(third)
            self.assertEqual(third_metrics["Discriminator"]["scheduled"], 0)
            self.assertEqual(len(update._recent_context_frames), 11)

    def test_sparse_matched_situations_still_train_global_branch(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = write_experts(Path(directory), length=32)
            windows = th.zeros(8, 2, 2, 51)
            windows[..., 2] = 91.25 / POSITION_SCALE[2]
            for car in (BLUE_START, ORANGE_START):
                windows[..., car + 2] = 17 / POSITION_SCALE[2]
                windows[..., car + 14] = 1
                windows[..., car + 16] = 1
            windows[-1, :, :, 0] = .11  # Only these two endpoints have expert matches.
            sampler = GlobalContextMinibatches(
                expert, batch_size=4, epochs=1, noise_std=0,
                context_length=32, stride=16,
            )
            for seed in range(10):
                th.manual_seed(seed)
                sample = next(sampler.sample_contexts(
                    windows.flatten(0, 1), th.arange(16), 2,
                ))
                self.assertEqual(len(sample["is_agent"]), 2)
                th.testing.assert_close(sample["window"][0, -1, 0], th.tensor(.11))

    def test_packed_context_preserves_padded_gradients(self):
        th.manual_seed(17)
        model = SceneDiscriminator(8, 8, 12, recurrent_global=True)
        scenes = th.randn(5, 6, 51, requires_grad=True)
        ages = th.tensor([1, 2, 3, 5, 6])
        fast = model.score_context(scenes, ages)
        stepwise = stepwise_context(model, scenes, ages)
        th.testing.assert_close(fast, stepwise)

        inputs = (scenes, *model.parameters())
        fast_grads = th.autograd.grad(fast.sum(), inputs, retain_graph=True)
        stepwise_grads = th.autograd.grad(stepwise.sum(), inputs)
        for optimized, original in zip(fast_grads, stepwise_grads):
            th.testing.assert_close(optimized, original, rtol=1e-4, atol=1e-5)
        for row, age in enumerate(ages.tolist()):
            th.testing.assert_close(
                fast_grads[0][row, :6 - age], th.zeros_like(scenes[row, :6 - age]),
            )

    def test_fused_sequence_preserves_individual_resets_and_gradients(self):
        th.manual_seed(18)
        model = SceneDiscriminator(8, 8, 12, recurrent_global=True)
        scenes = th.randn(6, 4, 51, requires_grad=True)
        initial_state = th.randn(1, 4, 8, requires_grad=True)
        for boundaries in ((), ((2, 0), (4, 2))):
            with self.subTest(boundaries=boundaries):
                reset = th.zeros(6, 4, dtype=th.bool)
                reset[0, 1] = True
                for step, actor in boundaries:
                    reset[step, actor] = True
                fast, fast_state = model.score_sequence(scenes, reset, initial_state)
                stepwise, stepwise_state = stepwise_sequence(
                    model, scenes, reset, initial_state,
                )
                th.testing.assert_close(fast, stepwise)
                th.testing.assert_close(fast_state, stepwise_state)
                inputs = (scenes, initial_state, *model.parameters())
                fast_grads = th.autograd.grad(
                    fast.sum() + fast_state.sum(), inputs, retain_graph=True,
                )
                stepwise_grads = th.autograd.grad(
                    stepwise.sum() + stepwise_state.sum(), inputs,
                )
                for optimized, original in zip(fast_grads, stepwise_grads):
                    th.testing.assert_close(optimized, original, rtol=1e-4, atol=1e-5)

    def test_frequent_resets_keep_streaming_batch_bounded(self):
        th.manual_seed(19)
        model = SceneDiscriminator(8, 8, 12, recurrent_global=True)
        scenes = th.randn(20, 4, 51)
        reset = th.ones(20, 4, dtype=th.bool)
        initial = th.randn(1, 4, 8)
        logits, final = model.score_sequence(scenes, reset, initial)
        reference, expected_final = stepwise_sequence(model, scenes, reset, initial)
        th.testing.assert_close(logits, reference)
        th.testing.assert_close(final, expected_final)

    def test_context_matches_stream_and_resets_individual_actors(self):
        th.manual_seed(7)
        model = SceneDiscriminator(8, 8, 12, recurrent_global=True)
        frames = th.randn(5, 4, 51) * .1
        reset = th.zeros(5, 4, dtype=th.bool)
        reset[3, :2] = True
        stream, _ = model.score_sequence(frames, reset)

        windows = frames[:, :, None].expand(-1, -1, 2, -1).reshape(-1, 2, 51)
        ended = th.zeros_like(reset)
        ended[2, 1] = True
        context, ages = generated_context_frames(
            windows, th.arange(20), 4, ended, length=5,
        )
        th.testing.assert_close(model.score_context(context, ages), stream.flatten())
        self.assertEqual(ages.reshape(5, 4)[:, 0].tolist(), [1, 2, 3, 1, 2])
        self.assertEqual(ages.reshape(5, 4)[:, 1].tolist(), [1, 2, 3, 1, 2])
        self.assertEqual(ages.reshape(5, 4)[:, 2].tolist(), [1, 2, 3, 4, 5])
        th.testing.assert_close(
            context[17, :, 0], frames[3:5, 1, 0][th.tensor([0, 0, 0, 0, 1])],
        )

        changed = frames.clone()
        changed[0, :, :9] += 1.0
        new_stream, _ = model.score_sequence(changed, reset)
        self.assertGreater(abs(float((new_stream[-1, 2] - stream[-1, 2]).detach())), 1e-6)
        th.testing.assert_close(new_stream[-1, 0], stream[-1, 0], atol=0, rtol=0)
        th.testing.assert_close(new_stream[-1, 1], stream[-1, 1], atol=0, rtol=0)

    def test_expert_context_stops_at_split_and_sampler_trains_real_sequences(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = write_experts(Path(directory))
            train_start = expert.train_window_starts[0]
            heldout_start = expert.heldout_window_starts[0]
            for start, heldout in ((train_start, False), (heldout_start, True)):
                pairs = th.tensor([[int(start), 0]])
                context, ages = expert.context_frames(
                    pairs, 5, heldout=heldout, return_age=True,
                )
                self.assertEqual(ages.tolist(), [1])
                th.testing.assert_close(context[0], context[0, :1].expand(5, -1))
            with self.assertRaisesRegex(ValueError, "unavailable window"):
                expert.context_frames(th.tensor([[int(heldout_start), 0]]), 5)

            pairs = th.stack((expert.train_window_starts[:6], th.zeros(6, dtype=th.long)), 1)
            generated = expert._windows_for_povs(pairs)
            windows = generated[:, None].expand(-1, 2, -1, -1).reshape(-1, 2, 51)
            ended = th.zeros(6, 2, dtype=th.bool)
            ended[2, 1] = True
            sampler = GlobalContextMinibatches(
                expert, batch_size=12, epochs=1, noise_std=0,
                context_length=4, stride=1,
            )
            sample = next(sampler.sample_contexts(windows, th.arange(12), 2, ended))
            n = len(sample["is_agent"]) // 2
            self.assertEqual(sample["window"].shape, (2 * n, 4, 51))
            self.assertTrue((sample["age"][:n] <= 4).all())
            th.testing.assert_close(sample["age"][:n], sample["age"][n:])
            training_frames = expert.frames[
                expert.train_window_starts + expert.partition_span, 0,
            ]
            self.assertTrue(th.isin(sample["window"][n:, -1, 0], training_frames).all())

            model = SceneDiscriminator(8, 8, 8, recurrent_global=True)
            optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
            metrics = train_discriminator_minibatch(
                sample, model, optimizer, GlobalContextLoss(model), 4, 1.0,
            )
            self.assertTrue(th.isfinite(metrics["global_loss"]))
            self.assertEqual(model.context_version, 1)
            self.assertTrue(th.isfinite(model.gru.weight_ih_l0.grad).all())

            factorized = FactorizedSceneDiscriminator(8, 8, 8, recurrent_global=True)
            optimizer = th.optim.Adam(factorized.parameters(), lr=1e-3)
            train_discriminator_minibatch(
                sample, factorized, optimizer, GlobalContextLoss(factorized), 4, 1.0,
            )
            self.assertEqual(factorized.context_version, 1)
            self.assertIsNotNone(factorized.global_discriminator.gru.weight_ih_l0.grad)
            self.assertIsNone(factorized.car_gru.weight_ih_l0.grad)
            self.assertIsNone(factorized.near_discriminator.gru.weight_ih_l0.grad)

            short = TensorBatch({
                "window": th.cat((generated, expert.sample(len(generated), "cpu"))),
                "is_agent": th.cat((th.ones(len(generated)), th.zeros(len(generated)))),
            })
            train_discriminator_minibatch(
                short, factorized, optimizer,
                SceneDiscriminatorLoss(factorized, specialists_only=True), 4, 1.0,
            )
            self.assertEqual(factorized.context_version, 1)
            self.assertIsNone(factorized.global_discriminator.gru.weight_ih_l0.grad)
            self.assertIsNotNone(factorized.car_gru.weight_ih_l0.grad)
            self.assertIsNotNone(factorized.near_discriminator.gru.weight_ih_l0.grad)

    def test_mined_resets_use_the_same_recurrent_context_as_expert_training(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = write_experts(Path(directory))
            th.manual_seed(11)
            for model in (
                SceneDiscriminator(8, 8, 8, recurrent_global=True),
                FactorizedSceneDiscriminator(8, 8, 8, recurrent_global=True),
            ):
                with self.subTest(factorized=getattr(model, "factorized", False)):
                    miner = ConfidentExpertResetTransform(
                        expert, expert.reset_dataset(), model, 2, context_length=4,
                    )
                    starts = miner.candidate_starts[-3:]
                    windows = expert.frames[starts[:, None] + expert.window_offsets]
                    context, ages = expert.context_frames(
                        th.stack((starts, th.zeros_like(starts)), dim=-1),
                        4, return_age=True,
                    )
                    with th.inference_mode():
                        global_model = (
                            model.global_discriminator if getattr(model, "factorized", False)
                            else model
                        )
                        logits = global_model.score_context(context, ages)
                        if getattr(model, "factorized", False):
                            near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
                            specialists = model.specialist_logits(windows)
                            selected = th.where(near, specialists[:, 1], specialists[:, 0])
                            logits = (GLOBAL_DISCRIMINATOR_WEIGHT * logits
                                      + SPECIALIST_DISCRIMINATOR_WEIGHT * selected)
                        th.testing.assert_close(miner._score(starts), th.sigmoid(-logits))

    def test_reward_carries_state_and_rebuilds_it_after_weight_update(self):
        model = AccumulatingFactorized()
        reward = SceneDiscriminatorReward(
            model, noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True, context_length=3, max_magnitude=100,
            batch_size=1,
        )

        def windows(values):
            scene = th.zeros(len(values), 4, 2, 51)
            scene[:, :, -1, 0] = th.tensor(values)
            return scene

        valid = th.ones(2, 4, dtype=th.bool)
        warmup = valid.clone()
        warmup[0, 2] = False
        ending = th.zeros_like(valid)
        ending[-1, 1] = True
        first = reward._score_windows(
            windows([[.1, .1, .1, .1], [.2, .2, .2, .2]]), warmup, ending,
        )
        th.testing.assert_close(first[-1, :, 2], th.exp(th.tensor([-.3, -.3, -.3, -.3])))

        second = reward._score_windows(
            windows([[.3, .4, .3, .4], [.4, .5, .4, .5]]),
            valid, th.zeros_like(valid),
        )
        th.testing.assert_close(second[0, :, 2], th.exp(th.tensor([-.3, -.4, -.6, -.7])))
        th.testing.assert_close(second[1, :, 2], th.exp(th.tensor([-.7, -.9, -1., -1.2])))

        model.global_discriminator.gain.data.fill_(2)
        model.context_version += 1
        later = reward._score_windows(
            windows([[.5, .1, .5, .1], [.6, .2, .6, .2]]),
            valid, th.zeros_like(valid),
        )
        th.testing.assert_close(later[0, :, 2], th.exp(th.tensor([-2.4, -2., -2.8, -2.4])))

    def test_reward_rebuilds_long_memory_after_training_over_short_rollouts(self):
        model = AccumulatingFactorized()
        reward = SceneDiscriminatorReward(
            model, noise_std=0, trajectory_length=2, context_length=32,
            exp_log_odds_reward=True, max_magnitude=100, batch_size=2,
        )
        valid = th.ones(8, 4, dtype=th.bool)
        ended = th.zeros_like(valid)
        for _ in range(4):
            windows = th.zeros(8, 4, 2, 51)
            windows[:, :, -1, 0] = .01
            reward._score_windows(windows, valid, ended)

        model.global_discriminator.gain.data.fill_(2)
        model.context_version += 1
        next_windows = th.zeros(1, 4, 2, 51)
        next_windows[:, :, -1, 0] = .02
        logits = reward._score_windows(
            next_windows, th.ones(1, 4, dtype=th.bool),
            th.zeros(1, 4, dtype=th.bool),
        )
        th.testing.assert_close(logits[0, :, 2], th.exp(th.full((4,), -.68)))


if __name__ == "__main__":
    unittest.main()
