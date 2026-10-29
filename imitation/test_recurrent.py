import copy
from types import SimpleNamespace

import torch

from imitation import recurrent
from imitation.dataset import ExpertDataset
from imitation.features import DISCRIMINATOR_WARMUP, ball_distances
from imitation.models import build_discriminator
from imitation.rewards import discriminator_rewards
from imitation.rollout import SceneRollout
from imitation.test_amp import arguments
from imitation.update import gradient_penalty_weights
from replays.layout import team_replay_row_size, team_scene_size


def setup(factorized=True, **overrides):
    args = arguments("--episode-memory", *(["--factorize"] if factorized else []),
                     "--frame-embedding", "4", "--temporal-hidden", "5", "--discriminator-hidden", "6",
                     "--trajectory-length", "3", "--discriminator-batch-size", "12",
                     "--discriminator-update-microbatch-size", "6")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args, build_discriminator(args, 2)


def batch_for(model, count=4, length=7):
    frames = torch.randn(count, length, team_scene_size(1)) * 0.1
    reset = torch.zeros(count, length, dtype=torch.bool)
    reset[:, 0] = True
    valid = torch.ones_like(reset)
    valid[:, :DISCRIMINATOR_WARMUP] = False
    return recurrent.RecurrentBatch(frames, recurrent.zero_hidden(model, count, frames.device), reset, valid)


def test_chunks_match_uninterrupted_episode_and_resets():
    for factorized in (False, True):
        _, model = setup(factorized)
        batch = batch_for(model)
        batch.reset[0, 4] = True
        batch.reset[2, 3] = True
        full, states = recurrent.forward_batch(model, batch)
        first = recurrent.RecurrentBatch(batch.frames[:, :3], batch.hidden, batch.reset[:, :3], batch.valid[:, :3])
        left, hidden = recurrent.forward_batch(model, first)
        second = recurrent.RecurrentBatch(batch.frames[:, 3:], hidden[:, -1].detach(), batch.reset[:, 3:], batch.valid[:, 3:])
        right, _ = recurrent.forward_batch(model, second)
        torch.testing.assert_close(torch.cat((left, right), 1), full)
        fresh = recurrent.RecurrentBatch(batch.frames[:1, 4:], batch.hidden[:1], batch.reset[:1, 4:], batch.valid[:1, 4:])
        torch.testing.assert_close(recurrent.forward_batch(model, fresh)[0], full[:1, 4:])
        changed = copy.deepcopy(batch)
        changed.frames[:, -1] += 10
        torch.testing.assert_close(recurrent.forward_batch(model, changed)[0][:, :-1], full[:, :-1])
        assert states.shape == (4, 7, 3 if factorized else 1, 5)


def test_tbptt_detaches_past_but_retains_memory():
    _, model = setup()
    batch = batch_for(model)
    batch.frames.requires_grad_(True)
    _, states = recurrent.forward_batch(model, batch)
    hidden = states[:, 2].detach()
    later = recurrent.RecurrentBatch(batch.frames[:, 3:], hidden, batch.reset[:, 3:], batch.valid[:, 3:])
    scores, _ = recurrent.forward_batch(model, later)
    scores.sum().backward()
    assert not batch.frames.grad[:, :3].any()
    assert batch.frames.grad[:, 3:].abs().sum() > 0
    fresh = copy.copy(later)
    fresh.hidden = torch.zeros_like(hidden)
    assert not torch.allclose(recurrent.forward_batch(model, fresh)[0], scores)


def test_expert_streams_continue_and_reset_at_segment_boundaries():
    args, model = setup()
    experts = SimpleNamespace(frames=torch.arange(30.)[:, None].expand(-1, team_scene_size(1)),
                              train_episodes=torch.tensor([[3, 8], [15, 20]]),
                              train_generator=torch.Generator().manual_seed(0))
    stream = recurrent.ExpertStreams(experts, model, 4, 3)
    first = stream.sample(4)
    stream.advance(model, first, args)
    second = stream.sample(4)
    torch.testing.assert_close(second.frames[:, 0, 0], first.frames[:, -1, 0] + 1)
    assert not second.reset[:, 0].any()
    assert second.reset[:, 2].all()
    assert not second.valid[:, 2].any()
    assert second.hidden.abs().sum() > 0
    # No frame from either gap or the excluded endpoints can be consumed.
    assert (((second.frames[..., 0] >= 3) & (second.frames[..., 0] < 8)) |
            ((second.frames[..., 0] >= 15) & (second.frames[..., 0] < 20))).all()


def test_history_preserves_seeds_and_masks():
    _, model = setup()
    batch = batch_for(model)
    batch.hidden.uniform_()
    history = recurrent.RecurrentHistory(2, 0, "cpu")
    history.add(batch[:2], 2)
    sampled = history.sample(10)
    for index in range(len(sampled)):
        matches = (batch.frames[:2] == sampled.frames[index]).flatten(1).all(1).nonzero().flatten()
        original = matches.item()
        torch.testing.assert_close(sampled.hidden[index], batch.hidden[original])
        torch.testing.assert_close(sampled.reset[index], batch.reset[original])
        torch.testing.assert_close(sampled.valid[index], batch.valid[original])
    history.add(batch[2:], 2)
    assert history.size == 2
    assert not history.data.hidden.requires_grad


def test_recurrent_penalty_and_training_are_finite():
    for factorized in (False, True):
        args, model = setup(factorized, discriminator_gradient_penalty_samples=2)
        generated, expert = batch_for(model, length=3), batch_for(model, length=3)
        near = ball_distances(expert.frames)[:, -1] <= args.near_distance
        weights = gradient_penalty_weights(near, model.factorized)
        penalty = recurrent.recurrent_penalty(model, expert, weights, torch.full((4,), 2))
        assert penalty.dtype == torch.float32
        penalty.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = recurrent.train_recurrent_batch(model, optimizer, generated, expert, args)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_episode_rewards_match_full_pass_and_terminal_observations():
    args, model = setup(differential=True, multiply_discriminator_reward=True)
    frames = torch.randn(13, 2, team_scene_size(1)) * 0.1
    scenes = SceneRollout(frames[0], 6, 3, team_scene_size(1))
    episode = recurrent.EpisodeRollout(scenes, model, args)
    previous = None
    for iteration in range(2):
        for step in range(6):
            t = iteration * 6 + step
            done = torch.tensor([t == 8, False])
            terminal = frames[t + 1].clone()
            terminal[0] += 1 if t == 8 else 0
            scenes.record(step, frames[t], frames[t + 1], terminal, done)
        scenes.prepare()
        if previous is not None:
            torch.testing.assert_close(episode.hidden, previous)
            # Boundary potentials must be recomputed rather than cached scalars.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.001)
        episode.prepare(model, args)
        torch.testing.assert_close(episode.batch.hidden[2:4], recurrent.forward_batch(model, episode.batch[:2])[1][:, -1])
        batch = recurrent.RecurrentBatch(scenes.scenes.transpose(0, 1), episode.hidden.clone(),
                               (scenes.ages == 0).T, (scenes.ages >= 2).T)
        scores, states = recurrent.forward_batch(model, batch)
        expected = torch.zeros(6, 2)
        for step in range(6):
            successor = recurrent.RecurrentBatch(scenes.successors[step, :, None], states[:, step],
                                       torch.zeros(2, 1, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool))
            next_scores, _ = recurrent.forward_batch(model, successor)
            near = ball_distances(torch.stack((scenes.scenes[step], scenes.successors[step]), 1)) <= args.near_distance
            expected[step] = discriminator_rewards(torch.stack((scores[:, step], next_scores[:, 0]), 1), near, args)
        expected.masked_fill_(scenes.ages < 2, 0)
        actual = episode.rewards(model, args)
        torch.testing.assert_close(actual, expected)
        previous = states[:, -1].detach()
        torch.testing.assert_close(episode.hidden, previous)


def test_recurrent_update_uses_history_and_heldout_streams():
    args, model = setup(discriminator_heldout_size=6, discriminator_gradient_penalty_samples=2)
    frames = torch.randn(7, 4, team_scene_size(1)) * 0.1
    scenes = SceneRollout(frames[0], 6, 3, team_scene_size(1))
    for step in range(6):
        scenes.record(step, frames[step], frames[step + 1], frames[step + 1], torch.zeros(4, dtype=torch.bool))
    scenes.prepare()
    episode = recurrent.EpisodeRollout(scenes, model, args)
    episode.prepare(model, args)
    experts = SimpleNamespace(frames=torch.randn(100, team_scene_size(1)) * 0.1,
                              train_episodes=torch.tensor([[0, 60]]), heldout_episodes=torch.tensor([[60, 100]]),
                              train_generator=torch.Generator().manual_seed(0), heldout_generator=torch.Generator().manual_seed(1))
    stream = recurrent.ExpertStreams(experts, model, 4, 3)
    validation = recurrent.ExpertStreams(experts, model, 2, 3, heldout=True)
    history = recurrent.RecurrentHistory(8, 0, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        metrics = recurrent.update_recurrent(model, optimizer, episode, stream, validation, history,
                                   torch.tensor([False, True]), 2, args, True)
        assert metrics["discriminator/updated"] == 1
        assert metrics["discriminator/minibatches"] == 1
        assert "discriminator/near_heldout_accuracy" in metrics
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert history.size == 8


def test_partial_expert_advance_does_not_change_unused_lanes():
    args, model = setup(discriminator_inference_microbatch_size=6)
    experts = SimpleNamespace(frames=torch.randn(30, team_scene_size(1)),
                              train_episodes=torch.tensor([[0, 30]]),
                              train_generator=torch.Generator().manual_seed(0))
    stream = recurrent.ExpertStreams(experts, model, 4, 3)
    before = stream.hidden[3].clone()
    stream.advance(model, stream.sample(3), args)
    torch.testing.assert_close(stream.hidden[3], before)
    assert stream.position[3] == 0


def test_padding_and_single_frame_chunks_keep_real_hidden_states():
    for length in (1, 4, 8):
        args, model = setup(trajectory_length=length, discriminator_inference_microbatch_size=6)
        frames = torch.randn(6, 2, team_scene_size(1)) * 0.1
        scenes = SceneRollout(frames[0], 5, length, team_scene_size(1))
        for step in range(5):
            scenes.record(step, frames[step], frames[step + 1], frames[step + 1], torch.zeros(2, dtype=torch.bool))
        scenes.prepare()
        episode = recurrent.EpisodeRollout(scenes, model, args)
        episode.prepare(model, args)
        assert episode.batch.valid.sum() == 6
        full = recurrent.RecurrentBatch(frames[:5].transpose(0, 1), episode.hidden.clone(),
                              (scenes.ages == 0).T, (scenes.ages >= 2).T)
        _, trace = recurrent.forward_batch(model, full)
        episode.rewards(model, args)
        torch.testing.assert_close(episode.hidden, trace[:, -1])
        for chunk in range((5 + length - 1) // length):
            if chunk:
                torch.testing.assert_close(episode.batch.hidden[chunk * 2:(chunk + 1) * 2], trace[:, chunk * length - 1])


def test_microbatching_preserves_the_classification_update():
    for factorized in (False, True):
        args, model = setup(factorized, discriminator_gradient_penalty=0)
        small = copy.deepcopy(model)
        generated, expert = batch_for(model, length=3), batch_for(model, length=3)
        generated.valid[0, 1] = True
        expert.valid[2, :] = True
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        small_optimizer = torch.optim.SGD(small.parameters(), lr=0.01)
        args.discriminator_update_microbatch_size = 12
        full = recurrent.train_recurrent_batch(model, optimizer, generated, expert, args)
        args.discriminator_update_microbatch_size = 3
        micro = recurrent.train_recurrent_batch(small, small_optimizer, generated, expert, args)
        torch.testing.assert_close(torch.tensor(full["loss"]), torch.tensor(micro["loss"]))
        for expected, actual in zip(model.parameters(), small.parameters()):
            torch.testing.assert_close(expected, actual)


def test_cpu_bf16_classification_and_fp32_penalty():
    args, model = setup(bf16=True, discriminator_gradient_penalty_samples=2)
    generated, expert = batch_for(model, length=3), batch_for(model, length=3)
    metrics = recurrent.train_recurrent_batch(model, torch.optim.Adam(model.parameters(), lr=1e-3), generated, expert, args)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_expert_episodes_exclude_discontinuities_and_heldout_files(tmp_path):
    import numpy as np

    for index in range(2):
        rows = np.zeros((15, team_replay_row_size(1)), np.float32)
        rows[:, 0] = np.arange(15) + 100 * index
        rows[5, -1] = 1
        rows[10, 0] = np.nan
        np.save(tmp_path / f"{index}.npy", rows)
    experts = ExpertDataset(tmp_path, 1, 3, None, "cpu", heldout_size=1, load_reset_data=False)
    for episodes in (experts.train_episodes, experts.heldout_episodes):
        assert (episodes[:, 1] - episodes[:, 0]).tolist() == [5, 4, 4]
        for start, end in episodes:
            assert torch.isfinite(experts.frames[start:end]).all()
            assert ((torch.arange(start, end) % 15) != 5).all()
    assert (experts.train_episodes[:, 0] // 15).unique().item() != (experts.heldout_episodes[:, 0] // 15).unique().item()
    assert not len(experts.reset_indices)
