import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from ppo import (
    add_ppo_arguments,
    autocast,
    collect,
    compile_functions,
    load_checkpoint,
    log_metrics,
    mlp,
    plain_tensor,
    prepare,
    run_directory,
    save_checkpoint,
    update_ppo
)
from replays import PARSED_DIR
from replays.layout import (
    BALL_SIZE,
    CAR_SIZE,
    team_replay_row_size,
    team_scene_size
)

POSITION_SCALE = (4108.0, 6000.0, 2076.0)


def pose_basis(forward, up):
    x = forward.new_tensor((1.0, 0.0, 0.0)).expand_as(forward)
    z = up.new_tensor((0.0, 0.0, 1.0)).expand_as(up)
    forward = F.normalize(
        torch.where(forward.norm(dim=-1, keepdim=True) > 1e-6, forward, x), dim=-1
    )
    up = up - (up * forward).sum(dim=-1, keepdim=True) * forward
    fallback = torch.where(forward[..., 2:3].abs() < 0.9, z, x)
    fallback = fallback - (fallback * forward).sum(dim=-1, keepdim=True) * forward
    up = F.normalize(torch.where(up.norm(dim=-1, keepdim=True) > 1e-6, up, fallback), dim=-1)

    return torch.stack((forward, torch.linalg.cross(up, forward, dim=-1), up), dim=-1)


def initial_frame_trajectory(windows, n_cars):
    ball = windows[..., :BALL_SIZE]
    cars = windows[..., BALL_SIZE:].reshape(*windows.shape[:2], n_cars, CAR_SIZE)

    basis = pose_basis(cars[:, 0, 0, 9:12], cars[:, 0, 0, 12:15])
    scale = windows.new_tensor(POSITION_SCALE)
    origin = cars[:, :1, 0, :3] * scale

    local = windows.clone()
    local_ball = local[..., :BALL_SIZE]
    local_cars = local[..., BALL_SIZE:].reshape_as(cars)
    local_ball[..., :3] = torch.einsum("btj,bjk->btk", ball[..., :3] * scale - origin, basis) / 6000
    local_cars[..., :3] = (
        torch.einsum(
            "btnj,bjk->btnk",
            cars[..., :3] * scale - origin[:, :, None],
            basis,
        )
        / 6000
    )

    for start in (3, 6):
        local_ball[..., start : start + 3] = torch.einsum(
            "btj,bjk->btk",
            ball[..., start : start + 3],
            basis,
        )

    for start in (3, 6, 9, 12):
        local_cars[..., start : start + 3] = torch.einsum(
            "btnj,bjk->btnk",
            cars[..., start : start + 3],
            basis,
        )

    motion = local.clone()
    motion[..., :BALL_SIZE] -= local[:, :1, :BALL_SIZE]
    motion_cars = motion[..., BALL_SIZE:].reshape_as(cars)
    motion_cars[..., :9] -= local_cars[:, :1, :, :9]
    car_basis = pose_basis(cars[:, 0, :, 9:12], cars[:, 0, :, 12:15])

    for start in (9, 12):
        motion_cars[..., start : start + 3] = torch.einsum(
            "btnj,bnjk->btnk",
            cars[..., start : start + 3],
            car_basis,
        )

    return local, motion


class DiscriminatorHead(nn.Module):

    def __init__(self, length, n_cars, hidden_size, mode):
        super().__init__()

        self.mode = mode
        indices = {
            "far": range(BALL_SIZE, BALL_SIZE + CAR_SIZE),
            "near": range(BALL_SIZE + CAR_SIZE),
            "global": range(BALL_SIZE + n_cars * CAR_SIZE),
        }[mode]

        self.register_buffer("indices", torch.tensor(list(indices)), persistent=False)
        width = len(self.indices) * (length + 2) + (6 if mode == "far" else 0)
        self.network = mlp(width, 1, hidden_size, 2, 1.0)

    def forward(self, windows, local, motion):
        features = [
            windows[:, 0, self.indices],
            local[:, 0, self.indices],
            motion[..., self.indices].flatten(1),
        ]

        if self.mode == "far":
            features.append(
                torch.cat(
                    (
                        local[:, 0, :3] - local[:, 0, BALL_SIZE : BALL_SIZE + 3],
                        local[:, 0, 3:6]
                        - local[:, 0, BALL_SIZE + 3 : BALL_SIZE + 6] * (2300 / 6000),
                    ),
                    dim=-1,
                )
            )

        return self.network(torch.cat(features, dim=-1)).squeeze(-1).float()


class Discriminator(nn.Module):

    def __init__(self, length, n_cars, hidden_size=256, factorize=False, near_distance=1000.0):
        super().__init__()

        self.n_cars = n_cars
        self.near_distance = near_distance
        modes = ("far", "near", "global") if factorize else ("global",)
        self.heads = nn.ModuleDict(
            {mode: DiscriminatorHead(length, n_cars, hidden_size, mode) for mode in modes}
        )

    def features(self, windows):
        with torch.autocast(windows.device.type, enabled=False):
            return initial_frame_trajectory(windows.float(), self.n_cars)

    def weights(self, windows):
        if len(self.heads) == 1:
            return torch.ones((len(windows), 1), device=windows.device)

        distance = (
            (
                (windows[..., :3] - windows[..., BALL_SIZE : BALL_SIZE + 3])
                * windows.new_tensor(POSITION_SCALE)
            )
            .norm(dim=-1)
            .amin(dim=-1)
        )
        near = (distance < self.near_distance).float()

        return torch.stack((1 - near, near, torch.ones_like(near)), dim=-1) * 0.5

    def forward(self, windows):
        local, motion = self.features(windows)

        return torch.stack([head(windows, local, motion) for head in self.heads.values()], dim=-1)


class ExpertDataset:

    def __init__(self, directory, team_size, length, frame_limit, device):
        scene_size = team_scene_size(team_size)
        chunks, starts = [], []
        offset = 0

        for path in sorted(Path(directory).rglob("*.npy")):
            rows = np.load(path, mmap_mode="r", allow_pickle=False)

            if rows.ndim != 2 or rows.shape[1] != team_replay_row_size(team_size):
                continue

            remaining = frame_limit - offset if frame_limit else len(rows)
            count = min(len(rows), remaining)

            if count < length:
                continue

            scene = np.array(rows[:count, :scene_size], dtype=np.float32)
            # Correction/discontinuity flags belong to the state they enter.
            bad = (
                (~np.isfinite(scene).all(axis=-1))
                | (rows[:count, -1] > 0.5)
                | (rows[:count, -2] > 0.5)
            )
            cumulative = np.concatenate(([0], np.cumsum(bad)))
            candidates = np.arange(count - length + 1)
            valid = cumulative[candidates + length] == cumulative[candidates]

            if valid.any():
                chunks.append(scene)
                starts.append(candidates[valid] + offset)
                offset += count

            if frame_limit and offset >= frame_limit:
                break

        if not starts:
            raise ValueError(
                "no valid expert windows; check replay directory, schema, and window length"
            )

        self.frames = torch.as_tensor(np.concatenate(chunks), device=device)
        self.starts = torch.as_tensor(np.concatenate(starts), device=device)
        self.offsets = torch.arange(length, device=device)

    def sample(self, count):
        choices = torch.randint(len(self.starts), (count,), device=self.starts.device)
        indices = self.starts[choices, None] + self.offsets

        return self.frames[indices]


class SceneRollout:

    def __init__(self, observation, steps, length, scene_size):
        agents = len(observation)
        self.length = length
        self.scene_size = scene_size
        self.scenes = observation.new_empty((steps, agents, scene_size))
        self.successors = torch.empty_like(self.scenes)
        self.ages = torch.empty((steps, agents), dtype=torch.long, device=observation.device)
        self.age = torch.zeros(agents, dtype=torch.long, device=observation.device)
        self.history = observation[None, :, :scene_size].expand(length - 2, -1, -1).clone()
        self.timeline = None
        self.valid_indices = None

    def record(self, step, observation, next_observation, final_observation, done):
        self.scenes[step].copy_(observation[:, : self.scene_size])
        successor = torch.where(
            done[:, None],
            final_observation[:, : self.scene_size],
            next_observation[:, : self.scene_size],
        )
        self.successors[step].copy_(successor)
        self.ages[step].copy_(self.age)
        self.age.add_(1).masked_fill_(done, 0)

    def prepare(self):
        self.timeline = torch.cat((self.history, self.scenes))
        self.valid_indices = (self.ages.flatten() >= self.length - 2).nonzero().flatten()

        if self.length > 2:
            self.history.copy_(self.timeline[-(self.length - 2) :])

    def windows(self, indices):
        agents = self.scenes.shape[1]
        step = indices // agents
        agent = indices % agents
        end = step + self.length - 2
        start = end - self.ages[step, agent]
        offsets = torch.arange(2 - self.length, 1, device=indices.device)
        frames = torch.maximum(end[:, None] + offsets, start[:, None])
        prefix = self.timeline[frames, agent[:, None]]

        return torch.cat((prefix, self.successors[step, agent, None]), dim=1)

    def sample(self, count):
        choices = torch.randint(len(self.valid_indices), (count,), device=self.scenes.device)

        return self.windows(self.valid_indices[choices])


@torch.no_grad()
def imitation_rewards(discriminator, scenes, rollout, args):
    indices = torch.arange(rollout.rewards.numel(), device=rollout.rewards.device)
    environment_rewards = rollout.rewards.flatten()
    total = torch.zeros((), device=indices.device)

    for batch in indices.split(args.discriminator_microbatch_size):
        windows = scenes.windows(batch)

        with autocast(indices.device, args.bf16):
            logits = discriminator(windows)

        scores = logits if args.log_odds_reward else F.softplus(logits)
        rewards = (scores * discriminator.weights(windows)).sum(-1)
        total.add_(rewards.sum())
        environment_rewards[batch] = (
            args.environment_reward_coef * environment_rewards[batch]
            + args.imitation_reward_coef * rewards
        )

    return (total / len(indices)).item()


def discriminator_loss(head, windows, local, motion, labels, weights):
    logits = head(windows, local, motion)
    errors = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    return (errors * weights).sum() / weights.sum().clamp_min(1e-8)


def update_discriminator(discriminator, optimizers, loss_function, experts, scenes, args):
    device = scenes.scenes.device

    if not len(scenes.valid_indices):
        return {}

    metrics = {}

    for _ in range(args.discriminator_updates):
        generated = scenes.sample(args.discriminator_batch_size)
        expert = experts.sample(args.discriminator_batch_size)
        windows = torch.cat((generated, expert))
        labels = torch.cat(
            (torch.zeros(len(generated), device=device), torch.ones(len(expert), device=device))
        )

        with torch.no_grad(), autocast(device, args.bf16):
            local, motion = discriminator.features(windows)
            weights = discriminator.weights(windows)
            logits = torch.cat(
                [
                    discriminator(batch)
                    for batch in windows.split(args.discriminator_microbatch_size)
                ]
            )
            correct = ((logits > 0) == labels[:, None].bool()).float()

            # Balance classes independently within each specialist's eligible pool.
            for group in (slice(0, len(generated)), slice(len(generated), None)):
                weights[group] /= weights[group].sum(0, keepdim=True).clamp_min(1e-8)

            denominator = weights.sum(0)
            accuracy = (correct * weights).sum(0) / denominator.clamp_min(1)

        active = (
            ((accuracy < args.discriminator_accuracy_target) & (denominator > 1.5)).cpu().tolist()
        )
        accuracies = accuracy.cpu().tolist()

        for index, (name, head) in enumerate(discriminator.heads.items()):
            metrics[f"discriminator/{name}_accuracy"] = accuracies[index]
            metrics[f"discriminator/{name}_updated"] = float(active[index])

            # No optimizer step means no momentum or weight-decay drift for gated heads.
            if not active[index]:
                continue

            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)

            for start in range(0, len(windows), args.discriminator_microbatch_size):
                stop = min(start + args.discriminator_microbatch_size, len(windows))
                batch = slice(start, stop)
                torch.compiler.cudagraph_mark_step_begin()

                with autocast(device, args.bf16):
                    loss = loss_function(
                        head,
                        windows[batch],
                        local[batch],
                        motion[batch],
                        labels[batch],
                        weights[batch, index],
                    )
                    # Microbatches contribute their share of the full balanced batch.
                    fraction = weights[batch, index].sum() / denominator[index].clamp_min(1)
                    loss = loss * fraction

                loss.backward()
                total_loss.add_(loss.detach())

            nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics[f"discriminator/{name}_loss"] = total_loss.item()

    return metrics


def add_gaifo_arguments(parser):
    group = parser.add_argument_group("GAIFO")
    group.add_argument("--replay-dir", type=Path, default=PARSED_DIR)
    group.add_argument("--expert-frame-limit", type=int, default=200_000)
    group.add_argument("--trajectory-length", type=int, default=8)
    group.add_argument("--discriminator-hidden", type=int, default=256)
    group.add_argument("--discriminator-batch-size", type=int, default=65_536)
    group.add_argument("--discriminator-microbatch-size", type=int, default=16_384)
    group.add_argument("--discriminator-updates", type=int, default=1)
    group.add_argument("--discriminator-every", type=int, default=5)
    group.add_argument("--discriminator-learning-rate", type=float, default=1e-4)
    group.add_argument("--discriminator-accuracy-target", type=float, default=0.85)
    group.add_argument("--factorize", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--near-distance", type=float, default=1000.0)
    group.add_argument("--log-odds-reward", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--imitation-reward-coef", type=float, default=1.0)
    group.add_argument("--environment-reward-coef", type=float, default=1.0)


def train(args):
    env, agent, optimizer, rollout = prepare(args)
    writer = None

    try:
        discriminator = Discriminator(
            args.trajectory_length,
            env.n_cars,
            args.discriminator_hidden,
            args.factorize,
            args.near_distance,
        ).to(env.device)
        optimizers = {
            name: torch.optim.Adam(
                head.parameters(), lr=args.discriminator_learning_rate, fused=True
            )
            for name, head in discriminator.heads.items()
        }
        experts = ExpertDataset(
            args.replay_dir,
            args.team_size,
            args.trajectory_length,
            args.expert_frame_limit,
            env.device,
        )
        step = iteration = 0

        if args.resume:
            payload = load_checkpoint(args.resume, "gaifo", args, agent, optimizer)
            discriminator.load_state_dict(payload["discriminator"])

            for name, state in payload["discriminator_optimizers"].items():
                optimizers[name].load_state_dict(state)

            step, iteration = payload["step"], payload["iteration"]

        policy, loss_function, gae_function = compile_functions(agent, args)
        disc_loss = discriminator_loss

        if args.compile:
            disc_loss = torch.compile(disc_loss, fullgraph=True)

        directory = run_directory(args, "gaifo")
        writer = SummaryWriter(directory)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        scenes = SceneRollout(
            observation, args.rollout, args.trajectory_length, team_scene_size(args.team_size)
        )

        while step < args.timesteps:
            started = time.perf_counter()
            progress = min(step / args.timesteps, 1.0)
            optimizer.param_groups[0]["lr"] = args.learning_rate + progress * (
                args.learning_rate_end - args.learning_rate
            )
            observation, episode_returns = collect(
                env, agent, policy, rollout, observation, args, scenes
            )
            scenes.prepare()
            metrics = {"environment_reward": rollout.rewards.mean().item()}
            # Score on-policy data before fitting the discriminator to this rollout.
            metrics["imitation_reward"] = imitation_rewards(discriminator, scenes, rollout, args)

            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation)

            gae_function(rollout, next_value, args.gamma, args.gae_lambda)
            metrics.update(update_ppo(agent, optimizer, loss_function, rollout, args))
            iteration += 1

            if iteration % args.discriminator_every == 0:
                metrics.update(
                    update_discriminator(
                        discriminator, optimizers, disc_loss, experts, scenes, args
                    )
                )

            step += rollout.rewards.numel()
            torch.cuda.synchronize(env.device)
            metrics["transitions_per_second"] = rollout.rewards.numel() / (
                time.perf_counter() - started
            )

            if episode_returns:
                metrics["episode_return"] = float(np.mean(episode_returns))

            log_metrics(writer, step, metrics)

            if iteration % args.save_every == 0 or step >= args.timesteps:
                save_checkpoint(
                    directory / "latest.pt",
                    "gaifo",
                    args,
                    agent,
                    optimizer,
                    step,
                    iteration=iteration,
                    discriminator=discriminator.state_dict(),
                    discriminator_optimizers={
                        name: opt.state_dict() for name, opt in optimizers.items()
                    },
                )
    finally:
        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean windowed GAIFO with PPO for CARL.")
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    args = parser.parse_args()
    train(args)
