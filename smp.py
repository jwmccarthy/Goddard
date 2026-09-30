"""Score-Matching Motion Priors for 1v1 Rocket League via CARL and JARL.

The expert-only prior follows Mu et al., SMP (https://arxiv.org/abs/2512.03028).
The learner-score correction follows Wu et al., SMILING
(https://arxiv.org/abs/2410.13855). Both models denoise entire joint-scene
windows; neither uses a binary expert/agent discriminator.
"""

import argparse
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from jarl.collect import (
    CriticCapture,
    LogProbCapture,
    RecurrentCriticCapture,
    RecurrentStateCapture,
    Runner,
)
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.runtime import Clock, OnPolicySchedule, Trainer
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import GAE, PrepareContext

from gaifo import (
    AdvancedTouchCapture,
    BALL_MAX_SPEED,
    BALL_RADIUS,
    BLUE_START,
    CAR_MAX_SPEED,
    CAR_SIZE,
    ExpertSceneDataset,
    GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE,
    GameplayDiagnostics,
    GOAL_HEIGHT,
    N_CARS,
    ORANGE_START,
    POSITION_SCALE,
    SCENE_SIZE,
    SelectPPOFields,
    build_critic,
    build_entropy_scheduler,
    build_env,
    build_policy,
    build_ppo_sampler,
    noise_mask,
    opponent_view,
)
from difo import DIFOSceneWindowCapture
from replay_resets import load_demonstration_reset_dataset


SMP_REPRESENTATION = "joint-1v1-scene-continuous-score-v1"


class SceneScoreModel(nn.Module):
    """Predict Gaussian noise on full joint ball/car motion clips.

    Car flags are discrete: keep them clean as conditions and score only the
    normalized continuous features. Relative ball/car position and velocity
    features make physical interactions visible to the temporal encoder.
    """

    def __init__(
        self, trajectory_length: int, hidden_size: int = 128,
        diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        if trajectory_length < 2 or hidden_size < 4 or hidden_size % 4:
            raise ValueError("score hidden size must be divisible by four and trajectory length >= 2")
        if diffusion_steps < 4:
            raise ValueError("diffusion steps must be at least four")
        self.trajectory_length = trajectory_length
        self.hidden_size = hidden_size
        self.diffusion_steps = diffusion_steps

        mask = noise_mask()
        self.register_buffer("continuous_mask", mask)
        betas = th.linspace(1e-4, 2e-2, diffusion_steps)
        alpha = (1.0 - betas).cumprod(0)
        self.register_buffer("sqrt_alpha", alpha.sqrt())
        self.register_buffer("sqrt_one_minus_alpha", (1.0 - alpha).sqrt())

        self.frame_encoder = nn.Sequential(
            nn.Linear(SCENE_SIZE + 15, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.position = nn.Parameter(th.zeros(1, trajectory_length, hidden_size))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=2 * hidden_size,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, SCENE_SIZE)
        )

    def _check_windows(self, windows: th.Tensor) -> None:
        if windows.ndim != 3 or windows.shape[1:] != (
            self.trajectory_length, SCENE_SIZE
        ):
            raise ValueError("joint-scene windows have the wrong shape")

    def diffuse(
        self, windows: th.Tensor, timesteps: th.Tensor,
        noise: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        self._check_windows(windows)
        if timesteps.shape != (len(windows),) or (
            (timesteps < 0).any() or (timesteps >= self.diffusion_steps).any()
        ):
            raise ValueError("invalid diffusion timesteps")
        if noise is None:
            noise = th.randn_like(windows)
        if noise.shape != windows.shape:
            raise ValueError("diffusion noise must match scene windows")
        mask = self.continuous_mask.view(1, 1, SCENE_SIZE)
        noise = noise * mask
        noised = (
            self.sqrt_alpha[timesteps, None, None] * windows
            + self.sqrt_one_minus_alpha[timesteps, None, None] * noise
        )
        return th.where(mask, noised, windows), noise

    def predict_noise(self, noisy: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        self._check_windows(noisy)
        if timesteps.shape != (len(noisy),):
            raise ValueError("diffusion timesteps must match scene windows")
        ball = noisy[..., :9]
        ego = noisy[..., BLUE_START:BLUE_START + CAR_SIZE]
        opponent = noisy[..., ORANGE_START:ORANGE_START + CAR_SIZE]
        speed_ratio = CAR_MAX_SPEED / BALL_MAX_SPEED
        relative = th.cat((
            ball[..., :3] - ego[..., :3],
            ball[..., :3] - opponent[..., :3],
            ego[..., :3] - opponent[..., :3],
            ball[..., 3:6] - ego[..., 3:6] * speed_ratio,
            ball[..., 3:6] - opponent[..., 3:6] * speed_ratio,
        ), dim=-1)
        half = self.hidden_size // 2
        frequencies = th.exp(
            -math.log(10_000.0)
            * th.arange(half, device=noisy.device, dtype=th.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies
        time = self.time_encoder(th.cat((angles.sin(), angles.cos()), dim=-1))
        features = self.frame_encoder(th.cat((noisy, relative), dim=-1))
        features = self.temporal(features + self.position + time[:, None])
        return self.output(features) * self.continuous_mask

    def denoising_error(
        self, noisy: th.Tensor, timesteps: th.Tensor, noise: th.Tensor,
    ) -> th.Tensor:
        if noise.shape != noisy.shape:
            raise ValueError("diffusion target must match scene windows")
        predicted = self.predict_noise(noisy, timesteps)
        residual = (predicted - noise) * self.continuous_mask
        return residual.square().sum(dim=(-1, -2)) / (
            self.continuous_mask.sum() * self.trajectory_length
        )

    def loss(self, windows: th.Tensor) -> th.Tensor:
        timesteps = th.randint(
            self.diffusion_steps, (len(windows),), device=windows.device
        )
        noisy, noise = self.diffuse(windows, timesteps)
        return self.denoising_error(noisy, timesteps, noise).mean()


class ModeBalancedExpertWindows:
    """Mix natural replay frequencies with touches, aerial and contest clips."""

    def __init__(self, expert: ExpertSceneDataset, seed: int) -> None:
        self.expert = expert
        self.generator = th.Generator(device=expert.frames.device).manual_seed(seed)
        starts = expert.train_window_starts
        scale = th.tensor(POSITION_SCALE, device=expert.frames.device)
        contact_prefix = (
            F.pad(expert.contact_frames.long().cumsum(0), (1, 0))
            if expert.contact_frames is not None else None
        )
        by_mode: list[list[th.Tensor]] = [[], [], [], [], []]
        # Parsed replay corpora can be very large; don't materialize every
        # 51-feature scene a second time solely to calculate mode indices.
        for offset in range(0, len(starts), 65_536):
            chunk = starts[offset:offset + 65_536]
            last = expert.frames[chunk + expert.trajectory_length - 1]
            ball = last[:, :3] * scale
            ego = last[:, BLUE_START:BLUE_START + 3] * scale
            opponent = last[:, ORANGE_START:ORANGE_START + 3] * scale
            ego_distance = th.linalg.vector_norm(ball - ego, dim=-1)
            opponent_distance = th.linalg.vector_norm(ball - opponent, dim=-1)
            nearby = th.minimum(ego_distance, opponent_distance) < 350.0
            aerial = (ball[:, 2] > GOAL_HEIGHT) | (
                (ball[:, 2] > 2 * BALL_RADIUS)
                & (last[:, BLUE_START + 16] < 0.5)
                & (ego_distance < 800.0)
            ) | (
                (ball[:, 2] > 2 * BALL_RADIUS)
                & (last[:, ORANGE_START + 16] < 0.5)
                & (opponent_distance < 800.0)
            )
            contest = (ego_distance < 900.0) & (opponent_distance < 900.0)
            modes = th.zeros(len(chunk), dtype=th.long, device=last.device)
            modes[nearby] = 1
            modes[aerial] = 2
            modes[contest] = 3
            if contact_prefix is not None:
                modes[
                    contact_prefix[chunk + expert.trajectory_length] > contact_prefix[chunk]
                ] = 4
            for mode in range(5):
                by_mode[mode].append(chunk[modes == mode])
        self.mode_starts = [th.cat(group) if group else starts[:0] for group in by_mode]
        self.nonempty = [mode for mode in range(4) if len(self.mode_starts[mode])]

    def sample(self, count: int) -> th.Tensor:
        if count < 1:
            raise ValueError("expert sample count must be positive")
        starts = self.expert.train_window_starts
        device = starts.device
        n_canonical = (count + 1) // 2
        n_uniform = (n_canonical + 1) // 2
        selected = [starts[th.randint(
            len(starts), (n_uniform,), device=device, generator=self.generator
        )]]
        n_balanced = n_canonical - n_uniform
        if n_balanced:
            assignments = th.randint(
                len(self.nonempty), (n_balanced,), device=device,
                generator=self.generator,
            )
            for index, mode in enumerate(self.nonempty):
                n = int((assignments == index).sum().item())
                if n:
                    candidates = self.mode_starts[mode]
                    selected.append(candidates[th.randint(
                        len(candidates), (n,), device=device, generator=self.generator
                    )])
        offsets = self.expert.window_offsets
        canonical = self.expert.frames[th.cat(selected)[:, None] + offsets]
        return th.stack((canonical, opponent_view(canonical)), dim=1).flatten(0, 1)[:count]


class MixtureWindowBuffer:
    """CPU reservoir over prior policy windows, retaining older behavior modes."""

    def __init__(self, capacity: int, trajectory_length: int, seed: int = 0) -> None:
        if capacity < 1 or trajectory_length < 2:
            raise ValueError("invalid policy window reservoir size")
        self.capacity = capacity
        self.trajectory_length = trajectory_length
        self.generator = th.Generator().manual_seed(seed)
        self.windows: th.Tensor | None = None
        self.priorities = th.empty(0, dtype=th.float64)

    @property
    def size(self) -> int:
        return len(self.priorities)

    def add(self, windows: th.Tensor, limit: int) -> None:
        if windows.shape[1:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("policy reservoir windows have the wrong shape")
        if limit < 1 or not len(windows):
            return
        if len(windows) > limit:
            selected = th.randperm(len(windows), device=windows.device)[:limit]
            windows = windows[selected]
        incoming = windows.detach().to(device="cpu", dtype=th.float32)
        priorities = th.rand(len(incoming), generator=self.generator, dtype=th.float64)
        combined = incoming if self.windows is None else th.cat((self.windows, incoming))
        combined_priorities = th.cat((self.priorities, priorities))
        kept = combined_priorities.topk(min(self.capacity, len(combined))).indices
        self.windows = combined[kept]
        self.priorities = combined_priorities[kept]

    def sample(self, count: int, device: str | th.device) -> th.Tensor:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty policy reservoir")
        assert self.windows is not None
        selected = th.randint(self.size, (count,), generator=self.generator)
        return self.windows[selected].to(device)


def score_optimizer_step(
    model: SceneScoreModel, optimizer: th.optim.Optimizer,
    windows: th.Tensor, microbatch: int, max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(windows), microbatch):
        chunk = windows[start:start + microbatch]
        loss = model.loss(chunk)
        (loss * len(chunk) / len(windows)).backward()
        total += loss.detach().item() * len(chunk) / len(windows)
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return total


@th.no_grad()
def calibrate_prior(
    model: SceneScoreModel, expert: ExpertSceneDataset,
    timesteps: list[int], sample_size: int, microbatch: int,
) -> th.Tensor:
    model.eval()
    device = expert.frames.device
    sample = (
        expert.sample_heldout if expert.heldout_total else expert.sample
    )(sample_size, device)
    normalizers = []
    for timestep in timesteps:
        total = th.zeros((), device=device)
        for start in range(0, len(sample), microbatch):
            chunk = sample[start:start + microbatch]
            indices = th.full((len(chunk),), timestep, device=device, dtype=th.long)
            noisy, noise = model.diffuse(chunk, indices)
            total += model.denoising_error(noisy, indices, noise).sum()
        normalizers.append((total / len(sample)).clamp_min(0.05))
    return th.stack(normalizers)


class SMPReward:
    """Ensemble SMP reward minus the SMILING score-difference cost.

    A full physical 1v1 scene is shared by both agents. Average both ego views
    into one shared prior reward rather than rewarding one team's imitation at
    the other team's expense; goals and touch bonuses remain zero-sum.
    """

    def __init__(
        self, expert: SceneScoreModel, agent: SceneScoreModel,
        timesteps: list[int], normalizers: th.Tensor,
        trajectory_length: int, batch_size: int = 512,
        prior_scale: float = 1.0, prior_weight: float = 1.0,
        contrast_weight: float = 0.25, contrast_clip: float = 2.0,
        goal_reward_weight: float = 1.0,
        aerial_touch_reward_weight: float = 0.5,
        flip_reset_reward_weight: float = 1.0,
    ) -> None:
        self.expert = expert
        self.agent = agent
        self.timesteps = timesteps
        self.normalizers = normalizers
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.prior_scale = prior_scale
        self.prior_weight = prior_weight
        self.contrast_weight = contrast_weight
        self.contrast_clip = contrast_clip
        self.goal_reward_weight = goal_reward_weight
        self.aerial_touch_reward_weight = aerial_touch_reward_weight
        self.flip_reset_reward_weight = flip_reset_reward_weight
        self.agent_ready = False
        self.last_metrics: dict[str, float] = {}
        if normalizers.shape != (len(timesteps),) or (
            ~th.isfinite(normalizers) | (normalizers <= 0)
        ).any():
            raise ValueError("SMP normalization must be finite and positive at each timestep")

    @th.no_grad()
    def _score(self, windows: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Score a batch of [pair, view, time, scene] with common noise."""
        count = len(windows)
        expert_total = th.zeros(count, device=windows.device)
        cost_total = th.zeros_like(expert_total)
        for index, timestep in enumerate(self.timesteps):
            t = th.full((count * N_CARS,), timestep, device=windows.device, dtype=th.long)
            base_noise = th.randn_like(windows[:, 0])
            # The opposing POV is a rotation/permutation of the same scene.
            # Mirrored perturbations reduce avoidable variance in pair rewards.
            noise = th.stack((base_noise, opponent_view(base_noise)), dim=1)
            noisy, target = self.expert.diffuse(
                windows.flatten(0, 1), t, noise.flatten(0, 1)
            )
            expert_loss = self.expert.denoising_error(
                noisy, t, target
            ).view(count, N_CARS).mean(-1)
            expert_total += expert_loss / self.normalizers[index]
            if self.agent_ready and self.contrast_weight:
                agent_loss = self.agent.denoising_error(
                    noisy, t, target
                ).view(count, N_CARS).mean(-1)
                cost_total += expert_loss - agent_loss
        return expert_total / len(self.timesteps), cost_total / len(self.timesteps)

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.ndim != 4 or windows.shape[1] % N_CARS or (
            windows.shape[:2] != valid.shape
            or windows.shape[2:] != (self.trajectory_length, SCENE_SIZE)
        ):
            raise ValueError("SMP rollout needs paired 1v1 scene windows")
        steps, actors = valid.shape
        pairs = actors // N_CARS
        pair_valid = valid.reshape(steps, pairs, N_CARS).all(-1)
        valid = pair_valid.repeat_interleave(N_CARS, dim=1)
        imitation = th.zeros_like(batch["reward"])
        smp_mean = cost_mean = contrast_mean = count = 0.0
        score_was_training = self.agent.training
        self.expert.eval()
        self.agent.eval()
        try:
            flat_pairs = windows.reshape(-1, N_CARS, self.trajectory_length, SCENE_SIZE)
            indices = th.nonzero(pair_valid.flatten(), as_tuple=False).squeeze(-1)
            flat_reward = imitation.reshape(-1, N_CARS)
            for start in range(0, len(indices), self.batch_size):
                selected = indices[start:start + self.batch_size]
                prior_loss, cost = self._score(flat_pairs[selected])
                smp = (-self.prior_scale * prior_loss).exp()
                contrast = (-cost).clamp(-self.contrast_clip, self.contrast_clip)
                flat_reward[selected] = (
                    self.prior_weight * (smp + self.contrast_weight * contrast)
                )[:, None]
                smp_mean += smp.sum().item()
                cost_mean += cost.sum().item()
                contrast_mean += contrast.sum().item()
                count += len(selected)
        finally:
            self.agent.train(score_was_training)
        self.last_metrics = {
            "prior": smp_mean / max(1.0, count),
            "cost": cost_mean / max(1.0, count),
            "contrast_bonus": contrast_mean / max(1.0, count),
            "valid_fraction": count / (steps * pairs),
        }
        goal = batch["reward"] * self.goal_reward_weight
        aerial = batch["aerial_touch_score"] * self.aerial_touch_reward_weight
        flip = batch["flip_reset_event"] * self.flip_reset_reward_weight
        for name, value in (("goal", goal), ("aerial", aerial), ("flip", flip)):
            if value.shape != imitation.shape:
                raise ValueError(f"{name} event reward must match the rollout")
        result = batch.with_fields(
            imitation_reward=imitation, goal_reward=goal,
            aerial_touch_reward=aerial, flip_reset_reward=flip,
            training_reward=imitation + goal + aerial + flip,
        )
        mask = valid | goal.ne(0) | aerial.ne(0) | flip.ne(0)
        if "learner_mask" in result:
            return result.replace_fields(learner_mask=result["learner_mask"].bool() & mask)
        return result.with_fields(learner_mask=mask)


class AgentScoreUpdate:
    """Fit g_pi on an accumulated mixture *after* PPO uses the prior scores."""

    def __init__(
        self, model: SceneScoreModel, optimizer: th.optim.Optimizer,
        reservoir: MixtureWindowBuffer, reward: SMPReward,
        updates: int, batch_size: int, microbatch: int, add_size: int,
        interval: int, max_grad_norm: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.reservoir = reservoir
        self.reward = reward
        self.updates = updates
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.add_size = add_size
        self.interval = interval
        self.max_grad_norm = max_grad_norm
        self.rollouts = 0

    def set_progress_callback(self, callback) -> None:
        return

    def run(self, experience: Rollout | TensorBatch):
        batch = experience.steps if isinstance(experience, Rollout) else experience
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        pair_valid = valid.reshape(valid.shape[0], -1, N_CARS).all(-1)
        indices = th.nonzero(
            pair_valid.repeat_interleave(N_CARS, dim=1).flatten(),
            as_tuple=False,
        ).squeeze(-1)
        if len(indices) == 0:
            return experience, {"AgentScore": {"loss": 0.0, "updates": 0.0, "windows": float(self.reservoir.size)}}
        selected = indices[
            th.randperm(len(indices), device=indices.device)[:self.add_size]
        ]
        flat = windows.reshape(-1, self.model.trajectory_length, SCENE_SIZE)
        self.reservoir.add(flat[selected], self.add_size)
        self.rollouts += 1
        if self.rollouts % self.interval:
            return experience, {"AgentScore": {"loss": 0.0, "updates": 0.0, "windows": float(self.reservoir.size)}}

        total = 0.0
        for _ in range(self.updates):
            sample = self.reservoir.sample(self.batch_size, flat.device)
            total += score_optimizer_step(
                self.model, self.optimizer, sample,
                self.microbatch, self.max_grad_norm,
            )
        self.reward.agent_ready = True
        return experience, {"AgentScore": {
            "loss": total / self.updates,
            "updates": float(self.updates),
            "windows": float(self.reservoir.size),
        }}


def atomic_save(payload: dict, path: Path) -> None:
    """Leave previous completed checkpoints intact if a write fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        th.save(payload, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(
            f"cannot save {path}; check free space and quotas on {path.parent}"
        ) from error


def prior_config(args: argparse.Namespace) -> dict:
    return {
        "algorithm": "smp_prior",
        "representation": SMP_REPRESENTATION,
        "frameskip": args.frameskip,
        "trajectory_length": args.trajectory_length,
        "score_hidden": args.score_hidden,
        "diffusion_steps": args.diffusion_steps,
        "score_timesteps": args.score_timesteps,
    }


def load_prior_checkpoint(path: Path, args: argparse.Namespace) -> dict:
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("config") != prior_config(args):
        raise ValueError(f"incompatible SMP expert prior in {path}")
    if "expert_prior" not in payload or "normalizers" not in payload:
        raise ValueError(f"SMP prior is missing weights or normalization in {path}")
    normalizers = payload["normalizers"]
    if not isinstance(normalizers, th.Tensor) or normalizers.shape != (len(args.score_timesteps),) or (
        ~th.isfinite(normalizers) | (normalizers <= 0)
    ).any():
        raise ValueError(f"invalid SMP prior normalization in {path}")
    return payload


def load_prior_training_checkpoint(path: Path, args: argparse.Namespace) -> dict:
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("config") != prior_config(args) or (
        payload.get("seed") != args.seed
        or payload.get("expert_frame_limit") != args.expert_frame_limit
        or payload.get("prior_ema") != args.prior_ema
    ):
        raise ValueError(f"incompatible SMP prior pretraining checkpoint in {path}")
    if payload.get("replay_dir") != str(args.replay_dir.resolve()):
        raise ValueError(f"SMP prior pretraining requires the original replays in {path}")
    step = payload.get("updates")
    if type(step) is not int or not 0 < step < args.prior_updates:
        raise ValueError("--prior-updates must exceed the saved prior update count")
    for name in (
        "expert_prior", "ema_prior", "optimizer", "sampler_rng_state", "torch_rng_state",
    ):
        if name not in payload:
            raise ValueError(f"SMP prior pretraining checkpoint is missing {name}")
    return payload


def pretrain_prior(
    args: argparse.Namespace, run_id: str, device: th.device,
) -> tuple[SceneScoreModel, th.Tensor, Path]:
    expert = ExpertSceneDataset(
        args.replay_dir, args.trajectory_length, args.expert_frame_limit,
        args.seed, frame_skip=args.frameskip, device=device,
        heldout_size=args.prior_heldout_size, reject_discontinuities=True,
    )
    if expert.train_total < 1:
        raise ValueError("no physics-continuous expert replay windows for SMP pretraining")
    sampler = ModeBalancedExpertWindows(expert, args.seed + 1)
    model = SceneScoreModel(
        args.trajectory_length, args.score_hidden, args.diffusion_steps
    ).to(device)
    ema = SceneScoreModel(
        args.trajectory_length, args.score_hidden, args.diffusion_steps
    ).to(device).eval().requires_grad_(False)
    ema.load_state_dict(model.state_dict())
    optimizer = th.optim.Adam(model.parameters(), lr=args.score_lr)
    path = args.prior_output or args.checkpoint_dir / "priors" / f"prior_{run_id}.pt"
    training_path = path.with_suffix(".training.pt")
    completed = 0
    if args.resume_prior is not None:
        state = load_prior_training_checkpoint(args.resume_prior, args)
        model.load_state_dict(state["expert_prior"])
        ema.load_state_dict(state["ema_prior"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.score_lr
        sampler.generator.set_state(state["sampler_rng_state"])
        th.set_rng_state(state["torch_rng_state"].cpu())
        if "cuda_rng_state" in state:
            th.cuda.set_rng_state_all(state["cuda_rng_state"])
        completed = state["updates"]
        print(f"Resuming SMP prior {args.resume_prior} from update {completed:,}")
    print(
        f"Pretraining SMP on {expert.train_total:,} expert windows "
        f"({expert.heldout_total:,} held out)"
    )
    for update in range(completed + 1, args.prior_updates + 1):
        batch = sampler.sample(args.prior_batch)
        loss = score_optimizer_step(
            model, optimizer, batch, args.prior_microbatch, args.max_grad_norm
        )
        with th.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, 1.0 - args.prior_ema)
        if update % args.prior_eval_interval == 0 or update == args.prior_updates:
            if expert.heldout_total:
                with th.no_grad():
                    heldout = expert.sample_heldout(
                        min(args.prior_microbatch, args.prior_heldout_size), device
                    )
                    heldout_loss = ema.loss(heldout).item()
                print(
                    f"SMP prior {update:,}/{args.prior_updates:,}: "
                    f"train MSE {loss:.4f}, heldout MSE {heldout_loss:.4f}"
                )
            else:
                print(
                    f"SMP prior {update:,}/{args.prior_updates:,}: "
                    f"train MSE {loss:.4f}"
                )
        if update % args.prior_save_interval == 0 and update < args.prior_updates:
            atomic_save({
                "config": prior_config(args),
                "replay_dir": str(args.replay_dir.resolve()),
                "seed": args.seed,
                "expert_frame_limit": args.expert_frame_limit,
                "prior_ema": args.prior_ema,
                "updates": update,
                "expert_prior": model.state_dict(),
                "ema_prior": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler_rng_state": sampler.generator.get_state(),
                "torch_rng_state": th.get_rng_state(),
                "cuda_rng_state": th.cuda.get_rng_state_all(),
            }, training_path)

    normalizers = calibrate_prior(
        ema, expert, args.score_timesteps,
        args.prior_calibration_size, args.prior_microbatch,
    )
    atomic_save({
        "config": prior_config(args),
        "updates": args.prior_updates,
        "ema_decay": args.prior_ema,
        "expert_prior": ema.state_dict(),
        "normalizers": normalizers.cpu(),
    }, path)
    training_path.unlink(missing_ok=True)
    print(f"SMP prior saved: {path}")
    return ema, normalizers, path


class SMPCheckpoints:
    """Atomic policy and score checkpoints; the large reservoir is rebuilt on resume."""

    def __init__(
        self, directory: Path, interval: int, keep: int,
        modules: dict[str, nn.Module],
        optimizers: dict[str, th.optim.Optimizer],
        reward: SMPReward, agent_update: AgentScoreUpdate,
        buffer: RolloutBuffer, args: argparse.Namespace,
    ) -> None:
        self.directory = Path(directory)
        self.interval = interval
        self.keep = keep
        self.modules = modules
        self.optimizers = optimizers
        self.reward = reward
        self.agent_update = agent_update
        self.buffer = buffer
        self.args = args
        self.step = 0
        self.next_step = 0
        self.clock: Clock | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("smp_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return self.buffer.position == 0 and step >= self.next_step

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            **{name: module.state_dict() for name, module in self.modules.items()},
            **{
                f"{name}_optimizer": optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            "normalizers": self.reward.normalizers.cpu(),
            "agent_ready": self.reward.agent_ready,
            "agent_rollouts": self.agent_update.rollouts,
            "config": {
                "architecture": (
                    GAIFO_GRU_ARCHITECTURE if self.args.gru else GAIFO_ARCHITECTURE
                ),
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
                "algorithm": "smp",
                "representation": SMP_REPRESENTATION,
            },
        }
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
            payload["torch_rng_state"] = th.get_rng_state()
            payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.directory / f"smp_{step:012d}.pt"
        atomic_save(payload, path)
        paths = sorted(self.directory.glob("smp_*.pt"))
        for old in paths[:-self.keep]:
            old.unlink()
        self.next_step = step + self.interval


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"SMP checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid SMP checkpoint in {path}")
    config = payload["config"]
    if config.get("algorithm") != "smp" or config.get("representation") != SMP_REPRESENTATION:
        raise ValueError(f"incompatible SMP checkpoint in {path}")
    arch = config.get("architecture")
    if arch not in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE) or (
        config.get("gru") != (arch == GAIFO_GRU_ARCHITECTURE)
    ):
        raise ValueError(f"invalid SMP policy architecture in {path}")
    for name in ("n_sim", "rollout"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"invalid SMP {name} in {path}")
    levels = config.get("score_timesteps")
    if not isinstance(levels, list) or not levels or any(
        type(level) is not int for level in levels
    ):
        raise ValueError(f"invalid SMP score timesteps in {path}")
    step = payload.get("step")
    if type(step) is not int or step < 0 or (
        step % (config["n_sim"] * N_CARS)
    ):
        raise ValueError(f"invalid SMP checkpoint step in {path}")
    required = (
        "policy", "critic", "expert_prior", "agent_score", "policy_optimizer",
        "critic_optimizer", "agent_score_optimizer", "normalizers", "agent_ready",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"SMP checkpoint is missing {', '.join(missing)} in {path}")
    normalizers = payload["normalizers"]
    if not isinstance(normalizers, th.Tensor) or normalizers.shape != (len(levels),) or (
        ~th.isfinite(normalizers) | (normalizers <= 0)
    ).any():
        raise ValueError(f"invalid SMP score normalization in {path}")
    if "clock" in payload:
        try:
            clock = Clock(**payload["clock"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid SMP checkpoint clock in {path}") from error
        if clock.env_steps != step:
            raise ValueError(f"SMP checkpoint clock differs from step {step} in {path}")
    return payload


def validate_resume_args(args: argparse.Namespace, payload: dict | None) -> None:
    if payload is None:
        return
    if args.timesteps <= payload["step"]:
        raise ValueError(
            f"--timesteps must exceed checkpoint step {payload['step']:,}; "
            "it is the total target, not additional steps"
        )
    config = payload["config"]
    for name in (
        "gru", "frameskip", "trajectory_length", "policy_hidden", "critic_hidden",
        "score_hidden", "diffusion_steps", "score_timesteps",
    ):
        if getattr(args, name) != config.get(name):
            raise ValueError(
                f"--{name.replace('_', '-')} must match the checkpoint "
                f"({config.get(name)}) when resuming"
            )


def restore_smp_checkpoint(
    payload: dict, args: argparse.Namespace,
    modules: dict[str, nn.Module],
    optimizers: dict[str, th.optim.Optimizer],
    reward: SMPReward, agent_update: AgentScoreUpdate,
) -> Clock:
    for name, module in modules.items():
        module.load_state_dict(payload[name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload[f"{name}_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.score_lr if name == "agent_score" else args.ppo_lr
    reward.agent_ready = bool(payload["agent_ready"])
    agent_update.rollouts = int(payload.get("agent_rollouts", 0))
    if "torch_rng_state" in payload:
        th.set_rng_state(payload["torch_rng_state"].cpu())
    if "cuda_rng_state" in payload:
        th.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if "clock" in payload:
        return Clock(**payload["clock"])
    vector_steps = payload["step"] // (args.n_sim * N_CARS)
    return Clock(
        vector_steps=vector_steps, env_steps=payload["step"],
        learner_updates=math.ceil(vector_steps / args.rollout),
    )


def parse_args() -> tuple[argparse.Namespace, dict | None]:
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--resume-checkpoint", type=Path)
    preliminary, _ = preliminary_parser.parse_known_args()
    resume = (
        load_resume_checkpoint(preliminary.resume_checkpoint)
        if preliminary.resume_checkpoint is not None else None
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--prior-checkpoint", type=Path,
                        help="reuse a frozen expert score prior")
    parser.add_argument("--resume-prior", type=Path,
                        help="continue an interrupted expert-prior pretraining run")
    parser.add_argument("--prior-only", action="store_true",
                        help="pretrain and save an expert score prior without starting CARL")
    parser.add_argument("--prior-output", type=Path)
    parser.add_argument("--replay-dir", type=Path,
                        help="1v1 replay directory (required for prior pretraining or replay resets)")
    parser.add_argument("--n-sim", type=int, default=16_384)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--trajectory-length", type=int, default=8)
    parser.add_argument("--score-hidden", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--score-timesteps", nargs="+", type=int, default=[8, 15, 22],
                        metavar="T", help="fixed ensemble of diffusion noise levels")
    parser.add_argument("--score-lr", type=float, default=3e-4)
    parser.add_argument("--prior-updates", type=int, default=20_000)
    parser.add_argument("--prior-ema", type=float, default=0.995,
                        help="exponential moving average decay for the frozen expert prior")
    parser.add_argument("--prior-batch", type=int, default=1_024)
    parser.add_argument("--prior-microbatch", type=int, default=256)
    parser.add_argument("--prior-eval-interval", type=int, default=1_000)
    parser.add_argument("--prior-save-interval", type=int, default=1_000)
    parser.add_argument("--prior-heldout-size", type=int, default=2_048)
    parser.add_argument("--prior-calibration-size", type=int, default=2_048)
    parser.add_argument("--agent-score-updates", type=int, default=32)
    parser.add_argument("--agent-score-batch", type=int, default=1_024)
    parser.add_argument("--agent-score-microbatch", type=int, default=256)
    parser.add_argument("--agent-update-interval", type=int, default=1)
    parser.add_argument("--history-capacity", type=int, default=65_536)
    parser.add_argument("--history-add-size", type=int, default=8_192)
    parser.add_argument("--smp-scale", type=float, default=1.0)
    parser.add_argument("--smp-weight", type=float, default=1.0)
    parser.add_argument("--contrast-weight", type=float, default=0.25)
    parser.add_argument("--contrast-clip", type=float, default=2.0)
    parser.add_argument("--reward-batch", type=int, default=512)
    parser.add_argument("--goal-reward-weight", type=float, default=1.0)
    parser.add_argument("--aerial-touch-reward-weight", type=float, default=0.5)
    parser.add_argument("--flip-reset-reward-weight", type=float, default=1.0)
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=0.70)
    parser.add_argument("--reset-state-limit", type=int, default=32_768)
    parser.add_argument("--ppo-batch", type=int, default=16_384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--gru", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA")
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000,
                        help="total target environment steps including resumed training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/smp"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            name: Path(value) if name in {
                "replay_dir", "log_dir", "checkpoint_dir", "prior_checkpoint", "prior_output"
            } and value is not None else value
            for name, value in resume["config"].items()
            if name in options and name not in {"resume_checkpoint", "resume_prior"}
        }
        parser.set_defaults(**inherited)
    args = parser.parse_args()
    return args, resume


def validate_args(args: argparse.Namespace, resume: dict | None) -> None:
    if args.prior_only and (resume is not None or args.prior_checkpoint is not None):
        raise ValueError("--prior-only needs fresh replay pretraining, not a checkpoint")
    if args.resume_prior is not None and (resume is not None or args.prior_checkpoint is not None):
        raise ValueError("--resume-prior cannot be combined with a completed checkpoint")
    if args.replay_dir is None:
        if resume is None and args.prior_checkpoint is None:
            raise ValueError("--replay-dir is required to pretrain the expert prior")
        if args.replay_reset_fraction:
            raise ValueError("--replay-dir is required for nonzero replay reset fraction")
    else:
        if not args.replay_dir.is_dir():
            raise FileNotFoundError(args.replay_dir)
        if not next(args.replay_dir.glob("*.npy"), None):
            for name in (f"pro_1v1_fs{args.frameskip}", "pro_1v1_fs4"):
                candidate = args.replay_dir / name
                if candidate.is_dir():
                    args.replay_dir = candidate
                    print(f"Using 1v1 replays from {candidate}")
                    break
        found_1v1 = False
        for path in args.replay_dir.glob("*.npy"):
            source = np.load(path, mmap_mode="r")
            if source.ndim == 2 and source.shape[1] == 161:
                found_1v1 = True
                break
        if not found_1v1:
            raise FileNotFoundError(f"no 1v1 replay files in {args.replay_dir}")

    for name in (
        "n_sim", "frameskip", "max_ticks", "rollout", "trajectory_length",
        "score_hidden", "diffusion_steps", "prior_updates", "prior_batch",
        "prior_microbatch", "prior_eval_interval", "prior_heldout_size",
        "prior_save_interval",
        "prior_calibration_size", "agent_score_updates", "agent_score_batch",
        "agent_score_microbatch", "agent_update_interval", "history_capacity",
        "history_add_size", "reward_batch", "reset_state_limit", "ppo_batch",
        "ppo_epochs", "sequence_length", "policy_hidden", "critic_hidden",
        "timesteps", "checkpoint_interval", "checkpoint_keep",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.trajectory_length < 2 or args.score_hidden < 4 or args.score_hidden % 4:
        raise ValueError("SMP needs --trajectory-length >= 2 and --score-hidden divisible by 4")
    if args.diffusion_steps < 4 or not args.score_timesteps or len(set(args.score_timesteps)) != len(args.score_timesteps) or (
        min(args.score_timesteps) < 0 or max(args.score_timesteps) >= args.diffusion_steps
    ):
        raise ValueError("--score-timesteps must be unique levels inside --diffusion-steps")
    if not math.isfinite(args.prior_ema) or not 0 <= args.prior_ema < 1:
        raise ValueError("--prior-ema must be in [0, 1)")
    if args.history_add_size > args.history_capacity:
        raise ValueError("--history-add-size must not exceed --history-capacity")
    if args.expert_frame_limit is not None and args.expert_frame_limit < args.trajectory_length:
        raise ValueError("--expert-frame-limit must fit an expert scene window")
    if args.rollout < args.trajectory_length - 1:
        raise ValueError("--rollout must fit a scene window")
    if args.ppo_batch > args.rollout * args.n_sim * N_CARS:
        raise ValueError("--ppo-batch must fit a rollout")
    if args.gru and (
        args.rollout % args.sequence_length or args.ppo_batch % args.sequence_length
    ):
        raise ValueError("GRU rollout and PPO batch must be divisible by sequence length")
    for name in (
        "no_touch_timeout", "score_lr", "ppo_lr", "max_grad_norm", "ppo_clip",
        "gamma", "smp_scale", "contrast_clip",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.gamma > 1:
        raise ValueError("--gamma must not exceed one")
    for name in (
        "smp_weight", "contrast_weight", "goal_reward_weight",
        "aerial_touch_reward_weight", "flip_reset_reward_weight", "entropy",
        "value_clip", "value_coef",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not 0 <= args.lambda_ <= 1 or not 0 <= args.replay_reset_fraction <= 1:
        raise ValueError("--lambda and --replay-reset-fraction must be in [0, 1]")
    if args.entropy_end is not None and (
        not math.isfinite(args.entropy_end) or args.entropy_end < 0
    ):
        raise ValueError("--entropy-end must be finite and nonnegative")


def build_runner(env, policy, critic, buffer, args, gameplay: GameplayDiagnostics) -> Runner:
    captures = [LogProbCapture()]
    if args.gru:
        captures.extend((RecurrentStateCapture(), RecurrentCriticCapture(critic)))
    else:
        captures.append(CriticCapture(critic))
    captures.extend((AdvancedTouchCapture(gameplay), DIFOSceneWindowCapture(args.trajectory_length)))
    return Runner(env, policy, buffer, captures=captures)


def build_learner(
    args: argparse.Namespace, policy, critic,
    expert: SceneScoreModel, agent: SceneScoreModel, normalizers: th.Tensor,
    policy_optimizer: th.optim.Optimizer,
    critic_optimizer: th.optim.Optimizer,
    agent_optimizer: th.optim.Optimizer,
) -> tuple[Algorithm, PPOLoss, SMPReward, AgentScoreUpdate]:
    reward = SMPReward(
        expert, agent, args.score_timesteps, normalizers,
        args.trajectory_length, batch_size=args.reward_batch,
        prior_scale=args.smp_scale, prior_weight=args.smp_weight,
        contrast_weight=args.contrast_weight, contrast_clip=args.contrast_clip,
        goal_reward_weight=args.goal_reward_weight,
        aerial_touch_reward_weight=args.aerial_touch_reward_weight,
        flip_reset_reward_weight=args.flip_reset_reward_weight,
    )
    ppo_loss = PPOLoss(policy, critic, PPOConfig(
        clip=args.ppo_clip, value_clip=args.value_clip, value_coef=args.value_coef,
        entropy_coef=args.entropy, normalize_advantage=True,
    ))
    ppo_update = Update(
        transforms=(
            reward,
            GAE(gamma=args.gamma, lambda_=args.lambda_, reward_field="training_reward"),
            SelectPPOFields(recurrent=args.gru),
        ),
        sampler=build_ppo_sampler(args), loss=ppo_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )
    reservoir = MixtureWindowBuffer(args.history_capacity, args.trajectory_length, args.seed)
    agent_update = AgentScoreUpdate(
        agent, agent_optimizer, reservoir, reward,
        updates=args.agent_score_updates, batch_size=args.agent_score_batch,
        microbatch=args.agent_score_microbatch, add_size=args.history_add_size,
        interval=args.agent_update_interval, max_grad_norm=args.max_grad_norm,
    )
    return Algorithm(ppo_update, agent_update), ppo_loss, reward, agent_update


def main() -> None:
    args, resume = parse_args()
    validate_resume_args(args, resume)
    validate_args(args, resume)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_id = datetime.now().strftime("smp-%Y%m%d-%H%M%S-%f")
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    if resume is not None:
        expert = SceneScoreModel(
            args.trajectory_length, args.score_hidden, args.diffusion_steps
        ).to(device)
        normalizers = resume["normalizers"].to(device)
    elif args.prior_checkpoint is not None:
        saved = load_prior_checkpoint(args.prior_checkpoint, args)
        expert = SceneScoreModel(
            args.trajectory_length, args.score_hidden, args.diffusion_steps
        ).to(device)
        expert.load_state_dict(saved["expert_prior"])
        normalizers = saved["normalizers"].to(device)
        print(f"Reusing frozen SMP prior from {args.prior_checkpoint}")
    else:
        expert, normalizers, _ = pretrain_prior(args, run_id, device)
    expert.eval().requires_grad_(False)
    if args.prior_only:
        return

    env = build_env(args)
    try:
        expert = expert.to(env.device)
        normalizers = normalizers.to(env.device)
        gameplay = env.register_reward(GameplayDiagnostics(
            env.n_sim, env.device,
            math.ceil(args.no_touch_timeout * 120.0 / args.frameskip),
        ))
        policy = build_policy(env, args)
        critic = build_critic(env, args)
        agent = SceneScoreModel(
            args.trajectory_length, args.score_hidden, args.diffusion_steps
        ).to(env.device)
        if args.replay_reset_fraction:
            resets = load_demonstration_reset_dataset(
                args.replay_dir, env.device, args.frameskip,
                limit=args.reset_state_limit, seed=args.seed,
            )
            env.reset_state_provider = DatasetResetSampler(
                resets, probability=args.replay_reset_fraction, seed=args.seed
            )

        modules = {
            "policy": policy, "critic": critic,
            "expert_prior": expert, "agent_score": agent,
        }
        optimizers = {
            "policy": th.optim.Adam(policy.parameters(), lr=args.ppo_lr),
            "critic": th.optim.Adam(critic.parameters(), lr=args.ppo_lr),
            "agent_score": th.optim.Adam(agent.parameters(), lr=args.score_lr),
        }
        buffer = RolloutBuffer(args.rollout, env.n_envs, env.device, copy_on_finish=False)
        runner = build_runner(env, policy, critic, buffer, args, gameplay)
        learner, ppo_loss, reward, agent_update = build_learner(
            args, policy, critic, expert, agent, normalizers,
            optimizers["policy"], optimizers["critic"], optimizers["agent_score"],
        )
        restored_clock = None
        if resume is not None:
            restored_clock = restore_smp_checkpoint(
                resume, args, modules, optimizers, reward, agent_update
            )
        value_scheduler = build_entropy_scheduler(args, ppo_loss)
        logger = Logger(args.log_dir / run_id)
        try:
            for section, key, label, fmt in (
                ("SMP", "prior", "SMP prior", ".4f"),
                ("SMP", "cost", "SMILING cost", ".4f"),
                ("SMP", "contrast_bonus", "SMILING bonus", ".4f"),
                ("SMP", "valid_fraction", "SMP valid", ".3f"),
                ("AgentScore", "loss", "agent score MSE", ".4f"),
                ("AgentScore", "updates", "agent score updates", ".0f"),
                ("AgentScore", "windows", "agent history", ".0f"),
                ("PPO", "policy_loss", "policy loss", ".4f"),
                ("PPO", "critic_loss", "critic loss", ".4f"),
                ("PPO", "entropy", "entropy", ".3f"),
                ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
                ("Gameplay", "aerial_touches_per_1000_steps", "aerial/1k", ".3f"),
                ("Gameplay", "flip_resets_per_1000_steps", "flip resets/1k", ".3f"),
                ("Gameplay", "goals_for_per_1000_steps", "goals for/1k", ".3f"),
                ("Gameplay", "goals_against_per_1000_steps", "goals against/1k", ".3f"),
                ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
            ):
                logger.register_progress_metric(section, key, label, fmt)
            if value_scheduler is not None:
                logger.register_progress_metric("Schedule", "entropy_coef", "entropy coef", ".4f")
            checkpoints = SMPCheckpoints(
                args.checkpoint_dir / run_id,
                args.checkpoint_interval, args.checkpoint_keep,
                modules, optimizers, reward, agent_update, buffer, args,
            )

            def update_callback(trainer: Trainer) -> None:
                trainer.logger.update({"SMP": reward.last_metrics}, step=trainer.clock.env_steps)
                metrics = gameplay.diagnostic_metrics()
                if metrics:
                    trainer.logger.update(metrics, step=trainer.clock.env_steps)

            trainer = Trainer(
                runner, buffer, learner, OnPolicySchedule(),
                logger=logger, checkpoint=checkpoints,
                value_scheduler=value_scheduler, update_callback=update_callback,
            )
            if restored_clock is not None:
                trainer.clock = restored_clock
                print(
                    f"Resuming {args.resume_checkpoint} at {restored_clock.env_steps:,} "
                    f"steps in new run {run_id}"
                )
            checkpoints.clock = trainer.clock
            checkpoints.save(trainer.clock.env_steps, force=True)
            trainer.run(args.timesteps)
            checkpoints.save(trainer.clock.env_steps, force=True)
        finally:
            logger.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
