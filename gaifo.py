import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import CaptureContext, CriticCapture, LogProbCapture, Runner
from jarl.collect.capture import CaptureBase
from jarl.data.batch import TensorBatch
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    LossOutput,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE, PrepareContext


SCENE_SIZE = 51
GAIFO_ARCHITECTURE = "scene-marl-gaifo-1v1-v1"
BALL_SIZE = 9
CAR_SIZE = 21
N_CARS = 2
BLUE_START = 9
ORANGE_START = 30
CAR_BOOL_START = 16
CAR_BOOL_END = 21


def noise_mask(device: str | th.device = "cpu") -> th.Tensor:
    """Boolean mask that is True for continuous scene features and False for car booleans."""
    mask = th.ones(SCENE_SIZE, dtype=th.bool, device=device)
    mask[BLUE_START + CAR_BOOL_START : BLUE_START + CAR_BOOL_END] = False
    mask[ORANGE_START + CAR_BOOL_START : ORANGE_START + CAR_BOOL_END] = False
    return mask


def add_scene_noise(windows: th.Tensor, std: float) -> th.Tensor:
    """Apply symmetric Gaussian noise only to continuous scene features."""
    if not math.isfinite(std) or std < 0.0:
        raise ValueError("scene noise standard deviation must be finite and nonnegative")
    if windows.shape[-1] != SCENE_SIZE:
        raise ValueError(f"scene windows must end in {SCENE_SIZE} features")
    if std <= 0.0:
        return windows
    mask = noise_mask(windows.device).view(*((1,) * (windows.ndim - 1)), SCENE_SIZE)
    noise = th.randn_like(windows) * std * mask
    return windows + noise


def extract_scene_observations(
    observation: th.Tensor,
    n_cars: int = N_CARS,
) -> th.Tensor:
    """Take the first actor per simulation and keep only the physical scene prefix."""
    n_envs = observation.shape[-2]
    if n_envs % n_cars:
        raise ValueError("actor count must be divisible by cars per simulation")
    if observation.shape[-1] < SCENE_SIZE:
        raise ValueError(f"actor observations require at least {SCENE_SIZE} features")
    return observation[..., ::n_cars, :SCENE_SIZE].contiguous()


def build_scene_windows(
    observation: th.Tensor,
    next_obs: th.Tensor,
    done: th.Tensor,
    trajectory_length: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Build [time, simulation, trajectory_length, SCENE_SIZE] scene windows.

    A window is valid once ``trajectory_length - 1`` same-episode transitions have
    been observed and it does not cross a terminated/truncated boundary. The
    window always ends with ``next_obs`` for the scored transition.
    """
    if trajectory_length < 2:
        raise ValueError("trajectory length must be at least 2")
    if observation.shape != next_obs.shape:
        raise ValueError("observation and next observation shapes must match")
    T, n_envs = observation.shape[:2]
    if done.shape != (T, n_envs):
        raise ValueError("done mask must match rollout time and actors")
    if n_envs % N_CARS:
        raise ValueError("1v1 rollouts require two actors per simulation")
    n_sim = n_envs // N_CARS

    obs_scene = extract_scene_observations(observation)
    next_scene = extract_scene_observations(next_obs)

    per_sim_done = (done[:, ::N_CARS] | done[:, 1::N_CARS]).to(th.bool)
    prefix = th.cat(
        [
            th.zeros(1, n_sim, dtype=th.long, device=per_sim_done.device),
            per_sim_done.long().cumsum(dim=0),
        ],
        dim=0,
    )

    valid = th.zeros(T, n_sim, dtype=th.bool, device=observation.device)
    windows = th.zeros(
        T, n_sim, trajectory_length, SCENE_SIZE, dtype=observation.dtype, device=observation.device
    )

    for t in range(trajectory_length - 2, T):
        start = t - trajectory_length + 2
        valid[t] = prefix[t] == prefix[start]
        windows[t] = th.cat((
            obs_scene[start:t + 1].transpose(0, 1),
            next_scene[t, :, None],
        ), dim=1)

    return windows, valid


class SceneWindowCapture(CaptureBase):
    """Capture scene trajectories continuously across rollout buffer boundaries."""

    def __init__(self, trajectory_length: int) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        self.trajectory_length = trajectory_length
        self.n_envs = 0
        self.history: th.Tensor | None = None
        self.history_length: th.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        if batch_size % N_CARS:
            raise ValueError("1v1 collection requires two actors per simulation")
        self.n_envs = batch_size
        self.history = None
        self.history_length = None

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        observation = context.observation
        next_obs = th.as_tensor(
            context.env_step.next_obs,
            dtype=observation.dtype,
            device=observation.device,
        )
        if len(observation) != self.n_envs:
            raise ValueError("scene capture batch changed after reset")

        current_scene = extract_scene_observations(observation)
        next_scene = extract_scene_observations(next_obs)
        n_sim = len(current_scene)
        history_size = self.trajectory_length - 1
        if self.history is None:
            self.history = th.zeros(
                n_sim,
                history_size,
                SCENE_SIZE,
                dtype=observation.dtype,
                device=observation.device,
            )
            self.history_length = th.zeros(
                n_sim,
                dtype=th.long,
                device=observation.device,
            )
        assert self.history_length is not None

        self.history = th.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = current_scene
        self.history_length += 1
        valid = self.history_length >= history_size
        window = th.cat((self.history, next_scene[:, None]), dim=1)

        done = th.as_tensor(
            context.env_step.done,
            dtype=th.bool,
            device=observation.device,
        )
        simulation_done = done.view(n_sim, N_CARS).any(dim=1)
        self.history[simulation_done] = 0
        self.history_length[simulation_done] = 0

        return {
            "scene_window": window.repeat_interleave(N_CARS, dim=0),
            "scene_window_valid": valid.repeat_interleave(N_CARS),
        }


class ExpertSceneDataset:
    """Expert scene windows extracted from raw 1v1 replay .npy files."""

    def __init__(
        self,
        replay_dir: Path,
        trajectory_length: int,
        limit: int | None = None,
        seed: int = 0,
        frame_skip: int | None = None,
    ) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        if limit is not None and limit < trajectory_length:
            raise ValueError("expert frame limit must fit one trajectory")
        self.trajectory_length = trajectory_length
        self.limit = limit

        rng = np.random.default_rng(seed)
        paths = sorted(Path(replay_dir).glob("*.npy"))
        groups: dict[tuple[str, ...], list[Path]] = {}
        for path in paths:
            source = np.load(path, mmap_mode="r")
            if source.ndim != 2 or source.shape[1] != 161:
                continue
            key = self._dedup_key(path)
            groups.setdefault(key, []).append(path)

        selected = [sorted(groups[key])[0] for key in sorted(groups)]
        if not selected:
            raise ValueError(f"no 1v1 replay files found in {replay_dir}")

        if limit is not None:
            rng.shuffle(selected)

        frames: list[th.Tensor] = []
        lengths: list[int] = []
        total = 0
        for path in selected:
            if frame_skip is not None:
                metadata_path = path.with_suffix(".unsafe-starts.npz")
                if not metadata_path.is_file():
                    raise ValueError(f"missing frame-skip metadata for {path.name}")
                with np.load(metadata_path) as metadata:
                    stored_frame_skip = int(metadata.get("frame_skip", -1))
                if stored_frame_skip != frame_skip:
                    raise ValueError(
                        f"expert replay {path.name} uses frame skip "
                        f"{stored_frame_skip}, expected {frame_skip}"
                    )
            source = np.array(
                np.load(path, mmap_mode="r")[:, :SCENE_SIZE], dtype=np.float32, copy=True
            )
            if limit is not None and total + len(source) > limit:
                keep = max(0, limit - total)
                if keep == 0:
                    break
                source = source[:keep]
            frames.append(th.from_numpy(source))
            lengths.append(len(source))
            total += len(source)
            if limit is not None and total >= limit:
                break

        if not frames:
            raise ValueError(f"no expert frames loaded from {replay_dir}")

        self.frames = th.cat(frames)
        self.lengths = lengths
        self.starts = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        self._rng = np.random.default_rng(seed)
        self.total_windows = sum(
            max(0, length - trajectory_length + 1) for length in lengths
        )
        if self.total_windows <= 0:
            raise ValueError(
                f"expert files are too short to build windows of length {trajectory_length}"
            )

    @staticmethod
    def _dedup_key(path: Path) -> tuple[str, ...]:
        """POV copies share {period}-{replay}; keep one canonical file."""
        parts = path.stem.split("-", 2)
        if len(parts) >= 3:
            return (parts[1], parts[2])
        return (path.stem,)

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` scene windows without crossing file boundaries."""
        if n < 1:
            raise ValueError("sample count must be positive")
        if self.total_windows <= 0:
            raise RuntimeError("no expert windows available")

        windows_per_file = np.array(
            [max(0, length - self.trajectory_length + 1) for length in self.lengths],
            dtype=np.float64,
        )
        probabilities = windows_per_file / windows_per_file.sum()
        file_indices = self._rng.choice(len(self.lengths), size=n, p=probabilities)
        starts = [
            self._rng.integers(0, int(windows_per_file[i])) for i in file_indices
        ]

        batch = th.empty(
            n, self.trajectory_length, SCENE_SIZE, dtype=self.frames.dtype
        )
        for j, (file_index, start) in enumerate(zip(file_indices, starts)):
            offset = int(self.starts[file_index]) + int(start)
            batch[j] = self.frames[offset : offset + self.trajectory_length]

        return batch.to(device)


class SceneDiscriminator(nn.Module):
    """Structured scene-level discriminator for short 1v1 trajectory windows."""

    def __init__(
        self,
        frame_embedding: int,
        temporal_hidden: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            frame_embedding * 3, temporal_hidden, batch_first=True
        )
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        B, T, _ = windows.shape
        ball = windows[..., :BALL_SIZE]
        cars = windows[..., BALL_SIZE:SCENE_SIZE].view(B, T, N_CARS, CAR_SIZE)
        blue, orange = cars[:, :, 0], cars[:, :, 1]

        sign = th.tensor([1.0, -1.0], device=windows.device, dtype=windows.dtype)
        blue_in = th.cat([blue, sign[0].expand_as(blue[..., :1])], dim=-1)
        orange_in = th.cat([orange, sign[1].expand_as(orange[..., :1])], dim=-1)

        ball_emb = self.ball_encoder(ball)
        blue_emb = self.car_encoder(blue_in)
        orange_emb = self.car_encoder(orange_in)

        frame_embedding = th.cat([ball_emb, blue_emb, orange_emb], dim=-1)
        gru_out, _ = self.gru(frame_embedding)
        return self.head(gru_out[:, -1]).squeeze(-1)


class SceneDiscriminatorLoss:
    """BCE-with-logits loss for generated-vs-expert scene windows."""

    def __init__(self, discriminator: SceneDiscriminator) -> None:
        self.discriminator = discriminator

    def after_update(self) -> None:
        return

    def __call__(self, batch: TensorBatch) -> LossOutput:
        logit = self.discriminator(batch["window"])
        target = batch["is_agent"]
        loss = F.binary_cross_entropy_with_logits(logit, target)

        agent = target.bool()
        with th.no_grad():
            agent_score = th.sigmoid(logit[agent]).mean()
            expert_score = th.sigmoid(logit[~agent]).mean()
            agent_accuracy = (logit[agent] > 0.0).float().mean()
            expert_accuracy = (logit[~agent] <= 0.0).float().mean()

        return LossOutput(
            loss,
            {
                "loss": loss,
                "agent_score": agent_score,
                "expert_score": expert_score,
                "agent_accuracy": agent_accuracy,
                "expert_accuracy": expert_accuracy,
            },
        )


class SceneGAIFOMinibatches:
    """Minibatch sampler for generated vs expert scene windows."""

    def __init__(
        self,
        expert: ExpertSceneDataset,
        batch_size: int,
        epochs: int,
        noise_std: float,
    ) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch size and epochs must be positive")
        self.expert = expert
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_std = noise_std
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def __call__(self, batch: TensorBatch):
        windows = batch["scene_window"][:, ::N_CARS]
        valid = batch["scene_window_valid"][:, ::N_CARS].bool()
        if windows.shape[-2:] != (self.expert.trajectory_length, SCENE_SIZE):
            raise ValueError("captured scene windows do not match expert trajectories")
        if not valid.any():
            raise RuntimeError("no valid generated scene windows in rollout")

        generated = windows[valid]

        for _ in range(self.epochs):
            indices = th.randperm(len(generated), device=generated.device)
            for start in range(0, len(generated), self.batch_size):
                selected = indices[start : start + self.batch_size]
                sample_count = len(selected)

                agent_windows = add_scene_noise(generated[selected], self.noise_std)
                expert_windows = add_scene_noise(
                    self.expert.sample(sample_count, agent_windows.device),
                    self.noise_std,
                )
                is_agent = th.cat(
                    [
                        th.ones(sample_count, device=agent_windows.device),
                        th.zeros(sample_count, device=agent_windows.device),
                    ]
                )

                yield TensorBatch(
                    {
                        "window": th.cat([agent_windows, expert_windows]),
                        "is_agent": is_agent,
                    }
                )

            if self._epoch_callback is not None:
                self._epoch_callback()


class SceneDiscriminatorReward:
    """Turn discriminator logits into a global scene imitation reward."""

    def __init__(
        self,
        discriminator: SceneDiscriminator,
        noise_std: float,
        trajectory_length: int,
        output_field: str = "imitation_reward",
    ) -> None:
        self.discriminator = discriminator
        self.noise_std = noise_std
        self.trajectory_length = trajectory_length
        self.output_field = output_field

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        windows = batch["scene_window"][:, ::N_CARS]
        valid = batch["scene_window_valid"][:, ::N_CARS].bool()
        if windows.shape[-2:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("captured scene windows have the wrong shape")

        scores = th.zeros_like(valid, dtype=batch["observation"].dtype)
        if valid.any():
            noisy = add_scene_noise(windows[valid], self.noise_std)
            logits = self.discriminator(noisy)
            scores[valid] = F.softplus(-logits)

        # Full team spirit: broadcast the same scene score to every actor.
        reward = scores.repeat_interleave(N_CARS, dim=1)
        valid_actors = valid.repeat_interleave(N_CARS, dim=1)
        result = batch.with_fields(**{self.output_field: reward})
        if "learner_mask" in result:
            return result.replace_fields(
                learner_mask=result["learner_mask"].bool() & valid_actors
            )
        return result.with_fields(learner_mask=valid_actors)


class GAIFOCheckpoints:
    """Periodic checkpointing for policy, critic, discriminator and optimizers."""

    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        discriminator: nn.Module,
        policy_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        discriminator_optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
    ) -> None:
        self.directory = Path(directory)
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.buffer = buffer
        self.args = args
        self.step = 0
        self.next_step = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("gaifo_*.pt.tmp"):
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
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "config": {
                "architecture": GAIFO_ARCHITECTURE,
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
            },
        }
        path = self.directory / f"gaifo_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)

        paths = sorted(self.directory.glob("gaifo_*.pt"))
        for old in paths[:-self.keep]:
            old.unlink()

        self.next_step = step + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAIfO imitation learning for 1v1 Rocket League via CARL and JARL."
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--trajectory-length", type=int, default=8)
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--discriminator-noise", type=float, default=0.01)
    parser.add_argument("--discriminator-batch", type=int, default=64)
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument("--discriminator-hidden", type=int, default=128)
    parser.add_argument("--frame-embedding", type=int, default=128)
    parser.add_argument("--temporal-hidden", type=int, default=128)
    parser.add_argument("--ppo-batch", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA"
    )
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gaifo"))
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)

    found_1v1 = False
    for path in args.replay_dir.glob("*.npy"):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            found_1v1 = True
            break
    if not found_1v1:
        raise FileNotFoundError(f"no 1v1 replay files in {args.replay_dir}")

    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "trajectory_length",
        "discriminator_batch",
        "discriminator_epochs",
        "discriminator_hidden",
        "frame_embedding",
        "temporal_hidden",
        "ppo_batch",
        "ppo_epochs",
        "policy_hidden",
        "critic_hidden",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if args.no_touch_timeout is not None and (
        not math.isfinite(args.no_touch_timeout) or args.no_touch_timeout <= 0.0
    ):
        raise ValueError("--no-touch-timeout must be positive or omitted")
    if args.expert_frame_limit is not None and args.expert_frame_limit < 1:
        raise ValueError("--expert-frame-limit must be positive")
    if (
        args.expert_frame_limit is not None
        and args.expert_frame_limit < args.trajectory_length
    ):
        raise ValueError("--expert-frame-limit must fit one trajectory")
    if not math.isfinite(args.discriminator_noise) or args.discriminator_noise < 0.0:
        raise ValueError("--discriminator-noise must be non-negative")
    if (
        not math.isfinite(args.ppo_lr)
        or not math.isfinite(args.discriminator_lr)
        or args.ppo_lr <= 0.0
        or args.discriminator_lr <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not math.isfinite(args.lambda_) or not 0.0 <= args.lambda_ <= 1.0:
        raise ValueError("--lambda must be in [0, 1]")
    if not math.isfinite(args.entropy) or args.entropy < 0.0:
        raise ValueError("--entropy must be non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if not math.isfinite(args.ppo_clip) or args.ppo_clip <= 0.0:
        raise ValueError("--ppo-clip must be positive")
    if not math.isfinite(args.value_clip) or args.value_clip < 0.0:
        raise ValueError("--value-clip must be non-negative")
    if not math.isfinite(args.value_coef) or args.value_coef < 0.0:
        raise ValueError("--value-coef must be non-negative")

    if args.rollout < args.trajectory_length - 1:
        raise ValueError("--rollout must be at least --trajectory-length - 1")

    generated_windows = max(
        0, args.rollout - (args.trajectory_length - 2)
    ) * args.n_sim
    if generated_windows < args.discriminator_batch:
        raise ValueError(
            f"rollout is too short to produce a discriminator batch: "
            f"{generated_windows} generated windows, batch size {args.discriminator_batch}"
        )

    n_envs = args.n_sim * 2
    if args.ppo_batch > args.rollout * n_envs:
        raise ValueError("--ppo-batch must fit the rollout size")


def build_env(args: argparse.Namespace) -> CARLTorchVectorEnv:
    return CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        discrete_actions=True,
    )


def build_policy(env, args: argparse.Namespace) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=LinearEncoder(args.policy_hidden, func=nn.ReLU),
        body=MLP(dims=[args.policy_hidden], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, args: argparse.Namespace) -> Critic:
    return Critic(
        foot=LinearEncoder(args.critic_hidden, func=nn.ReLU),
        body=MLP(dims=[args.critic_hidden], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)


def build_discriminator(args: argparse.Namespace) -> SceneDiscriminator:
    return SceneDiscriminator(
        frame_embedding=args.frame_embedding,
        temporal_hidden=args.temporal_hidden,
        hidden_size=args.discriminator_hidden,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    policy = build_policy(env, args)
    critic = build_critic(env, args)
    discriminator = build_discriminator(args).to(env.device)

    expert = ExpertSceneDataset(
        args.replay_dir,
        args.trajectory_length,
        args.expert_frame_limit,
        args.seed,
        frame_skip=args.frameskip,
    )
    if expert.total_windows < 1:
        raise ValueError("expert dataset contains no valid windows")

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
    discriminator_optimizer = th.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr
    )

    buffer = RolloutBuffer(args.rollout, env.n_envs, env.device)
    runner = Runner(
        env,
        policy,
        buffer,
        captures=(
            LogProbCapture(),
            CriticCapture(critic),
            SceneWindowCapture(args.trajectory_length),
        ),
    )

    discriminator_update = Update(
        transforms=(),
        sampler=SceneGAIFOMinibatches(
            expert,
            args.discriminator_batch,
            args.discriminator_epochs,
            args.discriminator_noise,
        ),
        loss=SceneDiscriminatorLoss(discriminator),
        optimizer_step=OptimizerStep(discriminator, discriminator_optimizer),
        section="Discriminator",
    )

    ppo_update = Update(
        transforms=(
            SceneDiscriminatorReward(
                discriminator,
                args.discriminator_noise,
                args.trajectory_length,
            ),
            GAE(
                gamma=args.gamma,
                lambda_=args.lambda_,
                reward_field="imitation_reward",
            ),
        ),
        sampler=RolloutMinibatches(args.ppo_batch, args.ppo_epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=args.ppo_clip,
                value_clip=args.value_clip,
                value_coef=args.value_coef,
                entropy_coef=args.entropy,
                normalize_advantage=True,
            ),
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )

    learner = Algorithm(discriminator_update, ppo_update)

    run_id = datetime.now().strftime("gaifo-%Y%m%d-%H%M%S-%f")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, fmt in (
        ("Discriminator", "loss", "discriminator loss", ".4f"),
        ("Discriminator", "agent_score", "agent score", ".3f"),
        ("Discriminator", "expert_score", "expert score", ".3f"),
        ("Discriminator", "agent_accuracy", "agent accuracy", ".3f"),
        ("Discriminator", "expert_accuracy", "expert accuracy", ".3f"),
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "entropy", "entropy", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, fmt)

    checkpoints = GAIFOCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        discriminator,
        policy_optimizer,
        critic_optimizer,
        discriminator_optimizer,
        buffer,
        args,
    )

    trainer = Trainer(
        runner,
        buffer,
        learner,
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
    )

    try:
        checkpoints.save(0, force=True)
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
