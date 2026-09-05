import argparse
import hashlib
import math

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from gymnasium.vector.utils import batch_space
from torch.optim import Adam
from torch.distributions import Normal

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import (
    CriticCapture,
    LogProbCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.learn import Algorithm, OptimizerStep, PPOConfig, PPOLoss, Update
from jarl.log.logger import Logger
from jarl.data import TensorBatch, TensorDataset
from jarl.envs import DatasetResetSampler
from jarl.modules import MLP, orthogonal_init
from jarl.modules.base import CompositeNet
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import DiagonalGaussianPolicy
from jarl.runtime import OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from distill import (
    ActionDecoder,
    ConditionalPrior,
    GOAL_STATE_SIZE,
    factor_actions,
    masked_logits,
)
from physics_utils import forward_up_to_quat
from replay_safety import infer_unsafe_start_mask
from rewards import AnnealedNextoReward, nexto_shaping_scale
from tracker import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    BOOST_MAX,
    CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED,
    POSITION_SCALE,
)


def load_demonstration_reset_dataset(
    replay_dir: Path,
    device,
    frame_skip: int,
    limit: int | None = None,
    seed: int = 0,
) -> TensorDataset:
    random = np.random.default_rng(seed)
    rows = []
    paths = []

    for path in sorted(replay_dir.glob("*.npy")):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            paths.append(path)

    if not paths:
        raise ValueError(f"no 1v1 demonstrations found in {replay_dir}")
    quota = None if limit is None else max(1, math.ceil(limit / len(paths)))

    for path in paths:
        source = np.load(path, mmap_mode="r")

        unsafe_path = path.with_suffix(".unsafe-starts.npz")
        if unsafe_path.is_file():
            with np.load(unsafe_path) as stored:
                unsafe = np.asarray(stored["unsafe"], dtype=bool)
                stored_skip = int(stored.get("frame_skip", frame_skip))
            if stored_skip != frame_skip:
                raise ValueError(
                    f"unsafe-start mask for {path.name} uses frame skip "
                    f"{stored_skip}, expected {frame_skip}"
                )
            if unsafe.shape != (len(source),):
                raise ValueError(f"unsafe-start mask for {path.name} has wrong shape")
        else:
            unsafe = infer_unsafe_start_mask(
                source[:, 3:6] * BALL_MAX_SPEED, frame_skip
            )

        cars = source[:, 9:51].reshape(-1, 2, 21)
        invalid = source[:, -4:].astype(bool).any(axis=-1)
        stable = cars[..., 16].astype(bool).all(axis=-1)
        stable &= ~cars[..., 17:21].astype(bool).any(axis=(-2, -1))
        eligible = np.flatnonzero(~unsafe & ~invalid & stable)
        if len(eligible):
            if quota is not None and len(eligible) > quota:
                eligible = random.choice(eligible, size=quota, replace=False)
            rows.append(np.asarray(source[eligible, :51], dtype=np.float32))

    if not rows:
        raise ValueError(f"no safe grounded 1v1 states found in {replay_dir}")

    states = np.concatenate(rows)
    if limit is not None and len(states) > limit:
        selected = random.choice(len(states), size=limit, replace=False)
        states = states[selected]
    state = th.from_numpy(np.ascontiguousarray(states)).to(device)
    ball = state[:, :9]
    cars = state[:, 9:51].reshape(-1, 2, 21)
    position_scale = th.tensor(POSITION_SCALE, device=device)
    data = TensorBatch({
        "ball_position": ball[:, :3] * position_scale,
        "ball_velocity": ball[:, 3:6] * BALL_MAX_SPEED,
        "ball_angular_velocity": ball[:, 6:9] * BALL_MAX_ANG_SPEED,
        "car_position": cars[..., :3] * position_scale,
        "car_rotation": forward_up_to_quat(cars[..., 9:12], cars[..., 12:15]),
        "car_velocity": cars[..., 3:6] * CAR_MAX_SPEED,
        "car_angular_velocity": cars[..., 6:9] * CAR_MAX_ANG_SPEED,
        "car_demoed": cars[..., 17].bool(),
        "car_boost": cars[..., 15] * BOOST_MAX,
    })
    return TensorDataset(data)


class FrozenPulseController(nn.Module):
    def __init__(
        self,
        prior: ConditionalPrior,
        decoder: ActionDecoder,
        action_codec,
    ) -> None:
        super().__init__()
        self.prior = prior.eval().requires_grad_(False)
        self.decoder = decoder.eval().requires_grad_(False)
        self.action_codec = action_codec

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        action_codec,
        device,
        frame_skip: int | None = None,
    ) -> "FrozenPulseController":
        payload = th.load(checkpoint, map_location=device, weights_only=True)
        config = payload["config"]
        if frame_skip is not None and int(config["frameskip"]) != frame_skip:
            raise ValueError(
                "self-play frame skip does not match the distillation artifact"
            )
        prior = ConditionalPrior(
            GOAL_STATE_SIZE,
            int(config["latent_size"]),
            list(config["encoder_hidden"]),
        ).to(device)
        decoder = ActionDecoder(
            GOAL_STATE_SIZE,
            int(config["latent_size"]),
            list(config["decoder_hidden"]),
        ).to(device)
        prior.load_state_dict(payload["prior"])
        decoder.load_state_dict(payload["decoder"])
        return cls(prior, decoder, action_codec)

    @property
    def latent_size(self) -> int:
        return self.prior.mean.out_features

    @th.no_grad()
    def decode(self, observation: th.Tensor, residual: th.Tensor) -> th.Tensor:
        state = observation[..., :GOAL_STATE_SIZE]
        prior_mean, _ = self.prior(state)
        logits = self.decoder(state, prior_mean + residual)
        return factor_actions(masked_logits(logits, state, self.action_codec))


class PulseLatentEnv:
    """Treat a frozen PULSE prior and decoder as the environment dynamics."""

    def __init__(self, env, controller: FrozenPulseController) -> None:
        self.env = env
        self.controller = controller
        self.n_envs = env.n_envs
        self.n_sim = env.n_sim
        self.device = env.device
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        self.single_action_space = gym.spaces.Box(
            -math.inf,
            math.inf,
            (controller.latent_size,),
            dtype="float32",
        )
        self.action_space = batch_space(self.single_action_space, self.n_envs)
        self._observation: th.Tensor | None = None

    def reset(self, **kwargs) -> th.Tensor:
        self._observation = self.env.reset(**kwargs)
        return self._observation

    def step(self, residual: th.Tensor):
        if self._observation is None:
            raise RuntimeError("latent environment must be reset before stepping")
        residual = th.as_tensor(residual, device=self.device)
        expected = (self.n_envs, self.controller.latent_size)
        if residual.shape != expected:
            raise ValueError(
                f"latent action has shape {tuple(residual.shape)}, expected {expected}"
            )
        action = self.controller.decode(self._observation, residual)
        result = self.env.step(action)
        self._observation = result[0]
        return result

    def close(self) -> None:
        self.env.close()


class FixedGaussianPolicy(DiagonalGaussianPolicy):
    def __init__(self, foot: nn.Module, body: nn.Module, head: nn.Module, std: float):
        super().__init__(foot, body, head)
        self.fixed_std = std

    def build(self, env) -> "FixedGaussianPolicy":
        super().build(env)
        with th.no_grad():
            self.log_std.fill_(math.log(self.fixed_std))
        self.log_std.requires_grad_(False)
        return self

    def dist(self, observation: th.Tensor) -> Normal:
        mean = CompositeNet.forward(self, observation)
        return Normal(mean, self.log_std.expand_as(mean).exp())

    def action(self, observation: th.Tensor) -> th.Tensor:
        return CompositeNet.forward(self, observation)


class SelfPlayCheckpoints:
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        controller: FrozenPulseController,
        args: argparse.Namespace,
    ) -> None:
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.optimizer = optimizer
        self.buffer = buffer
        self.controller = controller
        self.args = args
        self.step = 0
        self.next_step = interval
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("self_play_*.pt.tmp"):
            path.unlink()
        self.distill_sha256 = file_sha256(args.distill_checkpoint)
        source = th.load(args.distill_checkpoint, map_location="cpu", weights_only=True)
        artifact = directory / "frozen_pulse.pt"
        temporary = artifact.with_suffix(".pt.tmp")
        th.save({
            "prior": controller.prior.state_dict(),
            "decoder": controller.decoder.state_dict(),
            "config": source["config"],
            "sha256": self.distill_sha256,
        }, temporary)
        temporary.replace(artifact)
        self.pulse_sha256 = file_sha256(artifact)

    def ready(self, step: int) -> bool:
        self.step = step
        return step >= self.next_step and self.buffer.position == 0

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "distill_checkpoint": str(self.args.distill_checkpoint),
            "distill_sha256": self.distill_sha256,
            "pulse_artifact": "frozen_pulse.pt",
            "pulse_sha256": self.pulse_sha256,
            "config": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(self.args).items()
            },
        }
        path = self.directory / f"self_play_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("self_play_*.pt"))
        for old_path in paths[:-self.keep]:
            old_path.unlink()
        self.next_step = step + self.interval


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PULSE latent policy with Rocket League self-play."
    )
    parser.add_argument("--distill-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--exploration-std", type=float, default=0.22)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--current-fraction", type=float, default=0.5)
    parser.add_argument("--snapshot-interval", type=int, default=10_000_000)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--historical-policies", type=int, default=4)
    parser.add_argument("--demonstration-reset-fraction", type=float, default=0.8)
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--nexto-shaping-scale", type=float, default=1.0)
    parser.add_argument("--nexto-anneal-start", type=int, default=250_000_000)
    parser.add_argument("--nexto-anneal-end", type=int, default=1_750_000_000)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/self_play")
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "batch_size",
        "epochs",
        "lr",
        "exploration_std",
        "max_grad_norm",
        "snapshot_interval",
        "snapshot_pool_size",
        "historical_policies",
        "reset_state_limit",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.snapshot_pool_size < 3:
        raise ValueError("--snapshot-pool-size must be at least three")
    if not 0.0 <= args.current_fraction <= 1.0:
        raise ValueError("--current-fraction must be between zero and one")
    if not 0.0 <= args.demonstration_reset_fraction <= 1.0:
        raise ValueError("--demonstration-reset-fraction must be between zero and one")
    if not 0.0 <= args.nexto_shaping_scale <= 1.0:
        raise ValueError("--nexto-shaping-scale must be between zero and one")
    if args.nexto_anneal_start < 0:
        raise ValueError("--nexto-anneal-start cannot be negative")
    if args.nexto_anneal_end <= args.nexto_anneal_start:
        raise ValueError("--nexto-anneal-end must be greater than --nexto-anneal-start")
    if args.historical_policies >= args.snapshot_pool_size:
        raise ValueError("--historical-policies must be smaller than the snapshot pool")
    if not args.distill_checkpoint.is_file():
        raise FileNotFoundError(args.distill_checkpoint)
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)


def build_policy(env, exploration_std: float) -> FixedGaussianPolicy:
    return FixedGaussianPolicy(
        foot=LinearEncoder(2048, func=nn.ReLU),
        body=MLP(dims=[1024, 512], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        std=exploration_std,
    ).build(env).to(env.device)


def build_policy_and_critic(env, exploration_std: float):
    policy = build_policy(env, exploration_std)

    critic = Critic(
        foot=LinearEncoder(2048, func=nn.ReLU),
        body=MLP(dims=[1024, 512], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)
    return policy, critic


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)

    reset_dataset = load_demonstration_reset_dataset(
        args.replay_dir,
        "cuda:0",
        args.frameskip,
        args.reset_state_limit,
        args.seed,
    )
    reset_sampler = DatasetResetSampler(
        reset_dataset,
        probability=args.demonstration_reset_fraction,
        seed=args.seed,
    )
    reward = AnnealedNextoReward(1, 1, args.nexto_shaping_scale)
    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout_seconds,
        normalize=True,
        reset_state_provider=reset_sampler,
        reward_funcs=(reward,),
    )
    controller = FrozenPulseController.load(
        args.distill_checkpoint,
        base_env.action_codec,
        base_env.device,
        frame_skip=args.frameskip,
    )
    env = PulseLatentEnv(base_env, controller)
    policy, critic = build_policy_and_critic(env, args.exploration_std)

    run_id = datetime.now().strftime("self-play-%Y%m%d-%H%M%S-%f")
    pool = SnapshotPool(
        policy,
        max_size=args.snapshot_pool_size,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
        checkpoint_dir=None,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=args.n_sim,
        team_sizes=(1, 1),
        current_fraction=args.current_fraction,
        historical_ids=pool.select_ids(args.historical_policies),
        device=env.device,
        seed=args.seed,
    )
    buffer = RolloutBuffer(
        horizon=args.rollout,
        num_envs=env.n_envs,
        device=env.device,
        copy_on_finish=False,
    )
    runner = SelfPlayRunner(
        env,
        policy,
        buffer,
        opponent_pool=pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=args.historical_policies,
        captures=(LogProbCapture(), CriticCapture(critic)),
    )

    optimizer = Adam((*policy.parameters(), *critic.parameters()), lr=args.lr)
    update = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
        sampler=RolloutMinibatches(args.batch_size, args.epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(clip=0.2, value_clip=0.2, entropy_coef=0.0),
        ),
        optimizer_step=OptimizerStep(
            (policy, critic),
            optimizer,
            max_grad_norm=args.max_grad_norm,
        ),
        section="PPO",
    )
    value_scheduler = ValueScheduler(
        ScheduledValue.attribute(
            "nexto_shaping_scale",
            reward,
            "shaping_scale",
            lambda progress: nexto_shaping_scale(
                round(progress * args.timesteps),
                args.nexto_shaping_scale,
                args.nexto_anneal_start,
                args.nexto_anneal_end,
            ),
        ),
        section="Reward",
    )
    checkpoints = SelfPlayCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        optimizer,
        buffer,
        controller,
        args,
    )
    checkpoints.save(0, force=True)
    logger = Logger(args.log_dir / run_id)
    for section, key, label, format_spec in (
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "approx_kl", "approx KL", ".4f"),
        ("episode", "historical_reward", "historical reward", ".3f"),
        ("Reward", "nexto_shaping_scale", "reward shaping", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, format_spec)
    trainer = Trainer(
        runner,
        buffer,
        Algorithm(update),
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
    )

    try:
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
