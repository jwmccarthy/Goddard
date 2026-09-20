import argparse
import hashlib
import math

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch as th
import torch.nn as nn

from gymnasium.vector.utils import batch_space
from torch.optim import Adam
from torch.distributions import Normal

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import (
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.data.records import Evaluation, PolicyOutput
from jarl.learn import Algorithm, OptimizerStep, PPOConfig, PPOLoss, Update
from jarl.log.logger import Logger
from jarl.envs import DatasetResetSampler
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import DiagonalGaussianPolicy
from jarl.runtime import OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler
from jarl.sample import RolloutMinibatches
from jarl.store.rollout import RolloutBuffer
from jarl.transform import GAE

from distill import (
    ACTION_FORMAT,
    ActionDecoder,
    ConditionalPrior,
    GOAL_STATE_SIZE,
    factor_actions,
    masked_logits,
)
from replay_resets import load_demonstration_reset_dataset
from rewards import AnnealedNextoReward, nexto_shaping_scale


def primitive_discount(frameskip: int, half_life_seconds: float) -> float:
    """Per-step discount given the physics frame skip and a half-life."""
    if frameskip <= 0 or not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("frameskip and half_life_seconds must be positive")
    ticks_per_second = 120.0
    return math.exp(-math.log(2.0) * frameskip / (ticks_per_second * half_life_seconds))


class FrozenPulseController(nn.Module):
    """Frozen PULSE prior and decoder used as the self-play action layer."""

    def __init__(
        self,
        prior: ConditionalPrior,
        decoder: ActionDecoder,
        action_codec,
        bf16: bool = False,
    ) -> None:
        super().__init__()
        self.prior = prior.eval().requires_grad_(False)
        self.decoder = decoder.eval().requires_grad_(False)
        self.action_codec = action_codec
        self.bf16 = bf16

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        action_codec,
        device,
        frame_skip: int | None = None,
        bf16: bool = False,
    ) -> "FrozenPulseController":
        payload = th.load(checkpoint, map_location=device, weights_only=True)
        config = payload["config"]
        if config.get("action_format") != ACTION_FORMAT:
            raise RuntimeError("distillation checkpoint uses an incompatible action format")
        if frame_skip is not None and int(config["frameskip"]) != frame_skip:
            raise ValueError(
                "self-play frame skip does not match the distillation artifact"
            )
        control_state_size = int(config.get("control_state_size", GOAL_STATE_SIZE))
        prior = ConditionalPrior(
            control_state_size,
            int(config["latent_size"]),
            list(config["encoder_hidden"]),
        ).to(device)
        decoder = ActionDecoder(
            control_state_size,
            int(config["latent_size"]),
            list(config["decoder_hidden"]),
        ).to(device)
        prior.load_state_dict(payload["prior"])
        decoder.load_state_dict(payload["decoder"])
        return cls(prior, decoder, action_codec, bf16)

    @property
    def latent_size(self) -> int:
        return self.prior.latent_dim

    @th.no_grad()
    def prior_mean(self, observation: th.Tensor) -> th.Tensor:
        state = observation[..., : self.prior.state_dim]
        with th.autocast(
            device_type=state.device.type,
            dtype=th.bfloat16,
            enabled=self.bf16 and state.device.type == "cuda",
        ):
            mean, _ = self.prior(state)
        return mean

    @th.no_grad()
    def select_latent(
        self,
        observation: th.Tensor,
        residual: th.Tensor,
    ) -> th.Tensor:
        return self.prior_mean(observation) + residual

    @th.no_grad()
    def decode(self, observation: th.Tensor, residual: th.Tensor) -> th.Tensor:
        state = observation[..., : self.prior.state_dim]
        with th.autocast(
            device_type=state.device.type,
            dtype=th.bfloat16,
            enabled=self.bf16 and state.device.type == "cuda",
        ):
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


class TrainableGaussianPolicy(DiagonalGaussianPolicy):
    def __init__(self, foot: nn.Module, body: nn.Module, head: nn.Module, std: float):
        super().__init__(foot, body, head)
        self.fixed_std = std

    def build(self, env) -> "TrainableGaussianPolicy":
        super().build(env)
        with th.no_grad():
            self.log_std.fill_(math.log(self.fixed_std))
        return self

    def _distribution(self, features: th.Tensor) -> Normal:
        mean = self.head(features)
        return Normal(mean, self.log_std.expand_as(mean).exp())

    def body_features(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        reset: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        features = self.foot(observation)
        if hasattr(self.body, "initial_state"):
            if (
                state is not None
                and state.dtype != features.dtype
                and th.is_autocast_enabled()
            ):
                state = state.to(features.dtype)
            return self.body(features, state, reset)
        if state is not None or reset is not None:
            raise ValueError("stateless policy body does not accept state")
        return self.body(features), None

    def dist(self, observation: th.Tensor) -> Normal:
        features, _ = self.body_features(observation)
        return self._distribution(features)

    def action(self, observation: th.Tensor) -> th.Tensor:
        return self.dist(observation).mean

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        features, next_state = self.body_features(observation, state)
        distribution = self._distribution(features)
        action = distribution.mean if deterministic else distribution.sample()
        return PolicyOutput(
            action=action,
            next_state=next_state,
            log_prob=None if deterministic else self._logprob(distribution, action),
        )

    def evaluate_actions(
        self,
        observation: th.Tensor,
        action: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> Evaluation:
        features, _ = self.body_features(observation, state, reset)
        distribution = self._distribution(features)
        return Evaluation(
            log_prob=self._logprob(distribution, action),
            entropy=self._entropy(distribution),
        )


class CriticValueCapture(CaptureBase):
    def __init__(self, critic: Critic) -> None:
        self.critic = critic

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        next_observation = th.as_tensor(
            context.env_step.next_obs,
            device=context.observation.device,
        )
        return {
            "baseline_value": self.critic.value(context.observation),
            "baseline_next_value": self.critic.value(next_observation),
        }


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


def baseline_opponent_ids(pool: SnapshotPool, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("historical policy count must be positive")
    recent = tuple(snapshot for snapshot in pool.select_ids(count) if snapshot != 0)
    if count == 1:
        return (0,)
    return (0, *recent[-(count - 1):])


class DiagnosticSelfPlayRunner(SelfPlayRunner):
    """Self-play runner that tracks gameplay diagnostics for logging."""

    def __init__(self, *args, gameplay_reward: AnnealedNextoReward | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gameplay_reward = gameplay_reward
        self._diagnostics: dict[str, th.Tensor] | None = None

    def reset(self):
        observation = super().reset()
        self._diagnostics = {
            name: th.zeros((), dtype=th.float32, device=self.env.device)
            for name in (
                "steps",
                "touches",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts",
                "baseline_episodes",
                "baseline_wins",
            )
        }
        return observation

    def step(self):
        env_step = super().step()
        self._record_diagnostics(env_step)
        return env_step

    def after_update(self, timesteps: int) -> None:
        if self.opponent_pool is None or not self.opponent_pool.ready(timesteps):
            return
        self.opponent_pool.add(
            self.snapshot_policy,
            timesteps,
            protected_ids=(0,),
        )
        self.matchmaker.set_historical_ids(
            baseline_opponent_ids(self.opponent_pool, self.historical_policies)
        )
        remapped = self.matchmaker.remap_stale_opponents()
        if self.state is not None:
            keep = (~remapped).view(-1, *(1,) * (self.state.ndim - 1))
            self.state = self.state * keep

    def _baseline_mask(self) -> th.Tensor:
        baseline_matches = (
            self.matchmaker.opponent_ids.view(
                self.matchmaker.num_matches,
                self.matchmaker.players_per_match,
            )
            .eq(0)
            .any(-1)
        )
        return baseline_matches.repeat_interleave(self.matchmaker.players_per_match)

    def _record_diagnostics(self, env_step) -> None:
        if self.gameplay_reward is None or self._diagnostics is None:
            return
        touches = self.gameplay_reward.last_touches
        score = self.gameplay_reward.last_score_for_actor
        if touches is None or score is None:
            return

        learner = self.matchmaker.learner_mask
        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        no_touch_timeout = self.gameplay_reward.last_no_touch_timeout
        if no_touch_timeout is None:
            return
        no_touch_timeout = no_touch_timeout.repeat_interleave(
            self.matchmaker.players_per_match
        )
        score = score.reshape(-1)
        touches = touches.reshape(-1)
        baseline = learner & self._baseline_mask()

        self._diagnostics["steps"] += learner.sum()
        self._diagnostics["touches"] += (touches & learner).sum()
        self._diagnostics["goals_for"] += ((score > 0) & learner).sum()
        self._diagnostics["goals_against"] += ((score < 0) & learner).sum()
        self._diagnostics["episodes"] += (done & learner).sum()
        self._diagnostics["timeouts"] += (no_touch_timeout & learner).sum()
        self._diagnostics["baseline_episodes"] += (done & baseline).sum()
        self._diagnostics["baseline_wins"] += ((score > 0) & baseline).sum()

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        if self._diagnostics is None:
            return {}
        metrics = {}
        steps = self._diagnostics["steps"]
        if steps.item() > 0:
            metrics |= {
                "touches_per_1000_steps": self._diagnostics["touches"] / steps * 1000,
                "goals_for_per_1000_steps": self._diagnostics["goals_for"] / steps * 1000,
                "goals_against_per_1000_steps": self._diagnostics["goals_against"] / steps * 1000,
            }
            for name in ("steps", "touches", "goals_for", "goals_against"):
                self._diagnostics[name].zero_()

        episodes = self._diagnostics["episodes"]
        if episodes.item() > 0:
            metrics["timeout_fraction"] = self._diagnostics["timeouts"] / episodes
            self._diagnostics["episodes"].zero_()
            self._diagnostics["timeouts"].zero_()

        baseline_episodes = self._diagnostics["baseline_episodes"]
        if baseline_episodes.item() > 0:
            metrics["baseline_win_rate"] = (
                self._diagnostics["baseline_wins"] / baseline_episodes
            )
            self._diagnostics["baseline_episodes"].zero_()
            self._diagnostics["baseline_wins"].zero_()

        return {
            "Gameplay": {name: value.item() for name, value in metrics.items()}
        } if metrics else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a feed-forward PULSE latent policy with Rocket League self-play."
    )
    parser.add_argument("--distill-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--discount-half-life-seconds", type=float, default=10.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--feature-size", type=int, default=512)
    parser.add_argument("--policy-hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--exploration-std", type=float, default=0.22)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--current-fraction", type=float, default=0.5)
    parser.add_argument("--snapshot-interval", type=int, default=10_000_000)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--historical-policies", type=int, default=4)
    parser.add_argument("--demonstration-reset-fraction", type=float, default=0.5)
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--nexto-shaping-scale", type=float, default=1.0)
    parser.add_argument("--shaping-anneal-fraction", type=float, default=0.5)
    parser.add_argument("--goal-reward-scale", type=float, default=10.0)
    parser.add_argument("--touch-reward-scale", type=float, default=0.1)
    parser.add_argument("--no-touch-penalty", type=float, default=1.0)
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
        "feature_size",
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
        "discount_half_life_seconds",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.discount_half_life_seconds):
        raise ValueError("--discount-half-life-seconds must be finite")
    if not 0 < args.gae_lambda <= 1:
        raise ValueError("--gae-lambda must be in (0, 1]")
    if args.snapshot_pool_size < 3:
        raise ValueError("--snapshot-pool-size must be at least three")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("--entropy-coef must be finite and nonnegative")
    if args.bf16 and th.cuda.is_available() and not th.cuda.is_bf16_supported():
        raise ValueError("--bf16 requires BF16 support on the CUDA device")
    if not 0.0 <= args.current_fraction <= 1.0:
        raise ValueError("--current-fraction must be between zero and one")
    if not 0.0 <= args.demonstration_reset_fraction <= 1.0:
        raise ValueError("--demonstration-reset-fraction must be between zero and one")
    if not 0.0 <= args.nexto_shaping_scale <= 1.0:
        raise ValueError("--nexto-shaping-scale must be between zero and one")
    if not 0.0 < args.shaping_anneal_fraction <= 1.0:
        raise ValueError("--shaping-anneal-fraction must be in (0, 1]")
    if not math.isfinite(args.goal_reward_scale) or args.goal_reward_scale <= 0:
        raise ValueError("--goal-reward-scale must be positive and finite")
    for name in ("touch_reward_scale", "no_touch_penalty"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.historical_policies >= args.snapshot_pool_size:
        raise ValueError("--historical-policies must be smaller than the snapshot pool")
    if not args.distill_checkpoint.is_file():
        raise FileNotFoundError(args.distill_checkpoint)
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)


def build_policy(
    env,
    exploration_std: float,
    feature_size: int,
    hidden: list[int],
) -> TrainableGaussianPolicy:
    return TrainableGaussianPolicy(
        foot=LinearEncoder(feature_size, func=nn.ReLU),
        body=MLP(dims=list(hidden), func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        std=exploration_std,
    ).build(env).to(env.device)


def build_policy_and_critic(
    env,
    exploration_std: float,
    feature_size: int,
    policy_hidden: list[int],
    critic_hidden: list[int],
):
    policy = build_policy(env, exploration_std, feature_size, policy_hidden)
    critic = Critic(
        foot=LinearEncoder(feature_size, func=nn.ReLU),
        body=MLP(dims=list(critic_hidden), func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)
    return policy, critic


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)

    gamma = primitive_discount(args.frameskip, args.discount_half_life_seconds)

    reward = AnnealedNextoReward(
        1,
        1,
        shaping_scale=args.nexto_shaping_scale,
        goal_scale=args.goal_reward_scale,
        touch_scale=args.touch_reward_scale,
        no_touch_penalty=args.no_touch_penalty,
        no_touch_timeout_steps=math.ceil(
            args.no_touch_timeout_seconds * 120 / args.frameskip
        ),
    )
    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout_seconds,
        normalize=True,
        reward_funcs=(reward,),
        discrete_actions=True,
    )
    reset_dataset = load_demonstration_reset_dataset(
        args.replay_dir,
        base_env.device,
        args.frameskip,
        args.reset_state_limit,
        args.seed,
    )
    reset_sampler = DatasetResetSampler(
        reset_dataset,
        probability=args.demonstration_reset_fraction,
        seed=args.seed,
    )
    base_env.reset_state_provider = reset_sampler
    controller = FrozenPulseController.load(
        args.distill_checkpoint,
        base_env.action_codec,
        base_env.device,
        frame_skip=args.frameskip,
        bf16=args.bf16,
    )
    env = PulseLatentEnv(base_env, controller)
    policy, critic = build_policy_and_critic(
        env,
        args.exploration_std,
        args.feature_size,
        args.policy_hidden,
        args.critic_hidden,
    )

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
        historical_ids=baseline_opponent_ids(pool, args.historical_policies),
        device=env.device,
        seed=args.seed,
    )
    buffer = RolloutBuffer(
        horizon=args.rollout,
        num_envs=env.n_envs,
        device=env.device,
        copy_on_finish=False,
    )
    runner = DiagnosticSelfPlayRunner(
        env,
        policy,
        buffer,
        opponent_pool=pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=args.historical_policies,
        captures=(CriticValueCapture(critic),),
        gameplay_reward=reward,
    )

    optimizer = Adam((*policy.parameters(), *critic.parameters()), lr=args.lr)
    update = Update(
        transforms=(GAE(gamma=gamma, lambda_=args.gae_lambda),),
        sampler=RolloutMinibatches(args.batch_size, args.epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=0.2,
                value_clip=0.2,
                entropy_coef=args.entropy_coef,
                bf16=args.bf16,
            ),
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
                max(1, round(args.timesteps * args.shaping_anneal_fraction)),
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
        ("episode", "baseline_reward", "baseline reward", ".3f"),
        ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
        ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
        ("Gameplay", "baseline_win_rate", "base win", ".3f"),
        ("Reward", "nexto_shaping_scale", "reward shaping", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, format_spec)

    def log_diagnostics(trainer: Trainer) -> None:
        metrics = runner.diagnostic_metrics()
        if metrics:
            trainer.logger.update(metrics, step=trainer.clock.env_steps)

    trainer = Trainer(
        runner,
        buffer,
        Algorithm(update),
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
        update_callback=log_diagnostics,
    )

    try:
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
