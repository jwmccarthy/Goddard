import argparse
import math
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv, REGULATION_TICKS
from jarl.collect import (
    LogProbCapture,
    RecurrentStateCapture,
    RecurrentCriticCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    SPOConfig,
    SPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.modules import orthogonal_init
from jarl.runtime import (
    ConstantSchedule,
    LinearSchedule,
    MappedSchedule,
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE, TeamSpirit

from rewards import SeerReward
from replay_resets import load_demonstration_reset_dataset


class DiagnosticSeerReward(SeerReward):
    """Keep transition events available after CARL autoresets finished games."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_touches: torch.Tensor | None = None
        self.last_score_delta: torch.Tensor | None = None

    def __call__(self, context):
        self.last_touches = context.current.car_ball_touches.detach().clone()
        self.last_score_delta = context.events.score_delta.detach().clone()
        return super().__call__(context)


class SyntheticMatchResetProvider:
    def __init__(self, provider) -> None:
        self.provider = provider

    def __call__(self, reset_mask: torch.Tensor):
        sample = self.provider(reset_mask)
        if sample is None:
            return None
        state = dict(sample)
        indices = state["simulation_indices"]
        remaining = torch.randint(
            0,
            REGULATION_TICKS + 1,
            (len(indices),),
            device=reset_mask.device,
        )
        elapsed = REGULATION_TICKS - remaining
        elapsed_minutes = elapsed.float() / (120.0 * 60.0)
        scores = torch.poisson(
            elapsed_minutes[:, None].expand(-1, 2)
        ).to(torch.int32)
        state.update(
            simulation_indices=indices,
            blue_score=scores[:, 0].contiguous(),
            orange_score=scores[:, 1].contiguous(),
            episode_ticks=elapsed.to(torch.int32),
        )
        return state


class KLLimitedUpdate(Update):
    """Stop PPO minibatches when the on-policy ratio has drifted too far."""

    def __init__(self, *, target_kl: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target_kl = target_kl
        self._early_stopped = False
        self._stop_kl = 0.0
        self._minibatches = 0

    def _process_minibatches(self, batch):
        totals = {}
        self._minibatches = 0
        self._early_stopped = False
        self._stop_kl = 0.0
        for sample in self.sampler(batch):
            output = self._normalize_loss_output(self.loss(sample))
            kl = self._to_float(output.metrics["approx_kl"])
            if not math.isfinite(kl):
                raise RuntimeError("non-finite PPO approximate KL")
            if kl > self.target_kl:
                self._early_stopped = True
                self._stop_kl = kl
                break
            self.optimizer_step(output.loss)
            self._accumulate_metrics(totals, output.metrics)
            self._minibatches += 1
        return totals, self._minibatches

    def update(self, experience):
        metrics = super().update(experience)
        metrics[self.section].update(
            kl_early_stop=float(self._early_stopped),
            kl_stop_value=self._stop_kl,
            optimizer_minibatches=float(self._minibatches),
        )
        return metrics


from training_checkpoint import TrainingCheckpointer

from jarl.modules.operator import Critic as _Critic
from jarl.modules.policy import MultiCategoricalPolicy as _MCP

if not hasattr(_MCP, "build_composed"):
    def _mcp_build_composed(self, env, in_dim):
        self._build_head(in_dim, self._configure_actions(env))
        self.built = True
        return self

    _MCP.build_composed = _mcp_build_composed

if not hasattr(_Critic, "build_composed"):
    def _critic_build_composed(self, env, in_dim):
        self._build_shared_head(in_dim)
        self.built = True
        return self

    _Critic.build_composed = _critic_build_composed

from jarl.modules.operator import Critic as _Critic
from jarl.modules.policy import MultiCategoricalPolicy as _MCP

if not hasattr(_MCP, "build_composed"):
    def _mcp_build_composed(self, env, in_dim):
        self._build_head(in_dim, self._configure_actions(env))
        self.built = True
        return self

    _MCP.build_composed = _mcp_build_composed

if not hasattr(_Critic, "build_composed"):
    def _critic_build_composed(self, env, in_dim):
        self._build_shared_head(in_dim)
        self.built = True
        return self

    _Critic.build_composed = _critic_build_composed



def parse_arguments(algorithm: str = "ppo") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train a {algorithm.upper()} Rocket League agent"
    )
    parser.add_argument("--num-simulations",            type=int,   default=1024)
    parser.add_argument("--n-blue",                     type=int,   default=1)
    parser.add_argument("--n-orange",                   type=int,   default=1)
    parser.add_argument("--frameskip",                  type=int,   default=8)
    parser.add_argument("--max-ticks",                  type=int,   default=36_000)
    parser.add_argument(
        "--no-touch-timeout",
        type=float,
        default=30.0,
        help="end an episode after this many seconds without a ball touch",
    )
    parser.add_argument("--rollout-steps",              type=int,   default=512)
    parser.add_argument("--sequence-length",            type=int,   default=16)
    parser.add_argument("--hidden-size",                type=int,   default=256)
    parser.add_argument("--total-timesteps",            type=int,   default=10_000_000_000)
    parser.add_argument("--minibatch-size",             type=int,   default=65_536)
    parser.add_argument("--learning-rate",              type=float, default=1e-5)
    parser.add_argument("--learning-rate-end-factor",   type=float, default=0.5)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use BF16 autocast for PPO updates",
    )
    parser.add_argument("--epochs",                     type=int,   default=32)
    parser.add_argument("--target-kl",                  type=float, default=0.02)
    parser.add_argument("--entropy-coef",               type=float, default=0.01)
    parser.add_argument("--entropy-coef-end",           type=float, default=0.005)
    parser.add_argument("--self-play-current",          type=float, default=0.8)
    parser.add_argument("--snapshot-interval",          type=int,   default=16)
    parser.add_argument("--opponent-pool-size",         type=int,   default=8)
    parser.add_argument("--historical-policies",        type=int,   default=4)
    parser.add_argument("--team-spirit",                type=float, default=1.0)
    parser.add_argument("--reward-scale",               type=float, default=1.0)
    parser.add_argument("--goal-score-weight",          type=float, default=10.0)
    parser.add_argument("--goal-score-weight-end",      type=float, default=10.0)
    parser.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--discount-half-life",         type=float, default=10.0)
    parser.add_argument("--discount-half-life-end",     type=float, default=20.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="constant discount override; disables the half-life schedule",
    )
    parser.add_argument("--gae-lambda",                 type=float, default=0.99)
    parser.add_argument("--tensorboard-dir",            type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",             type=Path,  default=Path("checkpoints"))
    parser.add_argument("--resume-checkpoint",          type=Path,  default=None)
    parser.add_argument(
        "--replay-dataset",
        type=Path,
        default=Path("parsed_replays"),
    )
    parser.add_argument("--replay-reset-probability",   type=float, default=0.7)
    parser.add_argument("--reset-state-limit",          type=int,   default=100_000)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--seed",                       type=int,   default=0)
    # Recent-script aliases (same destinations, hidden from help).
    parser.add_argument("--replay-dir", dest="replay_dataset", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--n-sim", dest="num_simulations", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--no-touch-timeout-seconds", dest="no_touch_timeout", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rollout", dest="rollout_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", dest="minibatch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--lr-end-factor", dest="learning_rate_end_factor", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--timesteps", dest="total_timesteps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--current-fraction", dest="self_play_current", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--snapshot-pool-size", dest="opponent_pool_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--log-dir", dest="tensorboard_dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--replay-reset-fraction", dest="replay_reset_probability", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--demonstration-reset-fraction", dest="replay_reset_probability", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--policy-hidden", dest="hidden_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--critic-hidden", dest="hidden_size", type=int, default=argparse.SUPPRESS)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = {
        "num-simulations":        arguments.num_simulations,
        "n-blue":                 arguments.n_blue,
        "n-orange":               arguments.n_orange,
        "frameskip":              arguments.frameskip,
        "max-ticks":              arguments.max_ticks,
        "rollout-steps":          arguments.rollout_steps,
        "sequence-length":        arguments.sequence_length,
        "hidden-size":            arguments.hidden_size,
        "total-timesteps":        arguments.total_timesteps,
        "minibatch-size":         arguments.minibatch_size,
        "learning-rate":          arguments.learning_rate,
        "epochs":                 arguments.epochs,
        "target-kl":              arguments.target_kl,
        "reward-scale":           arguments.reward_scale,
        "discount-half-life":     arguments.discount_half_life,
        "discount-half-life-end": arguments.discount_half_life_end,
        "goal-score-weight":      arguments.goal_score_weight,
        "goal-score-weight-end":  arguments.goal_score_weight_end,
        "gae-lambda":             arguments.gae_lambda,
        "snapshot-interval":      arguments.snapshot_interval,
        "opponent-pool-size":     arguments.opponent_pool_size,
        "historical-policies":    arguments.historical_policies,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not math.isfinite(value) or value <= 0
    ]
    if invalid:
        raise ValueError(f"Arguments must be positive: {', '.join(invalid)}")
    if arguments.rollout_steps % arguments.sequence_length:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if arguments.minibatch_size % arguments.sequence_length:
        raise ValueError("minibatch-size must be divisible by sequence-length")
    if arguments.opponent_pool_size < 3:
        raise ValueError("opponent-pool-size must be at least three")
    if arguments.historical_policies >= arguments.opponent_pool_size:
        raise ValueError("historical-policies must be smaller than opponent-pool-size")
    if not math.isfinite(arguments.self_play_current) or not (
        0.0 <= arguments.self_play_current <= 1.0
    ):
        raise ValueError("self-play-current must be between zero and one")
    if not math.isfinite(arguments.team_spirit) or not (
        0.0 <= arguments.team_spirit <= 1.0
    ):
        raise ValueError("team-spirit must be between zero and one")
    if (
        not math.isfinite(arguments.entropy_coef)
        or not math.isfinite(arguments.entropy_coef_end)
        or arguments.entropy_coef < 0
        or arguments.entropy_coef_end < 0
    ):
        raise ValueError("entropy coefficients cannot be negative")
    if not math.isfinite(arguments.learning_rate_end_factor) or not (
        0.0 < arguments.learning_rate_end_factor <= 1.0
    ):
        raise ValueError("learning-rate-end-factor must be in (0, 1]")
    if not math.isfinite(arguments.replay_reset_probability) or not (
        0.0 <= arguments.replay_reset_probability <= 1.0
    ):
        raise ValueError("replay-reset-probability must be between zero and one")
    if not math.isfinite(arguments.gae_lambda) or arguments.gae_lambda > 1.0:
        raise ValueError("gae-lambda cannot exceed one")
    if arguments.gamma is not None and (
        not math.isfinite(arguments.gamma) or not 0.0 < arguments.gamma <= 1.0
    ):
        raise ValueError("gamma must be in (0, 1]")
    if arguments.n_blue != 1 or arguments.n_orange != 1:
        raise ValueError("The replay dataset currently supports only 1v1 training")
    if not arguments.replay_dataset.is_dir():
        raise ValueError(f"Replay dataset does not exist: {arguments.replay_dataset}")
    if (
        arguments.resume_checkpoint is not None
        and not arguments.resume_checkpoint.is_file()
    ):
        raise ValueError(
            f"Resume checkpoint does not exist: {arguments.resume_checkpoint}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CARL requires a CUDA-capable GPU")
    if arguments.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--bf16 requires a CUDA device with BF16 support")
    if not math.isfinite(arguments.no_touch_timeout) or arguments.no_touch_timeout <= 0:
        raise ValueError("no-touch-timeout must be positive and finite")


def build_policy(env, hidden_size: int, recurrent: bool = True):
    """Compatibility entry point for the checkpoint viewer."""
    from types import SimpleNamespace

    policy, _ = build_policy_and_critic(
        env, SimpleNamespace(hidden_size=hidden_size)
    )
    return policy


def build_policy_and_critic(
    environment: CARLTorchVectorEnv,
    arguments: argparse.Namespace,
):
    actor_head = LinearEncoder(arguments.hidden_size, func=nn.ReLU).build(environment)
    actor_body = GRU(hidden_size=arguments.hidden_size).build(actor_head.feats)
    actor = MultiCategoricalPolicy(
        foot=actor_head,
        body=actor_body,
        head=MLP(
            dims=[arguments.hidden_size, arguments.hidden_size // 2],
            func=nn.LeakyReLU,
            out_init_func=orthogonal_init(std=0.01),
        ),
        action_codec=environment.action_codec,
    )
    actor.build_composed(environment, actor_body.feats).to(environment.device)

    critic_head = LinearEncoder(arguments.hidden_size, func=nn.ReLU).build(environment)
    critic_body = GRU(hidden_size=arguments.hidden_size).build(critic_head.feats)
    critic = Critic(
        foot=critic_head,
        body=critic_body,
        head=MLP(
            dims=[arguments.hidden_size // 2, arguments.hidden_size // 4],
            func=nn.LeakyReLU,
            out_init_func=orthogonal_init(std=1.0),
        ),
    )
    critic.build_composed(environment, critic_body.feats).to(environment.device)
    return actor, critic


def build_policy_loss(
    algorithm: str,
    policy,
    critic,
    entropy_coef: float,
    bf16: bool = False,
):
    if algorithm == "ppo":
        return PPOLoss(
            policy,
            critic,
            PPOConfig(clip=0.2, entropy_coef=entropy_coef, bf16=bf16),
        )
    if algorithm == "spo":
        return SPOLoss(
            policy,
            critic,
            SPOConfig(ratio_epsilon=0.2, entropy_coef=entropy_coef),
        )
    raise ValueError(f"unknown policy optimization algorithm: {algorithm}")


class DiagnosticSelfPlayRunner(SelfPlayRunner):
    """Self-play runner that also tracks gameplay diagnostics for logging.

    The reward callback captures touches and scores before CARL autoresets
    finished games; the learner mask is captured before the league rematches.
    """

    reward_metric_keys = (
        "seer/aggregate/raw",
        "seer/aggregate/outcome_adjusted",
        "seer/aggregate/normalized",
        "seer/component/goal_scored",
        "seer/component/win_probability",
        "seer/component/boost_gain",
        "seer/component/player_ball_progress",
        "seer/component/touch_acceleration",
        "seer/component/aerial_touch",
    )

    def __init__(
        self,
        *args,
        n_blue: int = 1,
        no_touch_timeout_steps: int | None = None,
        transition_reward: DiagnosticSeerReward | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_blue = n_blue
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.transition_reward = transition_reward
        self._diagnostics: dict[str, torch.Tensor] | None = None
        self._touch_steps: torch.Tensor | None = None
        self._reward_diagnostics: dict[str, tuple[float, int]] = {}

    def reset(self):
        observation = super().reset()
        self._diagnostics = {
            name: torch.zeros((), dtype=torch.float32, device=self.env.device)
            for name in (
                "steps",
                "touches",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts",
            )
        }
        self._touch_steps = torch.zeros(
            self.env.n_sim, dtype=torch.long, device=self.env.device
        )
        self._reward_diagnostics.clear()
        return observation

    def step(self):
        learner_mask = self.matchmaker.learner_mask.clone()
        env_step = super().step()
        self._record_diagnostics(env_step, learner_mask)
        for name in self.reward_metric_keys:
            values = env_step.info.get(name, ())
            if values:
                total, count = self._reward_diagnostics.get(name, (0.0, 0))
                self._reward_diagnostics[name] = total + sum(values), count + len(values)
        return env_step

    def _record_diagnostics(self, env_step, learner_mask: torch.Tensor) -> None:
        if self._diagnostics is None or self._touch_steps is None:
            return

        reward = self.transition_reward
        if (
            reward is None
            or reward.last_touches is None
            or reward.last_score_delta is None
        ):
            raise RuntimeError("transition diagnostics require a recorded reward")
        touches = reward.last_touches
        score = reward.last_score_delta
        n_cars = touches.shape[-1]

        touch = touches.reshape(-1)
        score = score.repeat_interleave(n_cars)
        car_index = torch.arange(touch.shape[0], device=touch.device) % n_cars
        team_sign = torch.where(car_index < self.n_blue, 1.0, -1.0)
        score_for_actor = score * team_sign

        done = torch.as_tensor(
            env_step.done, dtype=torch.bool, device=self.env.device
        )
        truncated = torch.as_tensor(
            env_step.truncated, dtype=torch.bool, device=self.env.device
        )

        self._touch_steps += 1
        self._touch_steps[touches.any(dim=-1)] = 0
        if self.no_touch_timeout_steps is None:
            timeout = torch.zeros_like(done)
        else:
            simulation_timeout = truncated.reshape(-1, n_cars).all(dim=-1) & (
                self._touch_steps >= self.no_touch_timeout_steps
            )
            timeout = simulation_timeout.repeat_interleave(n_cars)
        self._touch_steps[done.reshape(-1, n_cars).any(dim=-1)] = 0

        learner = learner_mask
        self._diagnostics["steps"] += learner.sum()
        self._diagnostics["touches"] += (touch & learner).sum()
        self._diagnostics["goals_for"] += ((score_for_actor > 0) & learner).sum()
        self._diagnostics["goals_against"] += (
            (score_for_actor < 0) & learner
        ).sum()
        self._diagnostics["episodes"] += (done & learner).sum()
        self._diagnostics["timeouts"] += (timeout & learner).sum()

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        if self._diagnostics is None:
            return {}

        metrics: dict[str, torch.Tensor] = {}
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

        report = {}
        if metrics:
            report["Gameplay"] = {
                name: value.item() for name, value in metrics.items()
            }
        seer = {
            name.removeprefix("seer/"): total / count
            for name, (total, count) in self._reward_diagnostics.items()
        }
        self._reward_diagnostics.clear()
        reward = self.transition_reward
        if reward is not None and reward.normalize and reward._count:
            seer["normalizer_rms"] = (
                reward._variance + reward._mean.square()
            ).clamp_min(1e-8).sqrt().item()
        if seer:
            report["Seer"] = seer
        return report


def build_ppo(
    environment: CARLTorchVectorEnv,
    policy,
    critic,
    reward_function: DiagnosticSeerReward,
    arguments: argparse.Namespace,
    checkpoint_dir: Path,
    algorithm: str = "ppo",
) -> tuple[SelfPlayRunner, RolloutBuffer, Algorithm, ValueScheduler, dict]:
    rollout = RolloutBuffer(
        horizon=arguments.rollout_steps,
        num_envs=environment.n_envs,
        device=environment.device,
        copy_on_finish=False,
    )
    snapshot_rollout_timesteps = int(
        environment.n_envs
        * (1.0 + arguments.self_play_current)
        / 2.0
        * arguments.rollout_steps
    )
    opponent_pool = SnapshotPool(
        policy=policy,
        max_size=arguments.opponent_pool_size,
        snapshot_interval=(
            snapshot_rollout_timesteps * arguments.snapshot_interval
        ),
        active_cache_size=max(4, arguments.historical_policies * 2),
        seed=arguments.seed,
        checkpoint_dir=checkpoint_dir,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=environment.n_sim,
        team_sizes=(arguments.n_blue, arguments.n_orange),
        current_fraction=arguments.self_play_current,
        historical_ids=opponent_pool.select_ids(arguments.historical_policies),
        device=environment.device,
        seed=arguments.seed,
    )
    no_touch_timeout_steps = math.ceil(
        arguments.no_touch_timeout * 120.0 / arguments.frameskip
    )
    runner = DiagnosticSelfPlayRunner(
        env=environment,
        policy=policy,
        buffer=rollout,
        opponent_pool=opponent_pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=arguments.historical_policies,
        captures=(
            LogProbCapture(),
            RecurrentStateCapture(),
            RecurrentCriticCapture(critic),
        ),
        n_blue=arguments.n_blue,
        no_touch_timeout_steps=no_touch_timeout_steps,
        transition_reward=reward_function,
    )

    policy_optimizer = Adam(policy.parameters(), lr=arguments.learning_rate)
    critic_optimizer = Adam(critic.parameters(), lr=arguments.learning_rate)
    actions_per_second = 120.0 / arguments.frameskip
    initial_gamma = arguments.gamma or 0.5 ** (
        1.0 / (actions_per_second * arguments.discount_half_life)
    )
    gae = GAE(gamma=initial_gamma, lambda_=arguments.gae_lambda)
    policy_loss = build_policy_loss(
        algorithm,
        policy,
        critic,
        arguments.entropy_coef,
        arguments.bf16,
    )
    update_type = KLLimitedUpdate if algorithm == "ppo" else Update
    update = update_type(
        **({"target_kl": arguments.target_kl} if algorithm == "ppo" else {}),
        transforms=(
            TeamSpirit(
                num_matches=environment.n_sim,
                team_sizes=(arguments.n_blue, arguments.n_orange),
                spirit=arguments.team_spirit,
            ),
            gae,
        ),
        sampler=RecurrentRolloutMinibatches(
            sequence_length=arguments.sequence_length,
            sequences_per_batch=(
                arguments.minibatch_size // arguments.sequence_length
            ),
            epochs=arguments.epochs,
            fields=(
                "observation",
                "action",
                "advantage",
                "old_log_prob",
                "baseline_value",
                "returns",
            ),
        ),
        loss=policy_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                policy,
                policy_optimizer,
                max_grad_norm=0.5,
            ),
            OptimizerStep(
                critic,
                critic_optimizer,
                max_grad_norm=0.5,
            ),
        ),
        section=algorithm.upper(),
    )
    learning_rate = LinearSchedule(
        arguments.learning_rate,
        arguments.learning_rate * arguments.learning_rate_end_factor,
    )
    entropy_coef = LinearSchedule(
        arguments.entropy_coef,
        arguments.entropy_coef_end,
    )
    if arguments.gamma is None:
        half_life = LinearSchedule(
            arguments.discount_half_life,
            arguments.discount_half_life_end,
        )
        gamma = MappedSchedule(
            half_life,
            lambda seconds: 0.5 ** (1.0 / (actions_per_second * seconds)),
        )
    else:
        half_life = None
        gamma = ConstantSchedule(arguments.gamma)
    goal_score_weight = LinearSchedule(
        arguments.goal_score_weight,
        arguments.goal_score_weight_end,
    )

    def set_learning_rate(value: float) -> None:
        for optimizer in (policy_optimizer, critic_optimizer):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = value

    def set_entropy_coef(value: float) -> None:
        policy_loss.config = replace(policy_loss.config, entropy_coef=value)

    scheduled_values = [
        ScheduledValue("learning_rate", learning_rate, set_learning_rate),
        ScheduledValue("entropy_coef", entropy_coef, set_entropy_coef),
    ]

    if half_life is not None:
        scheduled_values.append(
            ScheduledValue.metric("discount_half_life", half_life)
        )

    scheduled_values.extend(
        (
            ScheduledValue.attribute(
                "gamma",
                gae,
                "gamma",
                gamma,
            ),
            ScheduledValue(
                "goal_score_weight",
                goal_score_weight,
                reward_function.set_goal_scored_weight,
            ),
        )
    )
    value_scheduler = ValueScheduler(*scheduled_values)
    return runner, rollout, Algorithm(update), value_scheduler, {
        "modules": {
            "policy": policy,
            "critic": critic,
        },
        "optimizers": {
            "policy": policy_optimizer,
            "critic": critic_optimizer,
        },
    }


def main(algorithm: str = "ppo") -> None:
    arguments = parse_arguments(algorithm)
    validate_arguments(arguments)
    torch.manual_seed(arguments.seed)
    prefix = "goddard" if algorithm == "ppo" else f"goddard-{algorithm}"
    run_id = arguments.run_name or datetime.now().strftime(
        f"{prefix}-%Y%m%d-%H%M%S"
    )
    run_dir = arguments.tensorboard_dir / run_id
    checkpoint_dir = arguments.checkpoint_dir / run_id

    replay_dataset = load_demonstration_reset_dataset(
        arguments.replay_dataset,
        "cuda:0",
        arguments.frameskip,
        arguments.reset_state_limit,
        arguments.seed,
        require_frame_skip_match=False,
    )
    reset_sampler = DatasetResetSampler(
        replay_dataset,
        probability=arguments.replay_reset_probability,
        seed=arguments.seed,
    )
    reset_sampler = SyntheticMatchResetProvider(reset_sampler)
    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=arguments.n_blue,
        n_orange=arguments.n_orange,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=arguments.max_ticks,
        no_touch_timeout_seconds=arguments.no_touch_timeout,
        synchronize=False,
        reward_scale=arguments.reward_scale,
        reset_state_provider=reset_sampler,
        normalize=arguments.normalize,
        discrete_actions=True,
    )
    reward_function = environment.register_reward(
        DiagnosticSeerReward(
            n_blue=arguments.n_blue,
            n_orange=arguments.n_orange,
            normalize=arguments.normalize_rewards,
            log_diagnostics=True,
        )
    )
    evaluator = None
    try:
        if arguments.total_timesteps < environment.n_envs:
            raise ValueError(
                "total-timesteps must include at least one vector step "
                f"({environment.n_envs:,} actor timesteps)"
            )
        policy, critic = build_policy_and_critic(environment, arguments)
        modules = {
            "policy": policy,
            "critic": critic,
        }
        if arguments.resume_checkpoint is not None:
            TrainingCheckpointer.load_modules(
                arguments.resume_checkpoint,
                modules,
                environment.device,
            )
        runner, rollout, learner, value_scheduler, training_objects = build_ppo(
            environment,
            policy,
            critic,
            reward_function,
            arguments,
            checkpoint_dir,
            algorithm,
        )
        logger = Logger(log_dir=str(run_dir))

        for section, key, label, format_spec in (
            ("PPO", "policy_loss", "policy loss", ".4f"),
            ("PPO", "critic_loss", "critic loss", ".4f"),
            ("PPO", "entropy", "entropy", ".3f"),
            ("PPO", "approx_kl", "approx KL", ".4f"),
            ("PPO", "kl_early_stop", "KL stop", ".0f"),
            ("PPO", "optimizer_minibatches", "minibatches", ".0f"),
            ("episode", "current_reward", "current reward", ".3f"),
            ("episode", "historical_reward", "historical reward", ".3f"),
            ("Seer", "aggregate/outcome_adjusted", "raw reward", ".3f"),
            ("Seer", "normalizer_rms", "reward RMS", ".3f"),
            ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
            ("Gameplay", "goals_for_per_1000_steps", "goals for/1k", ".3f"),
            ("Gameplay", "goals_against_per_1000_steps", "goals against/1k", ".3f"),
            ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
            ("Schedule", "learning_rate", "learning rate", ".2e"),
            ("Schedule", "entropy_coef", "entropy coef", ".4f"),
            ("Schedule", "gamma", "gamma", ".5f"),
            ("Schedule", "discount_half_life", "discount half-life", ".1f"),
            ("Schedule", "goal_score_weight", "goal weight", ".2f"),
        ):
            logger.register_progress_metric(section, key, label, format_spec)

        training_checkpointer = TrainingCheckpointer(
            checkpoint_dir / "training_latest.pt",
            **training_objects,
        )

        def update_callback(trainer: Trainer) -> None:
            training_checkpointer(trainer)
            metrics = runner.diagnostic_metrics()
            if metrics:
                trainer.logger.update(metrics, step=trainer.clock.env_steps)

        trainer = Trainer(
            runner,
            rollout,
            learner,
            OnPolicySchedule(),
            logger=logger,
            checkpoint=None,
            value_scheduler=value_scheduler,
            update_callback=update_callback,
        )
        if arguments.resume_checkpoint is not None:
            trainer.clock = training_checkpointer.load(
                arguments.resume_checkpoint,
                environment.device,
            )
        trainer.run(arguments.total_timesteps)
        training_checkpointer(trainer)
        torch.save(policy.state_dict(), checkpoint_dir / "actor_critic_final.pt")
    finally:
        if evaluator is not None:
            evaluator.close()
        environment.close()


if __name__ == "__main__":
    main()
