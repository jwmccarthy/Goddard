import argparse
import copy
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.optim import Adam

from carl.gymnasium import (
    CARLMatchReset, CARLResetState, CARLTorchVectorEnv, REGULATION_TICKS,
)
from jarl.collect import (
    CriticCapture,
    LogProbCapture,
    RecurrentStateCapture,
    RecurrentCriticCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.envs import DatasetResetSampler
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
from jarl.sample import RecurrentRolloutMinibatches, RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE, TeamSpirit

from reward_spec import RewardSpec
from replay_resets import (
    ReplayResetProvider, load_demonstration_reset_frames, reset_index_dataset,
)
from training_checkpoint import TrainingCheckpointer
from gaifo import (
    GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE,
    add_feature_option,
    build_critic as build_gaifo_critic,
    build_policy as build_gaifo_policy,
)


BASIC_POLICY_ARCHITECTURE = "basic-gru-v1"
DEFAULT_START_KL_COEF = 0.1


@dataclass(frozen=True, eq=False)
class PolicyCheckpoint:
    architecture: str
    hidden_size: int
    state: dict[str, torch.Tensor]
    observation_size: int
    policy_layers: int = 1
    policy_gru_layers: int = 1


def _weight_count(state: dict[str, torch.Tensor], prefix: str) -> int:
    return sum(name.startswith(prefix) and name.endswith(".weight") for name in state)


def _gru_layers(state: dict[str, torch.Tensor], path: Path, model: str) -> int:
    prefix = "body.rnn.weight_ih_l"
    layers = sum(name.startswith(prefix) for name in state)
    if not layers or any(f"{prefix}{index}" not in state for index in range(layers)):
        raise ValueError(f"checkpoint has invalid {model} GRU layers: {path}")
    return layers


def _critic_checkpoint_layers(
    payload: dict, path: Path, architecture: str,
) -> tuple[int, int]:
    modules = payload["modules"]
    state = modules.get("critic", modules.get("value_function"))
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no critic weights: {path}")
    if architecture == GAIFO_ARCHITECTURE:
        layers = _weight_count(state, "body.model.")
        gru_layers = 1
    else:
        layers = _weight_count(state, "head.model.") - 1
        gru_layers = _gru_layers(state, path, "critic")
    if layers < 1:
        raise ValueError(f"checkpoint has no supported critic layers: {path}")
    config = payload.get("config", {})
    if config.get("critic_layers", layers) != layers:
        raise ValueError(f"checkpoint critic layers do not match weights: {path}")
    if config.get("critic_gru_layers", gru_layers) != gru_layers:
        raise ValueError(f"checkpoint critic GRU layers do not match weights: {path}")
    return layers, gru_layers


def _model_layers(arguments: argparse.Namespace, architecture: str) -> tuple[int, int, int, int]:
    policy_layers = getattr(arguments, "policy_layers", None)
    if policy_layers is None:
        policy_layers = 2 if architecture == BASIC_POLICY_ARCHITECTURE else 1
    critic_layers = getattr(arguments, "critic_layers", None)
    if critic_layers is None:
        critic_layers = policy_layers if architecture == GAIFO_ARCHITECTURE else 2
    policy_gru_layers = getattr(arguments, "policy_gru_layers", None)
    critic_gru_layers = getattr(arguments, "critic_gru_layers", None)
    if architecture == GAIFO_ARCHITECTURE and critic_gru_layers not in (None, 1):
        raise ValueError("--critic-gru-layers requires a recurrent critic")
    return (
        policy_layers, critic_layers,
        1 if policy_gru_layers is None else policy_gru_layers,
        1 if critic_gru_layers is None else critic_gru_layers,
    )


def policy_checkpoint(payload: dict, path: Path) -> PolicyCheckpoint:
    """Recognize BASIC and GAIFO-architecture policies by their weights."""
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a state dictionary: {path}")
    if "modules" in payload:
        modules = payload["modules"]
        state = modules.get("policy") if isinstance(modules, dict) else None
    elif "policy" in payload:
        state = payload["policy"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no policy weights: {path}")

    foot = state.get("foot.model.0.weight")
    if not isinstance(foot, torch.Tensor) or foot.ndim != 2:
        raise ValueError(f"checkpoint has no supported policy encoder: {path}")
    if "body.rnn.weight_ih_l0" in state:
        head_layers = _weight_count(state, "head.model.")
        hidden = [state.get(f"head.model.{2 * index}.weight")
                  for index in range(head_layers - 1)]
        architecture = (
            BASIC_POLICY_ARCHITECTURE if head_layers >= 2
            and all(isinstance(weight, torch.Tensor) and weight.ndim == 2
                    for weight in hidden)
            and [weight.shape[0] for weight in hidden] == (
                [foot.shape[0]] * (len(hidden) - 1) + [foot.shape[0] // 2]
            ) else GAIFO_GRU_ARCHITECTURE
        )
    elif "body.model.0.weight" in state:
        architecture = GAIFO_ARCHITECTURE
    else:
        raise ValueError(f"unsupported policy architecture in {path}")
    if architecture != BASIC_POLICY_ARCHITECTURE and "head.model.0.weight" not in state:
        raise ValueError(f"unsupported policy head in {path}")

    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"invalid checkpoint configuration in {path}")
    saved_architecture = config.get("policy_architecture", config.get("architecture"))
    if saved_architecture is not None and saved_architecture != architecture:
        raise ValueError(f"checkpoint policy architecture does not match weights: {path}")
    if "gru" in config and architecture in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
        if config["gru"] != (architecture == GAIFO_GRU_ARCHITECTURE):
            raise ValueError(f"checkpoint GRU setting does not match architecture: {path}")
    hidden_size = foot.shape[0]
    saved_hidden = config.get("policy_hidden", config.get("hidden_size"))
    if saved_hidden is not None and saved_hidden != hidden_size:
        raise ValueError(f"checkpoint hidden size does not match weights: {path}")
    policy_layers = (
        _weight_count(state, "body.model.") if architecture == GAIFO_ARCHITECTURE
        else _weight_count(state, "head.model.")
        - (architecture == BASIC_POLICY_ARCHITECTURE)
    )
    if policy_layers < 1:
        raise ValueError(f"checkpoint has no supported policy layers: {path}")
    if config.get("policy_layers", policy_layers) != policy_layers:
        raise ValueError(f"checkpoint policy layers do not match weights: {path}")
    policy_gru_layers = (
        1 if architecture == GAIFO_ARCHITECTURE
        else _gru_layers(state, path, "policy")
    )
    if config.get("policy_gru_layers", policy_gru_layers) != policy_gru_layers:
        raise ValueError(f"checkpoint policy GRU layers do not match weights: {path}")
    return PolicyCheckpoint(
        architecture, hidden_size, state, foot.shape[1],
        policy_layers, policy_gru_layers,
    )


def load_policy_checkpoint(path: Path) -> tuple[PolicyCheckpoint, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return policy_checkpoint(payload, path), payload


def configure_starting_checkpoint(
    arguments: argparse.Namespace,
) -> tuple[PolicyCheckpoint | None, bool]:
    """Choose the source policy and retain KL settings across BASIC resumes."""
    if arguments.start_checkpoint is not None and arguments.resume_checkpoint is not None:
        raise ValueError("--start-checkpoint and --resume-checkpoint are mutually exclusive")

    starting = None
    resumed_reference = False
    arguments.checkpoint_observation_size = None
    source = arguments.start_checkpoint or arguments.resume_checkpoint
    if source is not None:
        checkpoint, payload = load_policy_checkpoint(source)
        if arguments.start_checkpoint is not None:
            starting = checkpoint
        else:
            if "modules" not in payload or "optimizers" not in payload:
                raise ValueError("--resume-checkpoint requires a BASIC training checkpoint")
            stateful = payload.get("stateful", {})
            if not isinstance(stateful, dict):
                raise ValueError("invalid BASIC training checkpoint state")
            resumed_reference = "start_policy" in stateful

        if arguments.hidden_size is None:
            arguments.hidden_size = checkpoint.hidden_size
        elif arguments.hidden_size != checkpoint.hidden_size:
            raise ValueError(
                f"--policy-hidden must match checkpoint ({checkpoint.hidden_size})"
            )
        arguments.policy_architecture = checkpoint.architecture
        for name, saved in (
            ("policy_layers", checkpoint.policy_layers),
            ("policy_gru_layers", checkpoint.policy_gru_layers),
        ):
            requested = getattr(arguments, name, None)
            if requested is not None and requested != saved:
                raise ValueError(
                    f"--{name.replace('_', '-')} must match checkpoint ({saved})"
                )
            setattr(arguments, name, saved)
        if arguments.resume_checkpoint is not None:
            critic_layers, critic_gru_layers = _critic_checkpoint_layers(
                payload, source, checkpoint.architecture,
            )
            for name, saved in (
                ("critic_layers", critic_layers),
                ("critic_gru_layers", critic_gru_layers),
            ):
                requested = getattr(arguments, name, None)
                if requested is not None and requested != saved:
                    raise ValueError(
                        f"--{name.replace('_', '-')} must match checkpoint ({saved})"
                    )
                setattr(arguments, name, saved)
        arguments.checkpoint_observation_size = checkpoint.observation_size

        if arguments.start_kl_coef is None and resumed_reference:
            arguments.start_kl_coef = payload.get("config", {}).get(
                "start_kl_coef", DEFAULT_START_KL_COEF
            )

    if arguments.hidden_size is None:
        arguments.hidden_size = 256
    if source is None:
        arguments.policy_architecture = BASIC_POLICY_ARCHITECTURE
    (
        arguments.policy_layers, arguments.critic_layers,
        arguments.policy_gru_layers, arguments.critic_gru_layers,
    ) = _model_layers(arguments, arguments.policy_architecture)
    if arguments.start_kl_coef is None:
        arguments.start_kl_coef = (
            DEFAULT_START_KL_COEF if starting is not None else 0.0
        )
    if arguments.start_kl_coef > 0 and starting is None and not resumed_reference:
        raise ValueError("--start-kl-coef requires --start-checkpoint or a saved start policy")
    if getattr(arguments, "sparse", None) is None:
        arguments.sparse = (
            bool(payload.get("config", {}).get("sparse", False))
            if arguments.resume_checkpoint is not None else False
        )
    return starting, resumed_reference


class ReferenceLogitsCapture(CaptureBase):
    """Capture frozen policy logits for mask-aware KL during PPO updates."""

    def __init__(self, policy: MultiCategoricalPolicy) -> None:
        self.policy = policy
        self.state: torch.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        self.state = self.policy.initial_state(batch_size)

    @torch.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, torch.Tensor]:
        features, next_state = self.policy.body_features(context.observation, self.state)
        if next_state is not None:
            done = torch.as_tensor(
                context.env_step.done, dtype=torch.bool, device=next_state.device
            )
            self.state = next_state * (~done).view(
                -1, *((1,) * (next_state.ndim - 1))
            )
        return {"reference_logits": self.policy.head(features)}


class StartingPolicyKLLoss(PPOLoss):
    """Regularize PPO toward the frozen starting action distribution."""

    def __init__(self, policy, critic, config, coefficient: float) -> None:
        super().__init__(policy, critic, config)
        self.coefficient = coefficient

    def __call__(self, sample) -> LossOutput:
        # Reuse PPO's logits instead of running the recurrent learner a second time.
        logits = []
        hook = self.policy.head.register_forward_hook(
            lambda _module, _inputs, output: logits.append(output)
        )
        try:
            output = super().__call__(sample)
        finally:
            hook.remove()
        if len(logits) != 1:
            raise RuntimeError("PPO did not evaluate exactly one policy distribution")

        batch, _, _, _, valid = self._unpack_sample(sample)
        learner_logits = logits[0].float()
        reference_logits = batch["reference_logits"].float()
        if reference_logits.shape != learner_logits.shape:
            raise ValueError("reference logits must match the policy's action space")
        observation = batch["observation"]
        reference = self.policy._factorized_distributions(reference_logits, observation)
        learner = self.policy._factorized_distributions(learner_logits, observation)
        start_kl = sum(
            kl_divergence(original, current)
            for original, current in zip(reference, learner)
        )[valid].mean()
        penalty = self.coefficient * start_kl
        return LossOutput(
            output.loss + penalty,
            {
                **output.metrics,
                "start_kl": start_kl.detach(),
                "start_kl_penalty": penalty.detach(),
            },
        )


class DiagnosticRewardSpec(RewardSpec):
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
    def __init__(self, provider: ReplayResetProvider) -> None:
        self.provider = provider

    def __call__(self, reset_mask: torch.Tensor) -> CARLResetState | None:
        state = self.provider(reset_mask)
        if state is None:
            return None
        remaining = torch.randint(
            0,
            REGULATION_TICKS + 1,
            (len(state.simulation_indices),),
            device=reset_mask.device,
        )
        elapsed = REGULATION_TICKS - remaining
        elapsed_minutes = elapsed.float() / (120.0 * 60.0)
        scores = torch.poisson(
            elapsed_minutes[:, None].expand(-1, 2)
        ).to(torch.int32)
        return replace(
            state,
            match=CARLMatchReset(
                blue_score=scores[:, 0].contiguous(),
                orange_score=scores[:, 1].contiguous(),
                episode_ticks=elapsed.to(torch.int32),
            ),
        )


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


if not hasattr(MultiCategoricalPolicy, "build_composed"):
    def _mcp_build_composed(self, env, in_dim):
        self._build_head(in_dim, self._configure_actions(env))
        self.built = True
        return self

    MultiCategoricalPolicy.build_composed = _mcp_build_composed

if not hasattr(Critic, "build_composed"):
    def _critic_build_composed(self, env, in_dim):
        self._build_shared_head(in_dim)
        self.built = True
        return self

    Critic.build_composed = _critic_build_composed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train a BASIC Rocket League agent"
    )
    parser.add_argument(
        "--n-sim", "--num-simulations", dest="num_simulations",
        type=int, default=1024, metavar="N_SIM",
    )
    parser.add_argument("--frameskip",                  type=int,   default=8)
    parser.add_argument("--max-ticks",                  type=int,   default=36_000)
    parser.add_argument(
        "--no-touch-timeout",
        type=float,
        default=30.0,
        help="end an episode after this many seconds without a ball touch",
    )
    parser.add_argument(
        "--rollout", "--rollout-steps", dest="rollout_steps",
        type=int, default=512, metavar="ROLLOUT",
    )
    parser.add_argument("--sequence-length",            type=int,   default=16)
    parser.add_argument(
        "--policy-hidden", "--hidden-size", dest="hidden_size",
        type=int, default=None, metavar="POLICY_HIDDEN",
        help="shared policy and critic width (default: 256, inferred from a checkpoint)",
    )
    parser.add_argument(
        "--policy-layers", type=int, default=None,
        help="policy hidden layers after the encoder/GRU (default: 2 for BASIC)",
    )
    parser.add_argument(
        "--critic-layers", type=int, default=None,
        help="critic hidden layers after the encoder/GRU (default: 2 for BASIC)",
    )
    parser.add_argument(
        "--policy-gru-layers", type=int, default=None,
        help="stacked policy GRU layers (default: 1)",
    )
    parser.add_argument(
        "--critic-gru-layers", type=int, default=None,
        help="stacked critic GRU layers (default: 1)",
    )
    parser.add_argument(
        "--timesteps", "--total-timesteps", dest="total_timesteps",
        type=int, default=10_000_000_000, metavar="TIMESTEPS",
    )
    parser.add_argument(
        "--ppo-batch", "--minibatch-size", dest="minibatch_size",
        type=int, default=65_536, metavar="PPO_BATCH",
    )
    parser.add_argument(
        "--ppo-lr", "--learning-rate", dest="learning_rate",
        type=float, default=1e-5, metavar="PPO_LR",
    )
    parser.add_argument(
        "--ppo-lr-end-factor", "--learning-rate-end-factor",
        dest="learning_rate_end_factor", type=float, default=0.5,
        metavar="PPO_LR_END_FACTOR",
    )
    add_feature_option(
        parser, "--bf16", default=True,
        help="use BF16 autocast for PPO updates",
    )
    parser.add_argument(
        "--ppo-epochs", "--epochs", dest="epochs", type=int, default=32,
        metavar="PPO_EPOCHS",
    )
    parser.add_argument("--target-kl",                  type=float, default=0.02)
    parser.add_argument(
        "--entropy", "--entropy-coef", dest="entropy_coef",
        type=float, default=0.01, metavar="ENTROPY",
    )
    parser.add_argument(
        "--entropy-end", "--entropy-coef-end", dest="entropy_coef_end",
        type=float, default=0.005, metavar="ENTROPY_END",
    )
    parser.add_argument("--self-play-current",          type=float, default=0.8)
    parser.add_argument("--snapshot-interval",          type=int,   default=16)
    parser.add_argument("--opponent-pool-size",         type=int,   default=8)
    parser.add_argument("--historical-policies",        type=int,   default=4)
    parser.add_argument("--team-spirit",                type=float, default=1.0)
    parser.add_argument("--reward-scale",               type=float, default=1.0)
    parser.add_argument("--goal-score-weight",          type=float, default=10.0)
    parser.add_argument("--goal-score-weight-end",      type=float, default=10.0)
    add_feature_option(
        parser, "--sparse", default=None,
        help="focus rewards on goals, shots, aerial mechanics, boost-free speed and demos (inherited on resume)",
    )
    add_feature_option(
        parser, "--normalize-rewards", default=True,
    )
    parser.add_argument("--discount-half-life",         type=float, default=10.0)
    parser.add_argument("--discount-half-life-end",     type=float, default=20.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="constant discount override; disables the half-life schedule",
    )
    parser.add_argument(
        "--lambda", "--gae-lambda", dest="gae_lambda",
        type=float, default=0.99, metavar="LAMBDA",
    )
    parser.add_argument(
        "--log-dir", "--tensorboard-dir", dest="tensorboard_dir",
        type=Path, default=Path("runs"), metavar="LOG_DIR",
    )
    parser.add_argument("--checkpoint-dir",             type=Path,  default=Path("checkpoints"))
    parser.add_argument("--resume-checkpoint",          type=Path,  default=None)
    parser.add_argument(
        "--start-checkpoint", type=Path, default=None,
        help="start a new BASIC run from a BASIC or GAIFO policy (fresh critic and clock)",
    )
    parser.add_argument(
        "--start-kl-coef", type=float, default=None,
        help="KL(reference || current) penalty; defaults to 0.1 with --start-checkpoint, 0 otherwise",
    )
    parser.add_argument(
        "--replay-dir", "--replay-dataset", dest="replay_dataset",
        type=Path, default=Path("parsed_replays"), metavar="REPLAY_DIR",
        help="folder containing parsed replay reset states",
    )
    parser.add_argument(
        "--replay-reset-fraction", "--replay-reset-probability",
        dest="replay_reset_probability", type=float, default=0.7,
        metavar="REPLAY_RESET_FRACTION",
    )
    parser.add_argument("--reset-state-limit",          type=int,   default=100_000)
    add_feature_option(
        parser, "--normalize", default=True,
    )
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--seed",                       type=int,   default=0)
    arguments = parser.parse_args()
    if (
        arguments.hidden_size is None
        and arguments.start_checkpoint is None
        and arguments.resume_checkpoint is None
    ):
        arguments.hidden_size = 256
    if arguments.start_checkpoint is None and arguments.resume_checkpoint is None:
        (
            arguments.policy_layers, arguments.critic_layers,
            arguments.policy_gru_layers, arguments.critic_gru_layers,
        ) = _model_layers(arguments, BASIC_POLICY_ARCHITECTURE)
    return arguments


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = {
        "n-sim":                  arguments.num_simulations,
        "frameskip":              arguments.frameskip,
        "max-ticks":              arguments.max_ticks,
        "rollout":                arguments.rollout_steps,
        "sequence-length":        arguments.sequence_length,
        "policy-hidden":          arguments.hidden_size,
        "policy-layers":          arguments.policy_layers,
        "critic-layers":          arguments.critic_layers,
        "policy-gru-layers":      arguments.policy_gru_layers,
        "critic-gru-layers":      arguments.critic_gru_layers,
        "timesteps":              arguments.total_timesteps,
        "ppo-batch":              arguments.minibatch_size,
        "ppo-lr":                 arguments.learning_rate,
        "ppo-epochs":             arguments.epochs,
        "target-kl":              arguments.target_kl,
        "reward-scale":           arguments.reward_scale,
        "discount-half-life":     arguments.discount_half_life,
        "discount-half-life-end": arguments.discount_half_life_end,
        "goal-score-weight":      arguments.goal_score_weight,
        "goal-score-weight-end":  arguments.goal_score_weight_end,
        "lambda":                 arguments.gae_lambda,
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
    if (
        getattr(arguments, "policy_architecture", BASIC_POLICY_ARCHITECTURE)
        != GAIFO_ARCHITECTURE
    ):
        if arguments.rollout_steps % arguments.sequence_length:
            raise ValueError("--rollout must be divisible by --sequence-length")
        if arguments.minibatch_size % arguments.sequence_length:
            raise ValueError("--ppo-batch must be divisible by --sequence-length")
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
    if not math.isfinite(arguments.start_kl_coef) or arguments.start_kl_coef < 0:
        raise ValueError("--start-kl-coef must be finite and non-negative")
    if not math.isfinite(arguments.learning_rate_end_factor) or not (
        0.0 < arguments.learning_rate_end_factor <= 1.0
    ):
        raise ValueError("--ppo-lr-end-factor must be in (0, 1]")
    if not math.isfinite(arguments.replay_reset_probability) or not (
        0.0 <= arguments.replay_reset_probability <= 1.0
    ):
        raise ValueError("--replay-reset-fraction must be between zero and one")
    if not math.isfinite(arguments.gae_lambda) or arguments.gae_lambda > 1.0:
        raise ValueError("--lambda cannot exceed one")
    if arguments.gamma is not None and (
        not math.isfinite(arguments.gamma) or not 0.0 < arguments.gamma <= 1.0
    ):
        raise ValueError("gamma must be in (0, 1]")
    if not arguments.replay_dataset.is_dir():
        raise ValueError(f"Replay directory does not exist: {arguments.replay_dataset}")
    if (
        arguments.resume_checkpoint is not None
        and not arguments.resume_checkpoint.is_file()
    ):
        raise ValueError(
            f"Resume checkpoint does not exist: {arguments.resume_checkpoint}"
        )
    if (
        arguments.start_checkpoint is not None
        and not arguments.start_checkpoint.is_file()
    ):
        raise ValueError(f"Start checkpoint does not exist: {arguments.start_checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CARL requires a CUDA-capable GPU")
    if arguments.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--bf16 requires a CUDA device with BF16 support")
    if not math.isfinite(arguments.no_touch_timeout) or arguments.no_touch_timeout <= 0:
        raise ValueError("no-touch-timeout must be positive and finite")


def build_policy_and_critic(
    environment: CARLTorchVectorEnv,
    arguments: argparse.Namespace,
    architecture: str = BASIC_POLICY_ARCHITECTURE,
):
    policy_layers, critic_layers, policy_gru_layers, critic_gru_layers = (
        _model_layers(arguments, architecture)
    )
    if architecture in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
        if policy_gru_layers != 1:
            raise ValueError("GAIFO policy checkpoints require one GRU layer")
        gaifo_args = argparse.Namespace(
            policy_hidden=arguments.hidden_size,
            critic_hidden=arguments.hidden_size,
            policy_layers=policy_layers,
            critic_layers=critic_layers,
            gru=architecture == GAIFO_GRU_ARCHITECTURE,
        )
        actor = build_gaifo_policy(environment, gaifo_args)
        if architecture == GAIFO_ARCHITECTURE:
            return actor, build_gaifo_critic(environment, gaifo_args)
    elif architecture == BASIC_POLICY_ARCHITECTURE:
        actor_head = LinearEncoder(arguments.hidden_size, func=nn.ReLU).build(environment)
        actor_body = GRU(
            hidden_size=arguments.hidden_size, num_layers=policy_gru_layers,
        ).build(actor_head.feats)
        actor = MultiCategoricalPolicy(
            foot=actor_head,
            body=actor_body,
            head=MLP(
                dims=[arguments.hidden_size] * (policy_layers - 1)
                + [arguments.hidden_size // 2],
                func=nn.LeakyReLU,
                out_init_func=orthogonal_init(std=0.01),
            ),
            action_codec=environment.action_codec,
        )
        actor.build_composed(environment, actor_body.feats).to(environment.device)
    else:
        raise ValueError(f"unsupported starting policy architecture: {architecture}")

    critic_head = LinearEncoder(arguments.hidden_size, func=nn.ReLU).build(environment)
    critic_body = GRU(
        hidden_size=arguments.hidden_size, num_layers=critic_gru_layers,
    ).build(critic_head.feats)
    critic = Critic(
        foot=critic_head,
        body=critic_body,
        head=MLP(
            dims=[arguments.hidden_size // 2] * (critic_layers - 1)
            + [arguments.hidden_size // 4],
            func=nn.LeakyReLU,
            out_init_func=orthogonal_init(std=1.0),
        ),
    )
    critic.build_composed(environment, critic_body.feats).to(environment.device)
    return actor, critic


def build_policy_loss(
    policy,
    critic,
    entropy_coef: float,
    bf16: bool = False,
    start_kl_coef: float = 0.0,
):
    config = PPOConfig(clip=0.2, entropy_coef=entropy_coef, bf16=bf16)
    if start_kl_coef:
        return StartingPolicyKLLoss(policy, critic, config, start_kl_coef)
    return PPOLoss(policy, critic, config)


class DiagnosticSelfPlayRunner(SelfPlayRunner):
    """Self-play runner that also tracks gameplay diagnostics for logging.

    The reward callback captures touches and scores before CARL autoresets
    finished games; the learner mask is captured before the league rematches.
    """

    reward_metric_keys = (
        "reward_spec/aggregate/raw",
        "reward_spec/aggregate/zero_sum",
        "reward_spec/aggregate/normalized",
        "reward_spec/component/goal_scored",
        "reward_spec/component/win_probability",
        "reward_spec/component/boost_gain",
        "reward_spec/component/player_ball_progress",
        "reward_spec/component/touch_acceleration",
        "reward_spec/component/aerial_touch",
        "reward_spec/component/shot",
        "reward_spec/component/air_dribble_setup",
        "reward_spec/component/car_velocity",
        "reward_spec/component/aerial_carry_progress",
        "reward_spec/component/aerial_speed_progress",
        "reward_spec/component/speed_progress",
        "reward_spec/component/boost_free_speed_progress",
        "reward_spec/component/soft_lift",
        "reward_spec/component/flip_reset",
        "reward_spec/component/demo",
    )

    def __init__(
        self,
        *args,
        no_touch_timeout_steps: int,
        transition_reward: DiagnosticRewardSpec | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
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
        team_sign = torch.where(car_index == 0, 1.0, -1.0)
        score_for_actor = score * team_sign

        done = torch.as_tensor(
            env_step.done, dtype=torch.bool, device=self.env.device
        )
        truncated = torch.as_tensor(
            env_step.truncated, dtype=torch.bool, device=self.env.device
        )

        self._touch_steps += 1
        self._touch_steps[touches.any(dim=-1)] = 0
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
        reward_metrics = {
            name.removeprefix("reward_spec/"): total / count
            for name, (total, count) in self._reward_diagnostics.items()
        }
        self._reward_diagnostics.clear()
        reward = self.transition_reward
        if reward is not None and reward.normalize and reward._count:
            reward_metrics["normalizer_rms"] = (
                reward._variance.clamp_min(1e-8).sqrt().item()
            )
        if reward_metrics:
            report["RewardSpec"] = reward_metrics
        return report


def build_ppo(
    environment: CARLTorchVectorEnv,
    policy,
    critic,
    reward_function: DiagnosticRewardSpec,
    arguments: argparse.Namespace,
    checkpoint_dir: Path,
    reference_policy: MultiCategoricalPolicy | None = None,
) -> tuple[SelfPlayRunner, RolloutBuffer, Algorithm, ValueScheduler, dict]:
    if arguments.start_kl_coef and reference_policy is None:
        raise ValueError("KL penalty requires a frozen starting policy")
    policy_layers, critic_layers, policy_gru_layers, critic_gru_layers = (
        _model_layers(arguments, arguments.policy_architecture)
    )
    recurrent = policy.initial_state(1) is not None
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
        team_sizes=(1, 1),
        current_fraction=arguments.self_play_current,
        historical_ids=opponent_pool.select_ids(arguments.historical_policies),
        device=environment.device,
        seed=arguments.seed,
    )
    no_touch_timeout_steps = math.ceil(
        arguments.no_touch_timeout * 120.0 / arguments.frameskip
    )
    captures = [LogProbCapture()]
    if recurrent:
        captures.extend((RecurrentStateCapture(), RecurrentCriticCapture(critic)))
    else:
        captures.append(CriticCapture(critic))
    if reference_policy is not None and arguments.start_kl_coef:
        captures.append(ReferenceLogitsCapture(reference_policy))

    runner = DiagnosticSelfPlayRunner(
        env=environment,
        policy=policy,
        buffer=rollout,
        opponent_pool=opponent_pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=arguments.historical_policies,
        captures=captures,
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
        policy,
        critic,
        arguments.entropy_coef,
        arguments.bf16,
        arguments.start_kl_coef,
    )
    if recurrent:
        fields = (
            "observation", "action", "advantage", "old_log_prob",
            "baseline_value", "returns",
        )
        if arguments.start_kl_coef:
            fields += ("reference_logits",)
        sampler = RecurrentRolloutMinibatches(
            sequence_length=arguments.sequence_length,
            sequences_per_batch=(
                arguments.minibatch_size // arguments.sequence_length
            ),
            epochs=arguments.epochs,
            fields=fields,
        )
    else:
        sampler = RolloutMinibatches(arguments.minibatch_size, arguments.epochs)
    update = KLLimitedUpdate(
        target_kl=arguments.target_kl,
        transforms=(
            TeamSpirit(
                num_matches=environment.n_sim,
                team_sizes=(1, 1),
                spirit=arguments.team_spirit,
            ),
            gae,
        ),
        sampler=sampler,
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
        section="PPO",
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
        "stateful": (
            {"start_policy": reference_policy}
            if reference_policy is not None else {}
        ),
        "config": {
            "policy_architecture": arguments.policy_architecture,
            "hidden_size": arguments.hidden_size,
            "policy_layers": policy_layers,
            "critic_layers": critic_layers,
            "policy_gru_layers": policy_gru_layers,
            "critic_gru_layers": critic_gru_layers,
            "start_kl_coef": arguments.start_kl_coef,
            "sparse": arguments.sparse,
        },
    }


def build_training_environment(
    arguments: argparse.Namespace,
    reset_provider: SyntheticMatchResetProvider,
) -> CARLTorchVectorEnv:
    saved_size = getattr(arguments, "checkpoint_observation_size", None)
    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=1,
        n_orange=1,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=arguments.max_ticks,
        no_touch_timeout_seconds=arguments.no_touch_timeout,
        synchronize=False,
        reward_scale=arguments.reward_scale,
        reset_state_provider=reset_provider,
        normalize=arguments.normalize,
        discrete_actions=True,
    )
    actual_size = environment.single_observation_space.shape[0]
    if saved_size is not None and saved_size != actual_size:
        environment.close()
        raise ValueError(
            f"checkpoint policy needs {saved_size} observation features, "
            f"but CARL provides {actual_size}"
        )
    return environment


def main() -> None:
    arguments = parse_arguments()
    starting, resumed_reference = configure_starting_checkpoint(arguments)
    validate_arguments(arguments)
    torch.manual_seed(arguments.seed)
    run_id = arguments.run_name or datetime.now().strftime(
        "goddard-%Y%m%d-%H%M%S"
    )
    run_dir = arguments.tensorboard_dir / run_id
    checkpoint_dir = arguments.checkpoint_dir / run_id

    replay_frames, replay_internal = load_demonstration_reset_frames(
        arguments.replay_dataset,
        "cuda:0",
        arguments.frameskip,
        arguments.reset_state_limit,
        arguments.seed,
        require_frame_skip_match=False,
    )
    reset_sampler = DatasetResetSampler(
        reset_index_dataset(torch.arange(
            len(replay_frames), device=replay_frames.device,
        )),
        probability=arguments.replay_reset_probability,
        seed=arguments.seed,
    )
    reset_provider = SyntheticMatchResetProvider(
        ReplayResetProvider(reset_sampler, replay_frames, replay_internal)
    )
    environment = build_training_environment(arguments, reset_provider)
    reward_function = environment.register_reward(
        DiagnosticRewardSpec(
            normalize=arguments.normalize_rewards,
            log_diagnostics=True,
            sparse=arguments.sparse,
            frameskip=arguments.frameskip,
        )
    )
    try:
        if arguments.total_timesteps < environment.n_envs:
            raise ValueError(
                "--timesteps must include at least one vector step "
                f"({environment.n_envs:,} actor timesteps)"
            )
        policy, critic = build_policy_and_critic(
            environment, arguments, arguments.policy_architecture
        )
        modules = {
            "policy": policy,
            "critic": critic,
        }
        if starting is not None:
            policy.load_state_dict(starting.state)
        elif arguments.resume_checkpoint is not None:
            TrainingCheckpointer.load_modules(
                arguments.resume_checkpoint,
                modules,
                environment.device,
            )
        reference_policy = (
            copy.deepcopy(policy).eval().requires_grad_(False)
            if (starting is not None and arguments.start_kl_coef) or resumed_reference
            else None
        )
        if reference_policy is not None and isinstance(reference_policy.body, GRU):
            reference_policy.body.rnn.flatten_parameters()
        runner, rollout, learner, value_scheduler, training_objects = build_ppo(
            environment,
            policy,
            critic,
            reward_function,
            arguments,
            checkpoint_dir,
            reference_policy,
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
            ("RewardSpec", "aggregate/zero_sum", "raw reward", ".3f"),
            ("RewardSpec", "normalizer_rms", "reward RMS", ".3f"),
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
        if arguments.start_kl_coef:
            logger.register_progress_metric("PPO", "start_kl", "start KL", ".4f")
            logger.register_progress_metric(
                "PPO", "start_kl_penalty", "start penalty", ".4f"
            )

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
        environment.close()


if __name__ == "__main__":
    main()
