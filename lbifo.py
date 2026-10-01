"""LBIfO: state-only 1v1 behavior pretraining, hindsight skills, and latent plans.

The representation corpus can contain lower-ranked players. From low-level
training onward the expert targets and sequence prior read only the pro 1v1
corpus. Replay resets have a separate optional source and never become expert
labels merely by being used as starts.
"""

import argparse
import bisect
import copy
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv

from gaifo import N_CARS, SCENE_SIZE
from lbifo_data import ExpertCorpus, PlaySequence, TrajectoryMemory
from lbifo_dynamics import DynamicsPairs
from lbifo_planning import (
    CalibratedSurprise, InferredSequence, PlanValue,
    SemiMarkovSegmenter, SphericalPlanPrior,
)
from lbifo_repr import SceneRepresentation, ema_update
from lbifo_skill import (
    BehaviorPolicy, ExpertResetProvider, JointPlanController,
    expert_prior_examples, hazard_examples, hindsight_labels,
)
from replay_resets import load_demonstration_reset_dataset


LBIFO_ARCHITECTURE = "lbifo-state-only-1v1-v1"


def replay_folder(folder: Path, frameskip: int) -> Path:
    """Require parsed 1v1 scene files, not similarly named 2v2 replays."""
    if folder.is_dir():
        candidates = [folder, folder / f"pro_1v1_fs{frameskip}", folder / "pro_1v1_fs4"]
        for candidate in candidates:
            for file in candidate.glob("*.npy"):
                source = np.load(file, mmap_mode="r")
                if source.ndim == 2 and source.shape[1] == 161:
                    return candidate
    raise FileNotFoundError(f"no parsed 1v1 replay (.npy, 161 columns) in {folder}")


@dataclass(frozen=True)
class ReplaySources:
    pretrain: Path
    target: Path | None
    resets: Path | None

    @classmethod
    def resolve(
        cls, pretrain: Path | None, target: Path | None, resets: Path | None,
        frameskip: int, pretrain_only: bool,
    ) -> "ReplaySources":
        if not pretrain_only and target is None:
            raise ValueError("--target-replay-dir is required for low-level training")
        if pretrain is None and target is None:
            raise ValueError("--pretrain-replay-dir or --target-replay-dir is required")
        pretrain_dir = replay_folder(pretrain or target, frameskip)
        target_dir = replay_folder(target, frameskip) if target is not None else None
        reset_dir = replay_folder(resets, frameskip) if resets is not None else None
        return cls(pretrain_dir, target_dir, reset_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--resume-checkpoint", type=Path)
    preliminary, _ = peek.parse_known_args(argv)
    saved = load_resume_checkpoint(preliminary.resume_checkpoint) if preliminary.resume_checkpoint else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-replay-dir", type=Path,
                        help="broader 1v1 replay corpus, including lower-ranked play, for representation pretraining only")
    parser.add_argument("--target-replay-dir", "--replay-dir", type=Path,
                        help="pro-level 1v1 demonstrations for target segments, the sequence prior and low-level requests")
    parser.add_argument("--replay-reset-dir", type=Path,
                        help="optional independent replay-reset pool; reset-only states never become expert targets")
    parser.add_argument("--no-replay-reset-dir", dest="replay_reset_dir", action="store_const",
                        const=None, help="draw low-level resets exclusively from pro target states")
    parser.add_argument("--external-reset-fraction", type=float, default=0.25,
                        help="fraction of resets drawn from --replay-reset-dir if supplied")
    parser.add_argument("--pretrain-only", action=argparse.BooleanOptionalAction, default=False,
                        help="train and save the state-only representation without starting CARL")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--pretrain-updates", type=int, default=10_000)
    parser.add_argument("--pretrain-checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--pretrain-batch", type=int, default=64)
    parser.add_argument("--pretrain-frame-limit", type=int)
    parser.add_argument("--target-frame-limit", type=int)
    parser.add_argument("--heldout-size", type=int, default=4_096)
    parser.add_argument("--representation-hidden", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--prior-hidden", type=int, default=128)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--min-duration", type=int, default=4)
    parser.add_argument("--max-duration", type=int, default=16)
    parser.add_argument("--plan-horizon", type=int, default=3)
    parser.add_argument("--replan-after", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--n-sim", type=int, default=64)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="minimum total CARL actor-steps after pretraining; the last joint step may overshoot")
    parser.add_argument("--representation-lr", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.8)
    parser.add_argument("--mask-adjust-interval", type=int, default=500)
    parser.add_argument("--min-concentration", type=float, default=2.0)
    parser.add_argument("--min-anchor-gap", type=float, default=0.01)
    parser.add_argument("--prior-lr", type=float, default=1e-4)
    parser.add_argument("--skill-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--prior-rounds", type=int, default=3)
    parser.add_argument("--prior-updates", type=int, default=200)
    parser.add_argument("--prior-batch", type=int, default=32)
    parser.add_argument("--expert-sequences", type=int, default=64)
    parser.add_argument("--segment-steps", type=int, default=64)
    parser.add_argument("--calibration-windows", type=int, default=64)
    parser.add_argument("--prior-weight", type=float, default=0.3)
    parser.add_argument("--boundary-penalty", type=float, default=1.0)
    parser.add_argument("--skill-updates", type=int, default=16)
    parser.add_argument("--skill-batch", type=int, default=512)
    parser.add_argument("--online-repr-updates", type=int, default=8)
    parser.add_argument("--online-prior-updates", type=int, default=16)
    parser.add_argument("--online-value-updates", type=int, default=8)
    parser.add_argument("--online-refit-interval", type=int, default=10)
    parser.add_argument("--dynamics-pairs", action=argparse.BooleanOptionalAction,
                        default=True, help="augment with cross-dynamics kinematic and action replay pairs")
    parser.add_argument("--dynamics-pair-interval", type=int, default=10)
    parser.add_argument("--dynamics-pair-batch", type=int, default=4)
    parser.add_argument("--single-reenactments", type=int, default=2,
                        help="additional single-controlled-car pro reenactments per online round")
    parser.add_argument("--expert-source-weight", type=float, default=0.7,
                        help="expert share of online representation and prior updates")
    parser.add_argument("--curriculum-rounds", type=int, default=100)
    parser.add_argument("--plan-candidates", type=int, default=4)
    parser.add_argument("--opponent-samples", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--memory-capacity", type=int, default=256)
    parser.add_argument("--reset-state-limit", type=int, default=16_384)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/lbifo"))
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    if saved is not None:
        defaults = {
            name: Path(value) if name.endswith("_dir") and value is not None else value
            for name, value in saved["config"].items()
            if name in {action.dest for action in parser._actions} and name != "resume_checkpoint"
        }
        defaults["pretrain_only"] = False  # a saved representation can start the pro-only stage
        defaults.pop("run_name", None)
        if saved["config"].get("pretrain_only") and th.cuda.is_available():
            defaults["device"] = "cuda"
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.min_duration < 2 or args.max_duration < args.min_duration:
        parser.error("--min-duration and --max-duration require 2 <= min <= max")
    if args.latent_dim < 3 or args.prior_hidden % 4:
        parser.error("--latent-dim must be >= 3 and --prior-hidden divisible by four")
    if not 1 <= args.replan_after <= args.plan_horizon:
        parser.error("--replan-after must be within the latent plan horizon")
    for name in (
        "pretrain_updates", "pretrain_checkpoint_interval", "pretrain_batch",
        "mask_adjust_interval",
        "representation_hidden", "policy_hidden",
        "prior_hidden", "plan_horizon", "frameskip", "n_sim", "max_ticks",
        "rollout", "timesteps", "prior_rounds", "prior_updates", "prior_batch",
        "expert_sequences", "segment_steps", "calibration_windows", "skill_updates",
        "skill_batch", "online_repr_updates", "online_prior_updates",
        "online_value_updates", "online_refit_interval", "curriculum_rounds",
        "dynamics_pair_interval", "dynamics_pair_batch",
        "plan_candidates", "opponent_samples", "diffusion_steps", "memory_capacity",
        "reset_state_limit", "checkpoint_interval", "checkpoint_keep",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.segment_steps < args.max_duration or args.rollout < args.min_duration:
        parser.error("--segment-steps must fit --max-duration and --rollout must fit --min-duration")
    if args.dynamics_pairs and args.dynamics_pair_batch < 2:
        parser.error("cross-dynamics contrastive pairs need --dynamics-pair-batch >= 2")
    if args.pretrain_frame_limit is not None and args.pretrain_frame_limit < args.min_duration + 1:
        parser.error("--pretrain-frame-limit must fit one behavior window")
    if args.target_frame_limit is not None and args.target_frame_limit < args.min_duration + 1:
        parser.error("--target-frame-limit must fit one behavior window")
    if args.heldout_size < 0 or args.seed < 0:
        parser.error("--heldout-size and --seed cannot be negative")
    if args.single_reenactments < 0:
        parser.error("--single-reenactments cannot be negative")
    for name in ("representation_lr", "prior_lr", "skill_lr", "value_lr", "no_touch_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("prior_weight", "boundary_penalty"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    if (not 0 < args.mask_ratio < 1 or not math.isfinite(args.min_concentration)
        or args.min_concentration <= 0 or not math.isfinite(args.min_anchor_gap)
        or args.min_anchor_gap < 0):
        parser.error("mask ratio, concentration and anchor-gap diagnostic thresholds are invalid")
    for name in ("external_reset_fraction", "expert_source_weight"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0 < args.gamma <= 1:
        parser.error("--gamma must be in (0, 1]")
    if not args.pretrain_only and not th.cuda.is_available():
        parser.error("low-level CARL training requires a CUDA-capable GPU")
    if not args.pretrain_only and th.device(args.device).type != "cuda":
        parser.error("low-level CARL training must use --device cuda")
    args.pretrain_data_needed = (
        saved is None or saved["pretrain_step"] < args.pretrain_updates or args.pretrain_only
    )
    sources = ReplaySources.resolve(
        args.pretrain_replay_dir if args.pretrain_data_needed else args.target_replay_dir,
        args.target_replay_dir, args.replay_reset_dir,
        args.frameskip, args.pretrain_only,
    )
    if args.pretrain_data_needed:
        args.pretrain_replay_dir = sources.pretrain
    args.target_replay_dir = sources.target
    args.replay_reset_dir = sources.resets
    if saved is not None:
        for name in (
            "frameskip", "latent_dim", "representation_hidden", "prior_hidden",
            "policy_hidden", "min_duration", "max_duration", "plan_horizon",
        ):
            if getattr(args, name) != saved["config"][name]:
                parser.error(f"--{name.replace('_', '-')} must match the resumed checkpoint")
        if args.timesteps <= saved["step"] and not args.pretrain_only:
            parser.error("--timesteps must exceed the resumed actor-step count")
        if saved["step"]:
            for name in ("seed", "heldout_size", "target_frame_limit"):
                if getattr(args, name) != saved["config"][name]:
                    parser.error(f"--{name.replace('_', '-')} must match the online checkpoint")
            if args.target_replay_dir.resolve() != Path(saved["config"]["target_replay_dir"]).resolve():
                parser.error("--target-replay-dir must match the online checkpoint")
    return args


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"LBIfO checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("architecture") != LBIFO_ARCHITECTURE:
        raise ValueError(f"incompatible LBIfO checkpoint: {path}")
    if not isinstance(payload.get("config"), dict) or not all(
        name in payload for name in ("step", "pretrain_step", "representation", "ema_encoder", "prior")
    ):
        raise ValueError(f"incomplete LBIfO checkpoint: {path}")
    return payload


class LBIFOTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = th.device(args.device)
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.pretrain_corpus = (
            ExpertCorpus(
                args.pretrain_replay_dir, args.min_duration, args.max_duration,
                args.frameskip, args.heldout_size, args.seed, args.pretrain_frame_limit,
            ) if args.pretrain_data_needed else None
        )
        self.target_corpus: ExpertCorpus | None = None
        if not args.pretrain_only:
            self.target_corpus = ExpertCorpus(
                args.target_replay_dir, args.min_duration, args.max_duration,
                args.frameskip, args.heldout_size, args.seed + 1, args.target_frame_limit,
            )
        self._target_span_starts = (
            [span.start for span in self.target_corpus.train_spans]
            if self.target_corpus is not None else []
        )
        self.representation = SceneRepresentation(
            args.representation_hidden, args.latent_dim,
        ).to(self.device)
        self.ema = copy.deepcopy(self.representation.encoder).eval().requires_grad_(False)
        self.prior = SphericalPlanPrior(
            args.latent_dim, args.prior_hidden, args.min_duration,
            args.max_duration, args.plan_horizon,
        ).to(self.device)
        self.calibration = CalibratedSurprise(args.max_duration)
        self.segmenter = SemiMarkovSegmenter(
            self.ema, self.representation.decoder, self.prior, self.calibration,
            args.min_duration, args.max_duration, args.prior_weight, args.boundary_penalty,
        )
        self.repr_optimizer = th.optim.Adam(self.representation.parameters(), lr=args.representation_lr)
        self.mask_ratio = args.mask_ratio
        self.prior_optimizer = th.optim.Adam(self.prior.parameters(), lr=args.prior_lr)
        self.memory = TrajectoryMemory(args.memory_capacity, args.min_duration, args.seed)
        self.value_memory: deque[tuple[th.Tensor, th.Tensor, th.Tensor]] = deque(
            maxlen=args.memory_capacity * 8
        )
        self.policy: BehaviorPolicy | None = None
        self.value: PlanValue | None = None
        self.policy_optimizer = self.value_optimizer = None
        self.step = self.pretrain_step = self.round = 0
        self.target_sequences: list[InferredSequence] = []
        self._target_chunks: list[tuple[int, th.Tensor]] = []
        self.run_id = args.run_name or datetime.now().strftime("lbifo-%Y%m%d-%H%M%S-%f")
        self.checkpoint_dir = args.checkpoint_dir / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _sample_representation(self, corpus: ExpertCorpus, count: int) -> th.Tensor:
        return corpus.sample_windows(count, self.device)

    def train_representation(self, updates: int, *, online: bool = False) -> dict[str, float]:
        metrics = {}
        for _ in range(updates):
            use_policy = online and self.memory.trajectories and (
                self.rng.random() >= self.args.expert_source_weight
            )
            if use_policy:
                length = int(self.rng.integers(
                    self.args.min_duration + 1, self.args.max_duration + 2
                ))
                try:
                    windows = self.memory.sample_windows(
                        self.args.pretrain_batch, length, self.device
                    )
                except ValueError:
                    use_policy = False
            if not use_policy:
                corpus = self.target_corpus if online else self.pretrain_corpus
                windows = self._sample_representation(corpus, self.args.pretrain_batch)
            domain = 1 if use_policy else 0
            loss, metrics = self.representation.loss(
                windows, domain, self.args.frameskip, mask_ratio=self.mask_ratio,
            )
            self.repr_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.representation.parameters(), 1.0)
            self.repr_optimizer.step()
            ema_update(self.ema, self.representation.encoder, 0.995)
        return {"loss": float(loss.detach()), **metrics}

    def check_representation(self, corpus: ExpertCorpus) -> dict[str, float]:
        windows = corpus.sample_windows(min(16, self.args.pretrain_batch), self.device)
        report = self.representation.anchor_diagnostics(
            windows, 0, self.args.frameskip,
        )
        if (report["expert_concentration"] < self.args.min_concentration
            or report["anchor_latent_gap"] < self.args.min_anchor_gap):
            self.mask_ratio = min(0.95, self.mask_ratio + 0.05)
        report["mask_ratio"] = self.mask_ratio
        return report

    def train_dynamics_pairs(self, replay: DynamicsPairs) -> dict[str, float]:
        """The *same* full behavior under demo/kinematic or original/action replay."""
        corpus = self.target_corpus
        length = self.args.min_duration + 1
        summary = {}
        if len(corpus.safe_reset_indices):
            originals, transformed = [], []
            attempts = 0
            while len(originals) < self.args.dynamics_pair_batch and attempts < 6 * self.args.dynamics_pair_batch:
                attempts += 1
                sampled = self._draw_safe_start(length)
                if sampled is None:
                    break
                start, _ = sampled
                reset = corpus.reset_dataset(th.tensor([start]), self.device)[0]
                physical = {name: value[0] for name, value in reset.items()}
                window = corpus.frames[start:start + length]
                rendered = replay.kinematic_positive(window, physical)
                if rendered is not None:
                    originals.append(window)
                    transformed.append(rendered)
            if len(originals) >= 2:
                source = th.stack(originals).to(self.device)
                paired = th.stack(transformed)
                loss, _ = self.representation.loss(
                    source, 0, self.args.frameskip, positive=paired,
                    positive_domain=1, mask_ratio=self.mask_ratio,
                )
                self.repr_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.representation.parameters(), 1.0)
                self.repr_optimizer.step()
                ema_update(self.ema, self.representation.encoder, 0.995)
                summary["kinematic_pair_loss"] = float(loss.detach())

        original, paired = [], []
        records = [record for record in self.memory.trajectories
                   if record.reset_state is not None and len(record.actions) >= self.args.min_duration
                   and (record.controlled is None or bool(record.controlled.all()))]
        for selected in self.rng.permutation(len(records))[:self.args.dynamics_pair_batch * 4]:
            record = records[int(selected)]
            regenerated = replay.action_positive(record, self.args.min_duration)
            if regenerated is not None:
                original.append(record.scenes[:length])
                paired.append(regenerated)
                if len(original) == self.args.dynamics_pair_batch:
                    break
        if len(original) >= 2:
            loss, _ = self.representation.loss(
                th.stack(original).to(self.device), 1, self.args.frameskip,
                positive=th.stack(paired), positive_domain=2,
                mask_ratio=self.mask_ratio,
            )
            self.repr_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.representation.parameters(), 1.0)
            self.repr_optimizer.step()
            ema_update(self.ema, self.representation.encoder, 0.995)
            summary["action_pair_loss"] = float(loss.detach())
        return summary

    @th.no_grad()
    def collect_single_reenactments(self, replay: DynamicsPairs) -> int:
        added = 0
        for _ in range(self.args.single_reenactments):
            sampled = self._draw_safe_start(self.args.min_duration + 1)
            if sampled is None:
                break
            start, end = sampled
            length = min(self.args.max_duration + 1, end - start)
            scenes = self.target_corpus.frames[start:start + length]
            physical = self.target_corpus.reset_dataset(th.tensor([start]), self.device)[0]
            reset = {name: value[0] for name, value in physical.items()}
            request = self.ema(scenes[None].to(self.device))[0][0, -1]
            record = replay.single_entity_rollout(
                scenes, reset, self.policy, self.prior, request,
                controlled=int(self.rng.integers(N_CARS)),
            )
            if record is not None:
                self.memory.add(record)
                added += 1
        return added

    def _draw_safe_start(self, length: int) -> tuple[int, int] | None:
        corpus = self.target_corpus
        for _ in range(256):
            if not len(corpus.safe_reset_indices):
                break
            start = int(corpus.safe_reset_indices[self.rng.integers(len(corpus.safe_reset_indices))])
            span = corpus.train_spans[bisect.bisect_right(self._target_span_starts, start) - 1]
            if start + length <= span.end:
                return start, span.end
        return None

    def pretrain(self) -> None:
        remaining = max(0, self.args.pretrain_updates - self.pretrain_step)
        for _ in range(remaining):
            stats = self.train_representation(1)
            self.pretrain_step += 1
            if self.pretrain_step % self.args.mask_adjust_interval == 0:
                stats.update(self.check_representation(self.pretrain_corpus))
                print(f"Representation diagnostic: concentration="
                      f"{stats['expert_concentration']:.2f}, "
                      f"anchor gap={stats['anchor_latent_gap']:.4f}, "
                      f"mask ratio={stats['mask_ratio']:.2f}")
            if self.pretrain_step % self.args.pretrain_checkpoint_interval == 0:
                self.checkpoint()
            if self.pretrain_step % max(1, self.args.pretrain_updates // 10) == 0:
                print(f"Representation {self.pretrain_step:,}/{self.args.pretrain_updates:,}: "
                      f"loss={stats['loss']:.4f}, kappa={stats['concentration']:.2f}")

    @th.no_grad()
    def recalibrate(self, domain: int = 0) -> None:
        if domain == 0:
            corpus = self.target_corpus or self.pretrain_corpus

            def sample(count: int, length: int) -> th.Tensor:
                sufficient = max(length, self.args.min_duration + 1)
                return corpus.sample_windows(count, self.device, length=sufficient)[:, :length]
        else:
            def sample(count: int, length: int) -> th.Tensor:
                sufficient = max(length, self.args.min_duration + 1)
                return self.memory.sample_windows(count, sufficient, self.device)[:, :length]
        self.calibration.fit(
            self.ema, self.representation.decoder, sample, domain,
            self.args.frameskip, self.args.calibration_windows,
        )

    def expert_chunks(self) -> list[tuple[int, th.Tensor]]:
        if self.target_corpus is None:
            raise ValueError("expert segments require the pro 1v1 target dataset")
        spans = [span for span in self.target_corpus.train_spans
                 if span.length > self.args.min_duration]
        if not spans:
            raise ValueError("no contiguous pro-level target segments exist")
        length = self.args.segment_steps + 1
        weights = np.asarray([
            max(1, span.length - length + 1) for span in spans
        ], dtype=np.float64)
        weights /= weights.sum()
        indices = self.rng.choice(len(spans), self.args.expert_sequences, p=weights)
        chunks = []
        for choice in indices:
            span = spans[choice]
            window = min(length, span.length)
            start = int(self.rng.integers(span.start, span.end - window + 1))
            chunks.append((start, self.target_corpus.frames[start:start + window]))
        return chunks

    def fit_prior(
        self, expert: list[InferredSequence],
        agent: list[InferredSequence] | None = None, updates: int | None = None,
    ) -> dict[str, float]:
        datasets = {}
        hazard_data = {}
        for domain, sequences in ((0, expert), (1, agent or [])):
            if sequences:
                hazard_data[domain] = hazard_examples(sequences)
            try:
                datasets[domain] = expert_prior_examples(sequences, self.args.plan_horizon)
            except ValueError:
                continue
        if 0 not in datasets and 1 not in datasets:
            raise ValueError("no full-length latent sequences for fitting the sequence prior")
        metrics = {}
        for _ in range(self.args.prior_updates if updates is None else updates):
            domain = 1 if 1 in datasets and (0 not in datasets or (
                self.rng.random() > self.args.expert_source_weight
            )) else 0
            data = datasets[domain]
            selected = th.randint(len(data[0]), (self.args.prior_batch,))
            state, plan, durations, previous, starts = (
                value[selected].to(self.device) for value in data
            )
            loss, metrics = self.prior.training_loss(
                state, plan, durations, previous, domain, self.args.frameskip,
                segment_starts=starts,
            )
            hazard = hazard_data[domain]
            sampled = th.randint(len(hazard[0]), (self.args.prior_batch,), device="cpu")
            begin, now, z, earlier, age, ends = (
                values[sampled].to(self.device) for values in hazard
            )
            ended = self.prior.hazard_probability(
                begin, now, z, earlier, age, domain, self.args.frameskip,
            )
            hazard_loss = F.binary_cross_entropy(ended.clamp(1e-5, 1 - 1e-5), ends)
            loss = loss + 0.1 * hazard_loss
            self.prior_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.prior.parameters(), 1.0)
            self.prior_optimizer.step()
        return {"prior_loss": float(loss.detach()),
                "expert_hazard_bce": float(hazard_loss.detach()), **metrics}

    def infer_targets(self, chunks: list[tuple[int, th.Tensor]]) -> list[InferredSequence]:
        return [self.segmenter.infer(scene, 0, self.args.frameskip, use_prior=bool(self.target_sequences))
                for _, scene in chunks]

    def segment_experts(self) -> list[tuple[int, th.Tensor]]:
        self.recalibrate()
        chunks = self.expert_chunks()
        self._target_chunks = chunks
        previous = None
        for cycle in range(self.args.prior_rounds):
            sequences = self.infer_targets(chunks)
            boundaries = [seq.boundaries for seq in sequences]
            if previous is not None and boundaries == previous:
                break
            self.fit_prior(sequences)
            previous = boundaries
            self.target_sequences = sequences
            self.expert_concentration = float(th.cat([
                seq.concentrations.flatten() for seq in sequences
            ]).mean().clamp_min(0.05))
            print(f"Expert segmentation {cycle + 1}/{self.args.prior_rounds}: "
                  f"{sum(len(seq.latents) for seq in sequences)} pro-level behaviors")
        return chunks

    @th.no_grad()
    def resume_targets(self, payload: dict) -> list[tuple[int, th.Tensor]] | None:
        """Keep the checkpoint's pro-only segment boundaries across resumes."""
        if "target_chunks" not in payload or "target_boundaries" not in payload:
            return None
        chunks = [(int(start), frames.cpu()) for start, frames in payload["target_chunks"]]
        boundaries = payload["target_boundaries"]
        if len(chunks) != len(boundaries):
            raise ValueError("saved target sequences and boundaries are misaligned")
        sequences = []
        for (_, scenes), bounds in zip(chunks, boundaries):
            if bounds[0] != 0 or bounds[-1] != len(scenes) - 1:
                raise ValueError("checkpoint target boundaries do not fit the replay scenes")
            z, concentration = [], []
            for start, stop in zip(bounds[:-1], bounds[1:]):
                mean, kappa = self.ema(scenes[start:stop + 1][None].to(self.device))
                z.append(mean[0, -1].cpu())
                concentration.append(kappa[0, -1].cpu())
            sequences.append(InferredSequence(
                scenes, tuple(bounds), th.stack(z), th.stack(concentration),
                0, self.args.frameskip,
            ))
        self._target_chunks = chunks
        self.target_sequences = sequences
        self.expert_concentration = float(th.cat([
            seq.concentrations.flatten() for seq in sequences
        ]).mean().clamp_min(0.05))
        return chunks

    @th.no_grad()
    def target_reset_pool(self, chunks: list[tuple[int, th.Tensor]]):
        corpus = self.target_corpus
        safe = corpus.safe_reset_indices
        starts, latents = [], []
        for (offset, _), sequence in zip(chunks, self.target_sequences):
            for index, boundary in enumerate(sequence.boundaries[:-1]):
                start = offset + boundary
                if th.isin(th.tensor(start), safe):
                    starts.append(start)
                    latents.append(sequence.latents[index])
        if not starts:
            # A boundary near a goal or collision may not be safe as a reset.
            # Draw other safe starts and re-encode their *pro-only* continuations.
            safe_start = []
            for span in corpus.train_spans:
                allowed = safe[(safe >= span.start) & (safe + self.args.min_duration < span.end)]
                safe_start.extend(allowed.tolist())
            if not safe_start:
                raise ValueError("no physics-safe pro 1v1 segment starts for low-level training")
            selected = self.rng.choice(safe_start, min(256, len(safe_start)), replace=False)
            starts = [int(value) for value in selected]
            for base in range(0, len(starts), 64):
                windows = corpus.frames[
                    th.tensor(starts[base:base + 64])[:, None]
                    + th.arange(self.args.min_duration + 1)
                ].to(self.device)
                latents.extend(self.ema(windows)[0][:, -1].cpu())
        starts = th.tensor(starts, dtype=th.long)
        pool = corpus.reset_dataset(starts, self.device)
        return pool, th.stack(latents)

    def attach_policy(self, environment: CARLTorchVectorEnv) -> None:
        args = self.args
        self.policy = BehaviorPolicy(
            environment.single_observation_space.shape[-1],
            tuple(int(size) for size in environment.single_action_space.nvec),
            args.latent_dim, args.policy_hidden, environment.action_codec,
        ).to(self.device)
        self.value = PlanValue(args.latent_dim, args.prior_hidden, args.plan_horizon).to(self.device)
        self.policy_optimizer = th.optim.Adam(self.policy.parameters(), lr=args.skill_lr)
        self.value_optimizer = th.optim.Adam(self.value.parameters(), lr=args.value_lr)

    def checkpoint(self, *, final: bool = False) -> Path:
        name = f"lbifo_{self.step:012d}.pt" if self.policy is not None else "pretrain_latest.pt"
        payload = {
            "architecture": LBIFO_ARCHITECTURE,
            "config": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(self.args).items()
                if name not in ("resume_checkpoint", "pretrain_data_needed")
            },
            "step": self.step,
            "pretrain_step": self.pretrain_step,
            "round": self.round,
            "representation": self.representation.state_dict(),
            "ema_encoder": self.ema.state_dict(),
            "prior": self.prior.state_dict(),
            "representation_optimizer": self.repr_optimizer.state_dict(),
            "prior_optimizer": self.prior_optimizer.state_dict(),
            "mask_ratio": self.mask_ratio,
            "calibration_mean": self.calibration.mean,
            "calibration_std": self.calibration.std,
            "torch_rng_state": th.get_rng_state(),
            "numpy_rng_state": self.rng.bit_generator.state,
        }
        if self.policy is not None:
            payload.update(
                policy=self.policy.state_dict(), value=self.value.state_dict(),
                policy_optimizer=self.policy_optimizer.state_dict(),
                value_optimizer=self.value_optimizer.state_dict(),
            )
        if self._target_chunks:
            payload["target_chunks"] = self._target_chunks
            payload["target_boundaries"] = [
                seq.boundaries for seq in self.target_sequences
            ]
        if self.device.type == "cuda":
            payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.checkpoint_dir / name
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        if self.policy is not None:
            previous = sorted(self.checkpoint_dir.glob("lbifo_*.pt"))
            for old in previous[:-self.args.checkpoint_keep]:
                old.unlink()
        if final:
            print(f"LBIfO checkpoint: {path}")
        return path

    def restore(self, checkpoint: Path, *, restore_policy: bool) -> None:
        saved = load_resume_checkpoint(checkpoint)
        self.representation.load_state_dict(saved["representation"])
        self.ema.load_state_dict(saved["ema_encoder"])
        self.prior.load_state_dict(saved["prior"])
        self.repr_optimizer.load_state_dict(saved["representation_optimizer"])
        self.prior_optimizer.load_state_dict(saved["prior_optimizer"])
        self.mask_ratio = float(saved.get("mask_ratio", self.args.mask_ratio))
        self.calibration.mean = saved["calibration_mean"]
        self.calibration.std = saved["calibration_std"]
        if restore_policy and "policy" in saved:
            self.policy.load_state_dict(saved["policy"])
            self.value.load_state_dict(saved["value"])
            self.policy_optimizer.load_state_dict(saved["policy_optimizer"])
            self.value_optimizer.load_state_dict(saved["value_optimizer"])
        self.step, self.pretrain_step, self.round = (
            saved["step"], saved["pretrain_step"], saved["round"]
        )
        th.set_rng_state(saved["torch_rng_state"].cpu())
        self.rng.bit_generator.state = saved["numpy_rng_state"]
        if self.device.type == "cuda" and "cuda_rng_state" in saved:
            th.cuda.set_rng_state_all(saved["cuda_rng_state"])

    @th.no_grad()
    def achieved_requests(self, count: int, probability: float) -> th.Tensor | None:
        if not self.memory.trajectories or probability <= 0:
            return None
        valid = [record for record in self.memory.trajectories
                 if len(record.scenes) > self.args.min_duration]
        if not valid:
            return None
        chosen = []
        for _ in range(count):
            record = valid[int(self.rng.integers(len(valid)))]
            span = self.args.min_duration + 1
            start = int(self.rng.integers(len(record.scenes) - span + 1))
            chosen.append(record.scenes[start:start + span])
        inputs = th.stack(chosen).to(self.device)
        achieved = self.ema(inputs)[0][:, -1]
        mask = th.rand(count, device=self.device) < probability
        return th.where(mask[:, None, None], achieved, th.zeros_like(achieved))

    @th.no_grad()
    def collect(
        self, env: CARLTorchVectorEnv, provider: ExpertResetProvider,
        planner: JointPlanController, observation: th.Tensor, steps: int,
    ) -> tuple[th.Tensor, list[PlaySequence], dict[str, float]]:
        count = env.n_sim
        traces: dict[str, list[th.Tensor]] = {
            key: [] for key in (
                "scenes", "next_scenes", "observation", "action", "reward",
                "request", "age", "plan", "new_plan", "done",
                "completed_plan",
            )
        }
        issued = self._issued
        started_at_reset = self._was_reset
        snapshots: dict[tuple[int, int], dict[str, th.Tensor]] = {}
        for tick in range(steps):
            if started_at_reset.any():
                indices = started_at_reset.nonzero(as_tuple=True)[0]
                for index, values in zip(indices.tolist(), provider.snapshots(indices)):
                    snapshots[tick, index] = values
            before = observation[:, 0, :SCENE_SIZE].clone()
            requests = planner.current_latents().clone()
            age = planner.age.clone()
            plans = planner.plan.clone()
            starting_plan = issued | ~self._value_active
            self._value_starts[starting_plan] = before[starting_plan]
            self._value_plans[starting_plan] = plans[starting_plan]
            self._value_returns[starting_plan] = 0
            self._value_discounts[starting_plan] = 1
            self._value_active[starting_plan] = True
            action = planner.act(observation)
            next_flat, reward, terminated, truncated, info = env.step(action.flatten(0, 1))
            next_observation = next_flat.reshape(count, N_CARS, -1)
            done = (terminated | truncated).reshape(count, N_CARS).any(-1)
            next_scene = next_observation[:, 0, :SCENE_SIZE].clone()
            if done.any():
                if "final_obs" not in info:
                    raise RuntimeError("CARL must expose the final state before autoresetting")
                final = info["final_obs"].reshape(count, N_CARS, -1)
                next_scene[done] = final[done, 0, :SCENE_SIZE]
            for key, value in (
                ("scenes", before), ("next_scenes", next_scene),
                ("observation", observation), ("action", action),
                ("reward", reward.reshape(count, N_CARS)), ("request", requests),
                ("age", age), ("plan", plans), ("new_plan", issued), ("done", done),
            ):
                traces[key].append(value.detach().cpu())
            step_reward = reward.reshape(count, N_CARS)
            self._value_returns += self._value_discounts[:, None] * step_reward
            self._value_discounts *= self.args.gamma
            replan = planner.advance(before, next_scene, done)
            traces["completed_plan"].append((replan | done).detach().cpu())
            complete = (replan | done) & self._value_active
            for index in complete.nonzero(as_tuple=True)[0].tolist():
                self.value_memory.append(tuple(value[index].detach().cpu().clone()
                                               for value in (
                                                   self._value_starts, self._value_plans,
                                                   self._value_returns,
                                               )))
            self._value_active[complete] = False
            if replan.any():
                indices = replan.nonzero(as_tuple=True)[0]
                planner.reset(
                    indices, next_observation[indices, 0, :SCENE_SIZE],
                    use_value=self.round >= self.args.curriculum_rounds,
                )
            if done.any():
                indices, targets, _ = provider.take_pending()
                expected = done.nonzero(as_tuple=True)[0]
                if not th.equal(indices, expected):
                    raise RuntimeError("auto-reset indices and replay requests are out of sync")
                completion = min(1.0, self.round / self.args.curriculum_rounds)
                achieved = self.achieved_requests(len(indices), 0.5 * completion)
                planner.reset(
                    indices, next_observation[indices, 0, :SCENE_SIZE],
                    targets, prefer_expert=max(0.25, 1 - 0.75 * completion),
                    achieved=achieved,
                    use_value=self.round >= self.args.curriculum_rounds,
                )
            issued = replan | done
            started_at_reset = done
            observation = next_observation
        self._was_reset = started_at_reset
        self._issued = issued

        stacked = {key: th.stack(value) for key, value in traces.items()}
        records = []
        for simulation in range(count):
            starts = [0]
            starts.extend(int(value) + 1 for value in stacked["done"][:, simulation].nonzero().flatten())
            if starts[-1] != steps:
                starts.append(steps)
            for start, stop in zip(starts[:-1], starts[1:]):
                if stop - start < self.args.min_duration:
                    continue
                values = {name: stacked[name][start:stop, simulation]
                          for name in (
                              "observation", "action", "reward", "request", "age",
                              "plan", "new_plan", "completed_plan",
                          )}
                scenes = th.cat((stacked["scenes"][start:stop, simulation],
                                 stacked["next_scenes"][stop - 1:stop, simulation]))
                records.append(PlaySequence(
                    scenes, values["observation"], values["action"], values["reward"],
                    values["request"], values["age"], self.args.frameskip,
                    values["plan"], values["new_plan"],
                    snapshots.get((start, simulation)),
                    completed_plans=values["completed_plan"],
                ))
        stats = {
            "actor_steps": float(count * N_CARS * steps),
            "episodes": float(stacked["done"].sum()),
            "goal_reward_per_actor_step": float(stacked["reward"].mean()),
            "stored_trajectories": float(len(records)),
        }
        return observation, records, stats

    def train_skill(self) -> tuple[dict[str, float], list[InferredSequence]]:
        records = self.memory.sample_trajectories(min(8, len(self.memory.trajectories)))
        if not records:
            return {}, []
        try:
            labels, inferred = hindsight_labels(
                records, self.segmenter, self.expert_concentration,
            )
        except ValueError:
            return {}, []
        data = {name: getattr(labels, name).to(self.device)
                for name in labels.__dataclass_fields__}
        results = {}
        for _ in range(self.args.skill_updates):
            selected = th.randint(len(data["action"]), (self.args.skill_batch,), device=self.device)
            nll = self.policy.negative_log_likelihood(
                data["observation"][selected], data["action"][selected],
                data["latent"][selected], data["age"][selected],
            )
            loss = (data["weight"][selected] * nll).mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            examples = th.randint(len(data["ends"]), (self.args.skill_batch // N_CARS,), device=self.device)
            probability = self.prior.hazard_probability(
                data["start_scene"][examples], data["current_scene"][examples],
                data["joint_latent"][examples], data["previous"][examples],
                data["joint_age"][examples], 1, self.args.frameskip,
            )
            hazard = F.binary_cross_entropy(
                probability.clamp(1e-5, 1 - 1e-5), data["ends"][examples],
                reduction="none",
            )
            hazard = (
                hazard * data["hazard_mask"][examples]
            ).sum() / data["hazard_mask"][examples].sum().clamp_min(1)
            self.prior_optimizer.zero_grad(set_to_none=True)
            hazard.backward()
            th.nn.utils.clip_grad_norm_(self.prior.parameters(), 1.0)
            self.prior_optimizer.step()
            results = {"policy_nll": float(loss.detach()),
                       "hazard_bce": float(hazard.detach()),
                       "hindsight_segments": float(sum(len(seq.latents) for seq in inferred))}
        return results, inferred

    def train_value(self) -> dict[str, float]:
        if not self.value_memory:
            return {}
        states, plans, returns = (
            th.stack([example[index] for example in self.value_memory]).to(self.device)
            for index in range(3)
        )
        for _ in range(self.args.online_value_updates):
            selected = th.randint(len(states), (self.args.prior_batch,), device=self.device)
            prediction = self.value(states[selected], plans[selected])
            loss = (prediction - returns[selected]).square().mean()
            self.value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.value_optimizer.step()
        return {"value_mse": float(loss.detach()), "value_plans": float(len(states))}

    def train_online(self, chunks: list[tuple[int, th.Tensor]]) -> None:
        args = self.args
        target_states, requests = self.target_reset_pool(chunks)
        external = (
            load_demonstration_reset_dataset(
                args.replay_reset_dir, self.device, args.frameskip,
                args.reset_state_limit, args.seed, require_frame_skip_match=False,
            ) if args.replay_reset_dir is not None else None
        )
        provider = ExpertResetProvider(
            target_states, requests, args.n_sim, external,
            args.external_reset_fraction if external is not None else 0,
            seed=args.seed,
        )
        environment = CARLTorchVectorEnv(
            n_sim=args.n_sim, n_blue=1, n_orange=1, seed=args.seed,
            frameskip=args.frameskip, max_ticks=args.max_ticks,
            no_touch_timeout_seconds=args.no_touch_timeout,
            normalize=True, discrete_actions=True, reset_state_provider=provider,
        )
        dynamics = None
        try:
            if args.dynamics_pairs:
                dynamics = DynamicsPairs(args.frameskip, args.seed + 2)
            self.attach_policy(environment)
            if args.resume_checkpoint is not None:
                self.restore(args.resume_checkpoint, restore_policy=True)
            planner = JointPlanController(
                self.policy, self.prior, self.value, args.n_sim, args.frameskip,
                candidates=args.plan_candidates, opponent_samples=args.opponent_samples,
                replan_after=args.plan_horizon, diffusion_steps=args.diffusion_steps,
            )
            observation = environment.reset().reshape(args.n_sim, N_CARS, -1)
            self._was_reset = th.ones(args.n_sim, dtype=th.bool, device=self.device)
            self._issued = th.ones(args.n_sim, dtype=th.bool, device=self.device)
            self._value_starts = th.zeros(args.n_sim, SCENE_SIZE, device=self.device)
            self._value_plans = th.zeros(
                args.n_sim, args.plan_horizon, N_CARS, args.latent_dim, device=self.device
            )
            self._value_returns = th.zeros(args.n_sim, N_CARS, device=self.device)
            self._value_discounts = th.ones(args.n_sim, device=self.device)
            self._value_active = th.zeros(args.n_sim, dtype=th.bool, device=self.device)
            indices, targets, _ = provider.take_pending()
            planner.reset(indices, observation[indices, 0, :SCENE_SIZE],
                          targets, prefer_expert=1.0)
            print(f"Low-level training: {len(target_states)} pro target starts, "
                  f"{len(external) if external else 0} independent reset-only states")
            self.recalibrate(1) if self.memory.trajectories else None
            while self.step < args.timesteps:
                remaining = args.timesteps - self.step
                steps = min(args.rollout, (remaining + environment.n_envs - 1) // environment.n_envs)
                observation, records, metrics = self.collect(
                    environment, provider, planner, observation, steps,
                )
                self.step += int(metrics["actor_steps"])
                self.round += 1
                for record in records:
                    self.memory.add(record)
                if dynamics is not None and args.single_reenactments:
                    metrics["single_entity_trajectories"] = float(
                        self.collect_single_reenactments(dynamics)
                    )
                if self.memory.trajectories:
                    metrics.update(self.train_representation(args.online_repr_updates, online=True))
                    if dynamics is not None and self.round % args.dynamics_pair_interval == 0:
                        metrics.update(self.train_dynamics_pairs(dynamics))
                    self.recalibrate(1)
                    skill, inferred = self.train_skill()
                    metrics.update(skill)
                    metrics.update(self.train_value())
                    if self.round % args.online_refit_interval == 0:
                        self.recalibrate(0)
                        metrics.update(self.check_representation(self.target_corpus))
                        self.target_sequences = self.infer_targets(chunks)
                        self.expert_concentration = float(th.cat([
                            seq.concentrations.flatten() for seq in self.target_sequences
                        ]).mean().clamp_min(0.05))
                        metrics.update(self.fit_prior(
                            self.target_sequences, inferred,
                            updates=args.online_prior_updates,
                        ))
                print(f"LBIfO round {self.round}, actors {self.step:,}/{args.timesteps:,}: "
                      + ", ".join(f"{name}={value:.3f}" for name, value in metrics.items()
                                     if name in ("loss", "policy_nll", "hazard_bce", "value_mse", "prior_loss")))
                if self.round % args.checkpoint_interval == 0:
                    self.checkpoint()
            self.checkpoint(final=True)
        finally:
            if dynamics is not None:
                dynamics.close()
            environment.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    trainer = LBIFOTrainer(args)
    if args.resume_checkpoint is not None:
        trainer.restore(args.resume_checkpoint, restore_policy=False)
    trainer.pretrain()
    if args.pretrain_only:
        trainer.checkpoint(final=True)
        return
    resumed = (
        trainer.resume_targets(load_resume_checkpoint(args.resume_checkpoint))
        if args.resume_checkpoint is not None else None
    )
    chunks = resumed if resumed is not None else trainer.segment_experts()
    trainer.train_online(chunks)


if __name__ == "__main__":
    main()
