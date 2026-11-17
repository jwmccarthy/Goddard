"""RocketSim GAIfO: Python setup/export, native C++ collection AND learning.

See training/GAIFO_CPU.md. Nothing in the training hot loop calls Python.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from replays.dataset import ExpertDataset
from replays.layout import team_scene_size
from training.arguments import (
    add_gaifo_arguments,
    add_ppo_arguments,
    resolve_critic_arguments,
    resolve_gaifo_defaults,
    resolve_joint_actions,
)
from training.discriminator import Discriminator
from training.models import Agent

ROOT = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/ZealanL/RLGymPPO_CPP.git"
REVISION = "ee4cc56fc8e43758cc898fdba92a9173a94e52f2"


class NativeActionCodec(nn.Module):
    """CARL ego-controller-v1 masks, independent of the installed CARL extension."""

    def mask(self, observation):
        grounded = observation[..., 25].bool()
        mask = torch.ones_like(observation[..., :18], dtype=torch.bool)
        mask[..., 10] = grounded
        mask[..., 12] = observation[..., 24] > 0
        mask[..., 17] = grounded | (observation[..., -29].bool() &
                                    (observation[..., -28] > 1 / 120))
        return mask


class PolicyExport(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.actor = agent.actor
        self.codec = agent.action_codec
        self.sizes = agent.action_sizes

    def forward(self, observation):
        logits = self.actor(observation).masked_fill(~self.codec.mask(observation), -1e9)
        return torch.stack([F.pad(x, (0, 3 - size), value=-1e9)
                            for x, size in zip(logits.split(self.sizes, -1), self.sizes)], -2)


class DataExport(nn.Module):
    def __init__(self, experts, resets=None):
        super().__init__()
        for name, value in {
            "frames": experts.frames, "ends": experts.train_ends,
            "lengths": experts.train_lengths,
            "reset_frames": resets.frames if resets else torch.empty(0),
            "reset_internal": resets.internal_states if resets else torch.empty(0),
        }.items():
            self.register_buffer(name, value)

    def forward(self):
        return self.frames


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_ppo_arguments(result)
    add_gaifo_arguments(result)
    result.set_defaults(n_sim=256, timesteps=1_000_000, discriminator_history_capacity=0,
                        observation_schema="ego-controller-v1", previous_controls=False,
                        bf16=False, compile=False, cuda_graphs=False,
                        compile_autotune=False, compile_ppo_update=False)
    native = result.add_argument_group("native RocketSim collector/LibTorch learner")
    native.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    native.add_argument("--torch-threads", type=int, default=1)
    native.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    native.add_argument("--build-dir", type=Path, default=ROOT / "runs/gaifo-cpu-build")
    native.add_argument("--upstream-dir", type=Path)
    native.add_argument("--collision-meshes", type=Path)
    native.add_argument("--build-only", action="store_true")
    native.add_argument("--build-jobs", type=int, default=4)
    native.add_argument("--warmup-iterations", type=int, default=2)
    native.add_argument("--profile-transfers", action="store_true",
                        help="synchronize around rollout copies to measure host-visible transfer "
                        "latency; use a separate profiling run because barriers perturb timing")
    native.add_argument("--compare-carl-log", type=Path,
                        help="optional gaifo.py TensorBoard directory for a descriptive SPS ratio")
    return result


def validate(args):
    from training.arguments import entropy_coefficients
    from training.schedules import validate_checkpoint_interval, validate_gamma_schedule

    validate_checkpoint_interval(args)
    validate_gamma_schedule(args)
    entropy_coefficients(args)
    if args.observation_schema != "ego-controller-v1" or args.previous_controls:
        raise ValueError("the native RocketSim backend currently requires ego-controller-v1")
    # A smaller explicit profile is preferable to silently benchmarking omitted work.
    unsupported = {
        "resume": args.resume, "bf16": args.bf16, "compile": args.compile,
        "cuda_graphs": args.cuda_graphs, "compile_autotune": args.compile_autotune,
        "compile_ppo_update": args.compile_ppo_update,
        "joint_actions": args.joint_actions,
        "mlp_critic": args.mlp_critic, "critic_relative": args.critic_relative,
        "experimental_jump_phase": args.experimental_jump_phase,
        "discriminator_hierarchical": args.discriminator_hierarchical,
        "expert_maneuver_index": args.expert_maneuver_index,
        "reward_plus": args.reward_plus, "gameplay_reward_weight": args.gameplay_reward_weight,
        "goal_reward_log_dir": args.goal_reward_log_dir,
    }
    if args.opponent_pool_size and args.historical_opponent_fraction:
        unsupported["historical_opponents"] = True
    active = [name for name, value in unsupported.items() if value]
    if active:
        raise ValueError("native benchmark does not implement: " + ", ".join(active))
    if args.goal_reward and args.goal_reward_weight is None:
        raise ValueError("use an explicit --goal-reward-weight for a reproducible comparison")
    args.goal_reward_weight = args.goal_reward_weight or 0.0
    if not 0 <= args.goal_reward_weight < float("inf"):
        raise ValueError("goal reward weight must be finite and nonnegative")
    for name in ("threads", "torch_threads", "build_jobs", "max_ticks"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmup_iterations < 0 or args.no_touch_timeout < 0:
        raise ValueError("warmup iterations and no-touch timeout must be nonnegative")
    for name in ("learning_rate", "learning_rate_end", "vf_coef", "clip_coef"):
        if not 0 <= getattr(args, name) < float("inf"):
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("adam_eps", "max_grad_norm", "target_kl"):
        value = getattr(args, name)
        if value is not None and not 0 < value < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    if not args.no_touch_timeout < float("inf"):
        raise ValueError("no-touch timeout must be finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu")
    if args.discriminator_history_mix_fraction and not args.discriminator_history_capacity:
        raise ValueError("history mixing requires a positive history capacity")


def build(args):
    if args.build_jobs < 1:
        raise ValueError("build jobs must be positive")
    directory = args.build_dir.resolve()
    upstream = (args.upstream_dir or directory / "upstream").resolve()
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(upstream)], check=True)
        subprocess.run(["git", "-C", str(upstream), "checkout", REVISION], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"expected RLGymPPO_CPP {REVISION}, found {revision}")
    # Build against the installed wheel's LibTorch. No separate CUDA SDK/nvcc required.
    cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake")
    if not Path(cmake).is_file():
        raise RuntimeError("install CMake: uv pip install cmake")
    subprocess.run([cmake, "-S", str(ROOT / "training/native_gaifo"), "-B", str(directory),
                    "-DCMAKE_BUILD_TYPE=Release", f"-DUPSTREAM={upstream}",
                    f"-DTORCH_ROOT={Path(torch.__file__).parent}",
                    f"-DTORCH_ABI={int(torch.compiled_with_cxx11_abi())}"], check=True)
    subprocess.run([cmake, "--build", str(directory), "--parallel", str(args.build_jobs)],
                   check=True)
    return directory / "gaifo_native"


def export(args, directory):
    from training.arguments import entropy_coefficients
    from training.replay_resets import ReplayResetDataset

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    # The controller schema is shared with current gaifo.py, not DefaultOBS/90 actions.
    size = team_scene_size(args.team_size) + 68 + 6 * args.team_size * 2 + 8 + 27
    agent = Agent(size, (3, 3, 3, 2, 2, 3, 2), NativeActionCodec(),
                  args.hidden_size, args.layers)
    discriminator = Discriminator(args, args.team_size * 2)
    torch.backends.mha.set_fastpath_enabled(False)
    observation = torch.zeros(3, size)
    observation[:, 25] = 1
    observation[:, -29:-27] = 1
    windows = torch.zeros(3, args.discriminator_window_length, team_scene_size(args.team_size))
    windows[0, :-1] = float("nan")
    for name, model, example in (
        ("actor", PolicyExport(agent), observation), ("critic", agent.critic, observation),
        ("discriminator", discriminator, windows),
    ):
        # Tracing captures explicit device arguments in discriminator feature packing.
        # Export on the actual learner device rather than moving a CPU trace afterward.
        model = model.to(args.device)
        example = example.to(args.device)
        traced = torch.jit.trace(model, example, check_trace=False)
        # Verify dynamic batches/padding, and that the native graph retains autograd.
        other = example.repeat(2, *([1] * (example.ndim - 1)))
        torch.testing.assert_close(traced(other), model(other))
        traced(other).sum().backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in traced.parameters()):
            raise RuntimeError(f"{name} export lost finite gradients")
        model.zero_grad(set_to_none=True)
        traced.save(str(directory / f"{name}.jit"))
    experts = ExpertDataset(args.replay_dir, args.team_size, "cpu",
                            frameskip=args.frameskip * args.discriminator_decision_stride,
                            source_frameskip=args.replay_frameskip,
                            window_length=args.discriminator_window_length,
                            frame_limit=args.expert_frame_limit, seed=args.seed)
    resets = ReplayResetDataset(args.replay_dir, args.team_size, "cpu",
                                max_states=args.replay_reset_size, seed=args.seed
                                ) if args.replay_reset_fraction else None
    torch.jit.script(DataExport(experts, resets)).save(str(directory / "data.jit"))
    config = dict(vars(args), observation_size=size, scene_size=team_scene_size(args.team_size),
                  entropy_coefficients=entropy_coefficients(args), upstream_revision=REVISION,
                  torch_version=torch.__version__, native_format=1,
                  overtime_timeout_ticks=36_000,
                  learner_gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  logical_cpus=os.cpu_count(),
                  expert_windows=len(experts.train_ends),
                  reset_states=len(resets.frames) if resets else 0)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in config.items()}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    args = parser().parse_args()
    if args.build_only:
        print(build(args))
        return
    if args.resume:
        raise ValueError("native benchmark does not implement resume")
    resolve_gaifo_defaults(args)
    resolve_joint_actions(args)
    resolve_critic_arguments(args)
    validate(args)
    if args.collision_meshes is None:
        args.collision_meshes = Path(distribution("rlgym-rocket-league").locate_file(
            "rlgym/rocket_league/sim/collision_meshes"))
    if not list((args.collision_meshes / "soccar").glob("*.cmf")):
        raise ValueError("--collision-meshes must contain soccar/*.cmf")
    binary = build(args)
    from training.logging import run_directory

    directory = run_directory(args, "gaifo-cpu").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "config.json").exists():
        raise ValueError(f"run already exists: {directory}; choose a fresh --run-name")
    export(args, directory)
    subprocess.run([str(binary), str(directory)], check=True)
    if args.compare_carl_log:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(args.compare_carl_log), size_guidance={"scalars": 0}).Reload()
        samples = events.Scalars("physics_ticks_per_second")[args.warmup_iterations:]
        if not samples:
            raise ValueError("no post-warmup CARL physics_ticks_per_second samples")
        native = json.loads((directory / "summary.json").read_text())
        if native["measured_seconds"] <= 0:
            raise ValueError("native run has no post-warmup iterations")
        # Equal rollout sizes: harmonic mean gives transitions / total elapsed time.
        carl = len(samples) / sum(1 / x.value for x in samples)
        print(f"Descriptive CARL/native agent-tick throughput ratio: "
              f"{carl / native['agent_physics_ticks_per_second']:.3f}x. "
              "Check matching workloads/hardware; this log comparison is not a controlled trial.")


if __name__ == "__main__":
    main()
