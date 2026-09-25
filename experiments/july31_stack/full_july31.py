"""Run Goddard a817186 with July 31 CARL/JARL from the separate vendor bundle."""

import argparse
import math
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from train import _configure_imports


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REWARD_VARIANTS = {
    "historical": {},
    "kickoff-velocity": {"kickoff": 0.1},
    "ball-progress-15": {"player_ball_progress": 15.0},
}


def _instrument(ppo, *, timeout_steps: int, reward_overrides: dict[str, float]) -> None:
    """Log contact rates and optionally change one reward weight for an ablation."""
    import torch
    from rewards import SeerRewardWeights

    original_reward = ppo.SeerReward
    original_runner = ppo.SelfPlayRunner
    original_logger = ppo.Logger
    active_runner = None

    class ObservedReward(original_reward):
        def __init__(self, *args, **kwargs):
            if reward_overrides:
                weights = kwargs.pop("weights", SeerRewardWeights())
                kwargs["weights"] = replace(weights, **reward_overrides)
            super().__init__(*args, **kwargs)

        def __call__(self, context):
            self.last_touches = context.current.car_ball_touches.detach().clone()
            return super().__call__(context)

    class ObservedRunner(original_runner):
        def __init__(self, *args, **kwargs):
            nonlocal active_runner
            super().__init__(*args, **kwargs)
            active_runner = self

        def reset(self):
            observation = super().reset()
            self._touched_episode = torch.zeros(
                self.n_envs, dtype=torch.bool, device=self.env.device
            )
            self._steps_since_touch = torch.zeros(
                self.env.n_sim, dtype=torch.long, device=self.env.device
            )
            self._diagnostics = {
                name: torch.zeros((), dtype=torch.long, device=self.env.device)
                for name in (
                    "learner_steps", "touches", "episodes", "touched_episodes", "timeouts"
                )
            }
            return observation

        def step(self):
            learner = self.matchmaker.learner_mask.clone()
            env_step = super().step()
            touches = self.env.reward_funcs[0].last_touches
            touch = touches.reshape(-1)
            done = torch.as_tensor(
                env_step.done, dtype=torch.bool, device=self.env.device
            )
            truncated = torch.as_tensor(
                env_step.truncated, dtype=torch.bool, device=self.env.device
            )
            self._diagnostics["learner_steps"] += learner.sum()
            self._diagnostics["touches"] += (touch & learner).sum()
            self._touched_episode |= touch
            self._diagnostics["episodes"] += (done & learner).sum()
            self._diagnostics["touched_episodes"] += (
                self._touched_episode & done & learner
            ).sum()

            self._steps_since_touch += 1
            self._steps_since_touch[touches.any(dim=-1)] = 0
            no_touch = truncated.reshape(-1, self.env.n_cars).all(dim=-1)
            no_touch &= self._steps_since_touch >= timeout_steps
            self._diagnostics["timeouts"] += (
                no_touch.repeat_interleave(self.env.n_cars) & learner
            ).sum()
            self._steps_since_touch[done.reshape(-1, self.env.n_cars).any(-1)] = 0
            self._touched_episode[done] = False
            return env_step

        def diagnostic_metrics(self):
            gameplay = {}
            learner_steps = self._diagnostics["learner_steps"].item()
            if learner_steps:
                gameplay["touches_per_1000_steps"] = (
                    self._diagnostics["touches"].item() / learner_steps * 1000
                )
                self._diagnostics["learner_steps"].zero_()
                self._diagnostics["touches"].zero_()
            episodes = self._diagnostics["episodes"].item()
            if episodes:
                gameplay["touch_episode_fraction"] = (
                    self._diagnostics["touched_episodes"].item() / episodes
                )
                gameplay["timeout_fraction"] = (
                    self._diagnostics["timeouts"].item() / episodes
                )
                for name in ("episodes", "touched_episodes", "timeouts"):
                    self._diagnostics[name].zero_()
            return {"Gameplay": gameplay} if gameplay else {}

    class ObservedLogger(original_logger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for key, label in (
                ("touches_per_1000_steps", "touches/1k"),
                ("touch_episode_fraction", "touch episodes"),
                ("timeout_fraction", "timeout frac"),
            ):
                self.register_progress_metric("Gameplay", key, label, ".3f")

        def update(self, info, step=None):
            report = {}
            if "PPO" in info and active_runner is not None:
                report = active_runner.diagnostic_metrics()
                if report:
                    super().update(report, step=step)
            super().update(info, step=step)
            if "PPO" in info:
                gameplay = report.get("Gameplay", {})

                def metric(name):
                    value = gameplay.get(name)
                    return "-" if value is None else f"{value:.4f}"

                print(
                    f"UPDATE learner_steps={self.step:,} "
                    f"touches/1k={metric('touches_per_1000_steps')} "
                    f"touch_episodes={metric('touch_episode_fraction')} "
                    f"timeout_fraction={metric('timeout_fraction')}",
                    flush=True,
                )

    ppo.SeerReward = ObservedReward
    ppo.SelfPlayRunner = ObservedRunner
    ppo.Logger = ObservedLogger


def _verify(ppo, *, reward_overrides: dict[str, float]) -> None:
    import torch
    from rewards import SeerRewardWeights

    env = ppo.CARLTorchVectorEnv(
        n_sim=2, n_blue=1, n_orange=1, frameskip=8,
        no_touch_timeout_seconds=0.4, normalize=True,
    )
    try:
        reward = env.register_reward(ppo.SeerReward(1, 1, log_diagnostics=True))
        assert reward.weights == replace(SeerRewardWeights(), **reward_overrides)
        observation = env.reset()
        actor, critic = ppo.build_policy_and_critic(
            env, SimpleNamespace(hidden_size=256)
        )
        with torch.no_grad():
            action = actor.act(observation, actor.initial_state(env.n_envs)).action
            values = critic.value(observation, critic.initial_state(env.n_envs))
            for step in range(6):
                _, rew, terminated, truncated, _ = env.step(torch.zeros_like(action))
                if step < 5:
                    assert not terminated.any() and not truncated.any()
        assert observation.shape == (4, 137) and action.shape == (4, 7)
        assert truncated.sum().item() == 4 and reward._count > 0
        assert torch.isfinite(values).all() and torch.isfinite(rew).all()
        print("Verified pinned Goddard actor/critic, reward, and no-touch reset.")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--historical-site", type=Path, default=HERE / "vendor/site_965cba1")
    parser.add_argument("--historical-jarl", type=Path, default=HERE / "vendor/jarl")
    parser.add_argument("--historical-goddard", type=Path, default=HERE / "vendor/goddard_a817186")
    parser.add_argument(
        "--replay-corpus", type=Path, default=None,
        help="load this July 31 reset_dataset instead of kickoff-only dummy",
    )
    parser.add_argument(
        "--initial-policy", type=Path, default=None,
        help="initialize the actor from an original July checkpoint; critic starts fresh",
    )
    parser.add_argument(
        "--reward-variant", choices=tuple(REWARD_VARIANTS),
        default="historical", help="one-weight kickoff reward ablation",
    )
    parser.add_argument("--verify-only", action="store_true")
    options, training_args = parser.parse_known_args()
    if training_args and training_args[0] == "--":
        training_args.pop(0)

    goddard = options.historical_goddard.resolve()
    if not (goddard / "ppo.py").is_file():
        raise FileNotFoundError(f"July 31 Goddard is missing: {goddard}")
    _configure_imports(
        "july31", options.historical_site.resolve(), options.historical_jarl.resolve()
    )
    sys.path.insert(0, str(goddard))
    import ppo

    if not Path(ppo.__file__).resolve().is_relative_to(goddard):
        raise RuntimeError("July 31 Goddard was shadowed by another checkout")
    print(f"Goddard: {Path(ppo.__file__).resolve()}", flush=True)
    reward_overrides = REWARD_VARIANTS[options.reward_variant]
    print(f"Reward variant: {options.reward_variant}", flush=True)
    if options.initial_policy is not None:
        import torch

        checkpoint = options.initial_policy.resolve(strict=True)
        original_build = ppo.build_policy_and_critic

        def build_with_initial_policy(environment, arguments):
            actor, critic = original_build(environment, arguments)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            actor.load_state_dict(
                saved["modules"]["policy"] if "modules" in saved else saved,
                strict=True,
            )
            return actor, critic

        ppo.build_policy_and_critic = build_with_initial_policy
        print(f"Initial policy: {checkpoint} (critic initialized fresh)", flush=True)
    if options.verify_only:
        _instrument(ppo, timeout_steps=6, reward_overrides=reward_overrides)
        _verify(ppo, reward_overrides=reward_overrides)
        return

    # Kickoff-only runs must never read the modern parsed-replay corpus with the
    # 2026-07-31 loader, so they get an unused one-row dataset. A --replay-corpus
    # run restores the native July 31 loader for the archived corpus instead.
    if options.replay_corpus is None:
        import torch
        from jarl.data import TensorBatch, TensorDataset

        def unused_kickoff_dataset(_dataset_root, device):
            return TensorDataset(TensorBatch({"unused": torch.zeros((1, 1), device=device)}))

        ppo.load_replay_dataset = unused_kickoff_dataset
        reset_arguments = ["--replay-reset-probability", "0", "--replay-dataset", str(PROJECT)]
    else:
        corpus = options.replay_corpus.resolve()
        if not (corpus / "CURRENT").is_file():
            raise FileNotFoundError(f"July 31 reset corpus has no CURRENT: {corpus}")
        import numpy as np

        if not hasattr(np.linalg, "vector_norm"):
            np.linalg.vector_norm = np.linalg.norm
        reset_arguments = [
            "--replay-dataset", str(corpus), "--replay-reset-probability", "0.7",
        ]
    sys.argv = [str(goddard / "ppo.py"), *reset_arguments, *training_args]
    arguments = ppo.parse_arguments()
    if options.replay_corpus is None:
        if arguments.replay_reset_probability != 0:
            parser.error("kickoff comparison requires --replay-reset-probability 0")
    else:
        if not 0.0 < arguments.replay_reset_probability <= 1.0:
            parser.error("replay runs need a positive --replay-reset-probability")
    if options.initial_policy is not None and arguments.resume_checkpoint is not None:
        parser.error("--initial-policy cannot be combined with --resume-checkpoint")
    if arguments.run_name is None:
        sys.argv.extend((
            "--run-name", f"full-july31-kickoff-{datetime.now():%Y%m%d-%H%M%S}",
        ))
    _instrument(
        ppo,
        timeout_steps=math.ceil(
            arguments.no_touch_timeout * 120.0 / arguments.frameskip
        ),
        reward_overrides=reward_overrides,
    )
    os.chdir(PROJECT)
    ppo.main()


if __name__ == "__main__":
    main()
