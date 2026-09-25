"""Run the current BASIC kickoff pipeline against current or July 31 CARL/JARL.

The historical packages are loaded from an isolated directory; the project's
normal .venv and the CARL/JARL checkouts are never modified.
"""

import argparse
import hashlib
import math
import os
import sys
import types
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CARL_BINARY = "_carl.cpython-311-x86_64-linux-gnu.so"
CARL_SHA256 = "bdff1d5d2ec6995e8a46d8dd708ede58923c7efe94426f2e541e2d645e52acf5"


def _configure_imports(stack: str, historical_site: Path, historical_jarl: Path) -> None:
    if stack == "july31":
        if not (historical_site / "carl" / "__init__.py").is_file():
            raise FileNotFoundError(f"July 31 CARL is missing: {historical_site}")
        if not (historical_jarl / "jarl" / "__init__.py").is_file():
            raise FileNotFoundError(f"July 31 JARL is missing: {historical_jarl}")
        with (historical_site / "carl" / CARL_BINARY).open("rb") as binary:
            if hashlib.file_digest(binary, "sha256").hexdigest() != CARL_SHA256:
                raise RuntimeError("CARL binary is not the verified July 31 build (965cba1)")
        sys.path[:0] = [str(historical_site), str(historical_jarl), str(PROJECT)]
    else:
        sys.path.insert(0, str(PROJECT))

    import carl
    import jarl

    if stack == "july31":
        if not Path(carl.__file__).resolve().is_relative_to(historical_site):
            raise RuntimeError("July 31 CARL was shadowed by another installation")
        if not Path(jarl.__file__).resolve().is_relative_to(historical_jarl):
            raise RuntimeError("July 31 JARL was shadowed by another installation")

        # July 31 exported init_layer rather than the equivalent factory.
        import jarl.modules
        from jarl.modules.utils import init_layer

        jarl.modules.orthogonal_init = lambda std: partial(init_layer, std=std)

    # A zero-probability reset sampler never reads its dataset. Avoid loading
    # replays (and importing the newer CARL-only tracker) for kickoff-only runs.
    import torch
    from jarl.data import TensorBatch, TensorDataset

    def unused_kickoff_dataset(_directory, device, *_args, **_kwargs):
        return TensorDataset(TensorBatch({"unused": torch.zeros((1, 1), device=device)}))

    replay_resets = types.ModuleType("replay_resets")
    replay_resets.load_demonstration_reset_dataset = unused_kickoff_dataset
    sys.modules["replay_resets"] = replay_resets

    print(f"CARL: {Path(carl.__file__).resolve()}", flush=True)
    print(f"JARL: {Path(jarl.__file__).resolve()}", flush=True)


def _adapt_july31(basic) -> None:
    import torch
    from jarl.data import TensorBatch
    from jarl.learn import LossOutput
    from jarl.modules.operator import Critic
    from jarl.modules.policy import MultiCategoricalPolicy
    from jarl.store.rollout import Rollout
    from jarl.transform.base import PrepareContext, apply_transforms

    old_environment = basic.CARLTorchVectorEnv

    def july31_environment(*args, discrete_actions=True, **kwargs):
        if not discrete_actions:
            raise ValueError("this comparison requires discrete actions")
        # CARL already used discrete actions in July 31; the flag was added later.
        return old_environment(*args, **kwargs)

    class July31Policy(MultiCategoricalPolicy):
        def __init__(self, *, foot, body, head, action_codec):
            # July 31 called the encoder 'head' and the output network 'foot'.
            super().__init__(head=foot, body=body, foot=head, action_codec=action_codec)

    class July31Critic(Critic):
        def __init__(self, *, foot, body, head):
            super().__init__(head=foot, body=body, foot=head)

    class July31KLLimitedUpdate(basic.KLLimitedUpdate):
        """Use July 31's Update loop with BASIC's same KL cutoff and metrics."""

        def update(self, experience):
            if isinstance(experience, Rollout):
                batch, context = experience.steps, PrepareContext(experience)
            elif isinstance(experience, TensorBatch):
                batch, context = experience, PrepareContext()
            else:
                raise TypeError("Update requires a Rollout or TensorBatch")

            prepared = apply_transforms(batch, self.transforms, context)
            totals = {}
            self._minibatches = 0
            self._early_stopped = False
            self._stop_kl = 0.0
            callback = self._progress_callback
            if callback is not None:
                callback.start(self.sampler.epochs, self.section)
            try:
                for sample in self.sampler(prepared):
                    output = self.loss(sample)
                    if isinstance(output, torch.Tensor):
                        output = LossOutput(output, {"loss": output})
                    if not isinstance(output, LossOutput):
                        raise TypeError("loss must return a tensor or LossOutput")
                    kl = float(output.metrics["approx_kl"])
                    if not math.isfinite(kl):
                        raise RuntimeError("non-finite PPO approximate KL")
                    if kl > self.target_kl:
                        self._early_stopped = True
                        self._stop_kl = kl
                        break
                    self.optimizer_step(output.loss)
                    for name, value in output.metrics.items():
                        totals[name] = totals.get(name, 0.0) + (
                            value.detach() if isinstance(value, torch.Tensor) else value
                        )
                    self._minibatches += 1
            finally:
                if callback is not None:
                    callback.finish()

            if not self._minibatches:
                raise RuntimeError("sampler produced no minibatches")
            self.optimizer_step.advance_scheduler()
            after_update = getattr(self.loss, "after_update", None)
            if after_update is not None:
                after_update()
            metrics = {
                name: float(value / self._minibatches)
                for name, value in totals.items()
            }
            metrics.update(
                kl_early_stop=float(self._early_stopped),
                kl_stop_value=self._stop_kl,
                optimizer_minibatches=float(self._minibatches),
            )
            return {self.section: metrics}

    basic.CARLTorchVectorEnv = july31_environment
    basic.MultiCategoricalPolicy = July31Policy
    basic.Critic = July31Critic
    basic.KLLimitedUpdate = July31KLLimitedUpdate


def _add_touch_logging(basic) -> None:
    import torch

    class TouchEpisodeRunner(basic.DiagnosticSelfPlayRunner):
        def reset(self):
            observation = super().reset()
            self._episode_learner_touched = torch.zeros(
                self.env.n_envs, dtype=torch.bool, device=self.env.device
            )
            self._touched_learner_episodes = torch.zeros(
                (), device=self.env.device
            )
            return observation

        def _record_diagnostics(self, env_step, learner_mask):
            touched = self.transition_reward.last_touches.reshape(-1)
            done = torch.as_tensor(
                env_step.done, dtype=torch.bool, device=self.env.device
            )
            self._episode_learner_touched |= touched
            self._touched_learner_episodes += (
                self._episode_learner_touched & done & learner_mask
            ).sum()
            self._episode_learner_touched[done] = False
            super()._record_diagnostics(env_step, learner_mask)

        def diagnostic_metrics(self):
            episodes = (
                self._diagnostics["episodes"].item()
                if self._diagnostics is not None else 0
            )
            fraction = (
                self._touched_learner_episodes.item() / episodes
                if episodes else None
            )
            report = super().diagnostic_metrics()
            if fraction is not None:
                report.setdefault("Gameplay", {})["touch_episode_fraction"] = fraction
                self._touched_learner_episodes.zero_()
            return report

    class TouchLogger(basic.Logger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_progress_metric(
                "Gameplay", "touch_episode_fraction", "touch episodes", ".3f"
            )

        def update(self, info, step=None):
            super().update(info, step=step)
            if "Gameplay" in info:
                gameplay = info["Gameplay"]

                def metric(name):
                    value = gameplay.get(name)
                    return "-" if value is None else f"{value:.4f}"

                print(
                    f"UPDATE learner_steps={self.step:,} "
                    f"touches/1k={metric('touches_per_1000_steps')} "
                    f"touch_episodes={metric('touch_episode_fraction')} "
                    f"timeout_fraction={metric('timeout_fraction')} "
                    f"goals_for/1k={metric('goals_for_per_1000_steps')}",
                    flush=True,
                )

    basic.DiagnosticSelfPlayRunner = TouchEpisodeRunner
    basic.Logger = TouchLogger


def _set_reward_variant(basic, variant: str) -> None:
    from rewards import SeerRewardWeights

    original = basic.DiagnosticSeerReward

    class VariantReward(original):
        def __init__(self, *args, **kwargs):
            weights = kwargs.pop("weights", SeerRewardWeights())
            weights = replace(weights, kickoff=0.0, kickoff_touch=0.0)
            if variant == "kickoff-touch":
                weights = replace(weights, kickoff_touch=1.0)
            super().__init__(*args, weights=weights, **kwargs)

    basic.DiagnosticSeerReward = VariantReward


def _verify_pipeline(basic, expected_kickoff_touch: float = 0.0) -> None:
    import torch

    env = basic.CARLTorchVectorEnv(
        n_sim=2,
        n_blue=1,
        n_orange=1,
        frameskip=8,
        no_touch_timeout_seconds=0.4,
        normalize=True,
        discrete_actions=True,
    )
    try:
        reward = env.register_reward(
            basic.DiagnosticSeerReward(1, 1, log_diagnostics=True)
        )
        assert reward.weights.kickoff_touch == expected_kickoff_touch
        observation = env.reset()
        actor, critic = basic.build_policy_and_critic(
            env, SimpleNamespace(hidden_size=256)
        )
        with torch.no_grad():
            action = actor.act(
                observation, actor.initial_state(env.n_envs)
            ).action
            values = critic.value(observation, critic.initial_state(env.n_envs))
            # 0.4 seconds at frameskip 8 must expire after six untouched steps.
            for step in range(6):
                _, rew, terminated, truncated, _ = env.step(torch.zeros_like(action))
                if step < 5:
                    assert not terminated.any() and not truncated.any()
        assert observation.shape == (4, 137)
        assert action.shape == (4, 7)
        assert truncated.sum().item() == 4
        assert torch.isfinite(values).all()
        assert torch.isfinite(rew).all() and reward._count > 0
        print("Verified 137-feature recurrent actor/critic, reward, and no-touch reset.")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stack", required=True, choices=("current", "july31"))
    parser.add_argument(
        "--reward-variant", choices=("historical", "kickoff-touch"),
        default="historical", help="kickoff contact bonus ablation",
    )
    parser.add_argument("--historical-site", type=Path, default=HERE / "vendor/site_965cba1")
    parser.add_argument("--historical-jarl", type=Path, default=HERE / "vendor/jarl")
    parser.add_argument("--verify-only", action="store_true")
    options, training_args = parser.parse_known_args()
    if training_args and training_args[0] == "--":
        training_args.pop(0)

    _configure_imports(
        options.stack,
        options.historical_site.resolve(),
        options.historical_jarl.resolve(),
    )
    import basic

    if options.stack == "july31":
        _adapt_july31(basic)
    _add_touch_logging(basic)
    _set_reward_variant(basic, options.reward_variant)
    print(f"Reward variant: {options.reward_variant}", flush=True)
    if options.verify_only:
        _verify_pipeline(basic, 1.0 if options.reward_variant == "kickoff-touch" else 0.0)
        return

    sys.argv = [str(PROJECT / "basic.py"), "--replay-reset-probability", "0", *training_args]
    arguments = basic.parse_arguments()
    if arguments.replay_reset_probability != 0:
        parser.error("kickoff comparison requires --replay-reset-probability 0")
    if options.stack == "july31" and arguments.resume_checkpoint is not None:
        parser.error("July 31 checkpoint keys need conversion before resuming")
    if arguments.run_name is None:
        sys.argv.extend((
            "--run-name",
            f"{options.stack}-kickoff-{datetime.now():%Y%m%d-%H%M%S}",
        ))
    os.chdir(PROJECT)
    basic.main()


if __name__ == "__main__":
    main()
