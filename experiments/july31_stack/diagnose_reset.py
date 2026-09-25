"""Audit replay resets against CARL observations and action masks.

Example: python -B diagnose_reset.py /Goddard/july31_reset_dataset
"""

import argparse
import sys
from pathlib import Path

from train import _configure_imports


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--num-simulations", type=int, default=256)
    parser.add_argument("--probability", type=float, default=0.7)
    args = parser.parse_args()

    _configure_imports("july31", HERE / "vendor/site_965cba1", HERE / "vendor/jarl")
    sys.path.insert(0, str(HERE / "vendor/goddard_a817186"))
    import numpy as np
    import torch
    from carl.gymnasium import CARLTorchVectorEnv
    from jarl.envs import DatasetResetSampler
    from ppo import SyntheticMatchResetProvider
    from replay_states import load_replay_dataset

    if not hasattr(np.linalg, "vector_norm"):
        np.linalg.vector_norm = np.linalg.norm
    torch.manual_seed(0)
    dataset = load_replay_dataset(args.corpus, "cuda:0")
    sampler = DatasetResetSampler(dataset, probability=args.probability, seed=0)

    class RecordingProvider:
        def __init__(self):
            self.provider = SyntheticMatchResetProvider(sampler)
            self.last_sample = None

        def __call__(self, mask):
            self.last_sample = self.provider(mask)
            return self.last_sample

    recorder = RecordingProvider()
    env = CARLTorchVectorEnv(
        n_sim=args.num_simulations,
        n_blue=1,
        n_orange=1,
        seed=0,
        frameskip=8,
        no_touch_timeout_seconds=30.0,
        normalize=True,
        reset_state_provider=recorder,
    )
    try:
        observation = env.reset()
        state = env._state_from_carl(env._env.get_state())
        sample = recorder.last_sample
        assert sample is not None
        n = args.num_simulations
        replay_indices = sample["simulation_indices"]
        kickoff = torch.ones(n, dtype=torch.bool, device=env.device)
        kickoff[replay_indices] = False
        torch.testing.assert_close(
            state.ball_position[replay_indices], sample["ball_position"]
        )
        torch.testing.assert_close(
            state.car_position[replay_indices], sample["car_position"]
        )
        torch.testing.assert_close(state.car_boost[replay_indices], sample["car_boost"])
        torch.testing.assert_close(
            state.ball_position[kickoff, :2],
            torch.zeros_like(state.ball_position[kickoff, :2]),
        )

        z = state.car_position[..., 2]
        on_ground = state.car_on_ground
        masked_pitch = ~env.action_mask(observation)[..., 4:6].any(-1).reshape(n, 2)
        ball_y = state.ball_position[:, 1] / 6000.0
        observed_ball_y = observation.reshape(n, 2, -1)[:, :, 1]
        torch.testing.assert_close(observed_ball_y[:, 0], ball_y, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(observed_ball_y[:, 1], -ball_y, atol=1e-6, rtol=1e-6)

        def show(label, current, mask):
            print(
                f"{label}: cars={mask.sum().item()} "
                f"on_ground={current.car_on_ground[mask].float().mean().item():.3f} "
                f"pitch_masked={masked_pitch[mask].float().mean().item():.3f}",
                flush=True,
            )

        print(
            f"dataset_states={len(dataset)} replay={len(replay_indices)}/{n} "
            f"observation={tuple(observation.shape)}"
        )
        print(f"sampled_state_roundtrip=exact, obs_blue_orange_y_inversion=exact")
        show("initial all", state, torch.ones_like(on_ground))
        replay_cars = (~kickoff)[:, None].expand(n, 2)
        show("initial replay car_z>50", state, (z > 50) & replay_cars)
        show("initial replay car_z>100", state, (z > 100) & replay_cars)
        show("initial replay car_z>300", state, (z > 300) & replay_cars)
        clearly_airborne = (z > 300) & (
            state.car_position[..., 0].abs() < 3000
        ) & (state.car_position[..., 1].abs() < 3500) & replay_cars
        show("initial high and away from walls", state, clearly_airborne)

        actions = torch.zeros(env.n_envs, 7, dtype=torch.int64, device=env.device)
        next_obs, _, terminated, truncated, _ = env.step(actions)
        next_state = env._state_from_carl(env._env.get_transition_state())
        next_masked_pitch = ~env.action_mask(next_obs)[..., 4:6].any(-1).reshape(n, 2)
        print(
            f"after_step done={int((terminated | truncated).sum().item() // 2)} "
            f"initially_high_and_away={int(clearly_airborne.sum().item())} "
            f"on_ground={next_state.car_on_ground[clearly_airborne].float().mean().item():.3f} "
            f"pitch_masked={next_masked_pitch[clearly_airborne].float().mean().item():.3f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
