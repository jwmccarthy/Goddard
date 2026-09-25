"""Compare modern Seer defaults with the committed f7cab81 reward on events.

This is a development parity check: it loads the historical source from Git
without changing or installing the old trainer or simulator packages.
"""

import subprocess
import sys
import types
from pathlib import Path

import torch

from rewards import SeerReward
from test_rewards import make_context


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    revision = "f7cab8177941b1b049e51852c2fc79ab467772bf"
    source = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{revision}:rewards.py"], text=True
    )
    module = types.ModuleType("historical_f7cab81_rewards")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:rewards.py", "exec"), module.__dict__)

    scenarios = [
        {},
        {"ball_y": 700.0, "ball_speed": 2000.0, "touched_car": 0},
        {"ball_y": -600.0, "ball_z": 500.0, "car_z": 380.0, "touched_car": 1},
        {"score_delta": 1, "previous_ball_speed": 3000.0},
        {"score_delta": -1, "episode_ticks": 35900, "touched_car": 1},
        {"demoed_car": 0, "ball_x": 1000.0, "ball_y": 400.0},
        {"touched_car": 0, "truncated": True},
        {"episode_ticks": 200},
    ]
    comparisons = 0
    for normalize in (False, True):
        modern = SeerReward(1, 1, normalize=normalize, log_diagnostics=True)
        historical = module.SeerReward(1, 1, normalize=normalize, log_diagnostics=True)
        for parameters in scenarios:
            context = make_context(**parameters)
            current = modern(context)
            expected = historical(context)
            torch.testing.assert_close(current.reward, expected.reward, atol=1e-6, rtol=1e-6)
            for key, value in expected.info.items():
                torch.testing.assert_close(
                    torch.tensor(current.info[key]), torch.tensor(value), atol=1e-6, rtol=1e-6
                )
            comparisons += 1
    print(f"Matched {comparisons} event/reward comparisons with {revision}")


if __name__ == "__main__":
    main()
