"""Run pinned July 31 PPO with an exact committed Goddard reward revision.

Only rewards.py is imported from the requested Git revision. The trainer,
replay sampler, architecture, and CLI remain the pinned a817186 versions.
"""

import argparse
import subprocess
import sys
import types

import ppo


REWARD_COMMITS = {
    "de8975f": "de8975ff55277f13c0de39491aa08f1f69a214c9",
    "f7cab81": "f7cab8177941b1b049e51852c2fc79ab467772bf",
    "e53f73e": "e53f73e37e62709571d55b098809436f8f3678f3",
    "a817186": "a817186d3c1441706d287faea4eea42d7e06b720",
}


def load_reward(commit: str):
    revision = REWARD_COMMITS[commit]
    source = subprocess.check_output(
        ["git", "-C", "/Goddard", "show", f"{revision}:rewards.py"],
        text=True,
    )
    module = types.ModuleType(f"goddard_reward_{commit}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:rewards.py", "exec"), module.__dict__)
    return module.SeerReward


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reward-commit", required=True, choices=REWARD_COMMITS)
    options, ppo_args = parser.parse_known_args()
    ppo.SeerReward = load_reward(options.reward_commit)
    print(f"Historical reward commit: {REWARD_COMMITS[options.reward_commit]}", flush=True)
    sys.argv = [sys.argv[0], *ppo_args]
    ppo.main()


if __name__ == "__main__":
    main()
