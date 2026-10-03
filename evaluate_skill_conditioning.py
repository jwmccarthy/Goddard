"""Compare ASE skills from identical 1v1 starts against one fixed opponent."""

import argparse
from pathlib import Path

import torch as th

from carl.gymnasium import CARLTorchVectorEnv
from jarl.envs import DatasetResetSampler

from gaifo import GAIFO_ASE_ARCHITECTURE, POSITION_SCALE, load_resume_checkpoint
from gaifo_ase import sample_skills
from replay_resets import (
    ReplayResetProvider, load_demonstration_reset_frames, reset_index_dataset,
)
from watch_checkpoints import load_policy_checkpoint


def comparison_latents(
    count: int, size: int, device: th.device, generator: th.Generator,
) -> th.Tensor:
    """Different ego skills, one identical opponent skill in each match."""
    ego = sample_skills(count, size, device, generator)
    opponent = sample_skills(1, size, device, generator)
    latents = th.empty(2 * count, size, device=device)
    latents[::2] = ego
    latents[1::2] = opponent
    return latents


def evaluate_skill_conditioning(
    checkpoint: Path, replay_dir: Path, *, skills: int = 8,
    starts: int = 8, steps: int = 32, seed: int = 0,
    reset_state_limit: int = 256,
) -> dict[str, float]:
    """Change only the ego skill; hold start, opponent and opponent skill fixed."""
    if min(skills, starts, steps, reset_state_limit) < 1:
        raise ValueError("skills, starts, steps and reset_state_limit must be positive")
    if skills < 2:
        raise ValueError("at least two skills are needed for a comparison")
    if not th.cuda.is_available():
        raise RuntimeError("CARL skill-swap evaluation requires CUDA")
    payload = load_resume_checkpoint(checkpoint)
    config = payload["config"]
    if config["architecture"] != GAIFO_ASE_ARCHITECTURE:
        raise ValueError("skill-swap evaluation requires an ASE GAIFO checkpoint")

    device = th.device("cuda:0")
    frameskip = int(config["frameskip"])
    frames, internal = load_demonstration_reset_frames(
        replay_dir, device, frameskip, reset_state_limit, seed,
    )
    generator = th.Generator(device=device).manual_seed(seed)
    indices = th.randperm(len(frames), device=device, generator=generator)[:starts]
    latent_dim = int(config["ase_skill_dim"])
    latents = comparison_latents(skills, latent_dim, device, generator)
    first, second = th.triu_indices(skills, skills, offset=1, device=device)

    env = CARLTorchVectorEnv(
        n_sim=skills, n_blue=1, n_orange=1, seed=seed,
        frameskip=frameskip, normalize=True, discrete_actions=True,
        max_ticks=int(config.get("max_ticks", 1_000_000)),
    )
    try:
        viewer, _ = load_policy_checkpoint(checkpoint, env, frameskip, None)
        policy = viewer.policy
        scale = th.tensor(POSITION_SCALE, device=device)
        first_disagreements = []
        car_differences = []
        ball_differences = []
        goal_returns = []
        for index in indices.tolist():
            sampler = DatasetResetSampler(
                reset_index_dataset(th.tensor([index], device=device)),
                probability=1.0, seed=seed,
            )
            env.reset_state_provider = ReplayResetProvider(sampler, frames, internal)
            observation = env.reset()
            th.testing.assert_close(
                observation[::2], observation[:1].expand(skills, -1),
            )
            th.testing.assert_close(
                observation[1::2], observation[1:2].expand(skills, -1),
            )
            scene = observation[::2, :51]
            car_position = scene[:, 9:12].clone()
            ball_position = scene[:, :3].clone()
            alive = th.ones(skills, dtype=th.bool, device=device)
            score = th.zeros(skills, device=device)
            for t in range(steps):
                with th.inference_mode():
                    action = policy.act(
                        th.cat((observation, latents), dim=-1),
                        deterministic=True,
                    ).action
                if t == 0:
                    ego_actions = action[::2]
                    first_disagreements.append(
                        ego_actions[first].ne(ego_actions[second]).any(dim=-1)
                        .float().mean()
                    )
                observation, reward, terminated, truncated, _ = env.step(action)
                score += th.where(alive, reward[::2], 0)
                done = (terminated | truncated)[::2]
                alive &= ~done
                scene = observation[::2, :51]
                car_position = th.where(alive[:, None], scene[:, 9:12], car_position)
                ball_position = th.where(alive[:, None], scene[:, :3], ball_position)
                if not alive.any():
                    break
            car_differences.append(
                ((car_position[first] - car_position[second]) * scale)
                .norm(dim=-1).mean()
            )
            ball_differences.append(
                ((ball_position[first] - ball_position[second]) * scale)
                .norm(dim=-1).mean()
            )
            goal_returns.append(score.mean())
        return {
            "first_action_disagreement": float(th.stack(first_disagreements).mean().item()),
            "car_difference_uu": float(th.stack(car_differences).mean().item()),
            "ball_difference_uu": float(th.stack(ball_differences).mean().item()),
            "mean_goal_return": float(th.stack(goal_returns).mean().item()),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--skills", type=int, default=8)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-state-limit", type=int, default=256)
    args = parser.parse_args()
    metrics = evaluate_skill_conditioning(
        args.checkpoint, args.replay_dir,
        skills=args.skills, starts=args.starts, steps=args.steps,
        seed=args.seed, reset_state_limit=args.reset_state_limit,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
