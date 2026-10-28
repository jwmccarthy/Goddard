import argparse
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


def mlp(input_size: int, output_size: int, hidden_size: int, layers: int, gain: float):
    modules = []

    for _ in range(layers):
        linear = nn.Linear(input_size, hidden_size)
        nn.init.orthogonal_(linear.weight, math.sqrt(2))
        nn.init.zeros_(linear.bias)
        modules.extend((linear, nn.ReLU()))
        input_size = hidden_size

    linear = nn.Linear(input_size, output_size)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)

    return nn.Sequential(*modules, linear)


class Agent(nn.Module):
    def __init__(self, observation_size, action_sizes, action_codec, hidden_size=256, layers=2):
        super().__init__()
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.action_codec = action_codec
        self.actor = mlp(observation_size, sum(self.action_sizes), hidden_size, layers, 0.01)
        self.critic = mlp(observation_size, 1, hidden_size, layers, 1.0)

    def get_value(self, observations):
        return self.critic(observations).squeeze(-1).float()

    def get_action_and_value(self, observations, actions=None):
        logits = self.actor(observations).float()
        mask = self.action_codec.mask(observations)
        logits = logits.masked_fill(~mask, -1e9)
        width = max(self.action_sizes)
        logits = torch.stack(
            [
                F.pad(head, (0, width - size), value=-1e9)
                for head, size in zip(logits.split(self.action_sizes, dim=-1), self.action_sizes)
            ],
            dim=-2,
        )
        log_probabilities = logits.log_softmax(-1)

        if actions is None:
            noise = torch.empty_like(logits).exponential_().log()
            actions = (logits - noise).argmax(-1)

        log_prob = log_probabilities.gather(-1, actions.long().unsqueeze(-1))
        entropy = -(log_probabilities.exp() * log_probabilities).sum((-1, -2))

        return actions, log_prob.squeeze(-1).sum(-1), entropy, self.get_value(observations)


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    timeout_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @classmethod
    def allocate(cls, steps, agents, observation_size, action_size, device):
        shape = (steps, agents)

        return cls(
            observations=torch.empty((*shape, observation_size), device=device),
            actions=torch.empty((*shape, action_size), dtype=torch.long, device=device),
            log_probs=torch.empty(shape, device=device),
            rewards=torch.empty(shape, device=device),
            dones=torch.empty(shape, dtype=torch.bool, device=device),
            values=torch.empty(shape, device=device),
            timeout_values=torch.empty(shape, device=device),
            advantages=torch.empty(shape, device=device),
            returns=torch.empty(shape, device=device),
        )


def build_env(args):
    from carl.gymnasium import CARLTorchVectorEnv

    return CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=args.team_size,
        n_orange=args.team_size,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        discrete_actions=True,
    )


def plain_tensor(tensor):
    return tensor if type(tensor) is torch.Tensor else tensor.as_subclass(torch.Tensor)


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def collect(env, agent, policy, rollout, observation, args, scenes=None):
    episode_returns = []
    rollout.timeout_values.zero_()

    for step in range(args.rollout):
        torch.compiler.cudagraph_mark_step_begin()
        rollout.observations[step].copy_(observation)

        with autocast(env.device, args.bf16):
            actions, log_prob, _, values = policy(observation)

        rollout.actions[step].copy_(actions)
        rollout.log_probs[step].copy_(log_prob)
        rollout.values[step].copy_(values)

        next_observation, reward, terminated, truncated, info = env.step(rollout.actions[step])
        next_observation = plain_tensor(next_observation)
        done = terminated | truncated
        final_observation = plain_tensor(info.get("final_obs", next_observation))

        if "final_obs" in info:
            with autocast(env.device, args.bf16):
                final_values = agent.get_value(final_observation)

            rollout.timeout_values[step].copy_(final_values * truncated)

        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(done)

        if scenes is not None:
            scenes.record(
                step, rollout.observations[step], next_observation, final_observation, done
            )

        episode_returns.extend(info.get("reward", ()))
        observation = next_observation

    return observation, episode_returns


@torch.no_grad()
def compute_gae(rollout, next_value, gamma, gae_lambda):
    advantage = torch.zeros_like(next_value)

    for step in reversed(range(rollout.rewards.shape[0])):
        continuing = (~rollout.dones[step]).float()
        reward = rollout.rewards[step] + gamma * rollout.timeout_values[step]
        delta = reward + gamma * continuing * next_value - rollout.values[step]
        advantage = delta + gamma * gae_lambda * continuing * advantage
        rollout.advantages[step].copy_(advantage)
        next_value = rollout.values[step]

    rollout.returns.copy_(rollout.advantages + rollout.values)


def ppo_loss(
    agent,
    observations,
    actions,
    old_log_probs,
    advantages,
    returns,
    old_values,
    clip_coef,
    ent_coef,
    vf_coef,
    clip_value_loss,
):
    _, log_probs, entropy, values = agent.get_action_and_value(observations, actions)
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef),
    ).mean()
    value_error = (values - returns).square()

    if clip_value_loss:
        clipped_values = old_values + (values - old_values).clamp(-clip_coef, clip_coef)
        value_error = torch.maximum(value_error, (clipped_values - returns).square())

    value_loss = 0.5 * value_error.mean()
    entropy = entropy.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean()

    metrics = torch.stack(
        (policy_loss.detach(), value_loss.detach(), entropy.detach(), kl, clip_fraction)
    )

    return loss, metrics


def update_ppo(agent, optimizer, loss_function, rollout, args):
    observations = rollout.observations.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    log_probs = rollout.log_probs.flatten()
    advantages = rollout.advantages.flatten()
    returns = rollout.returns.flatten()
    values = rollout.values.flatten()
    device = observations.device
    metrics_sum = torch.zeros(5, device=device)
    updates = 0

    for _ in range(args.update_epochs):
        indices = torch.randperm(len(observations), device=device)
        epoch_kl = torch.zeros((), device=device)

        for batch in indices.split(args.minibatch_size):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)

            with autocast(device, args.bf16):
                loss, metrics = loss_function(
                    agent,
                    observations[batch],
                    actions[batch],
                    log_probs[batch],
                    advantages[batch],
                    returns[batch],
                    values[batch],
                    args.clip_coef,
                    args.ent_coef,
                    args.vf_coef,
                    args.clip_value_loss,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics_sum.add_(metrics)
            epoch_kl.add_(metrics[3] * len(batch))
            updates += 1

        if args.target_kl is not None and (epoch_kl / len(observations)).item() > args.target_kl:
            break

    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")

    return dict(zip(names, (metrics_sum / updates).cpu().tolist()))


def prepare(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires CUDA")

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use --no-bf16")

    env = build_env(args)

    try:
        agent = Agent(
            env.single_observation_space.shape[0],
            env.single_action_space.nvec,
            env.action_codec,
            args.hidden_size,
            args.layers,
        ).to(env.device)
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True
        )
        rollout = Rollout.allocate(
            args.rollout,
            env.n_envs,
            env.single_observation_space.shape[0],
            len(env.single_action_space.nvec),
            env.device,
        )
    except BaseException:
        env.close()
        raise

    return env, agent, optimizer, rollout


def compile_functions(agent, args):
    policy = agent.get_action_and_value
    loss_function = ppo_loss
    gae_function = compute_gae

    if args.compile:
        mode = "reduce-overhead" if args.cuda_graphs else "default"
        policy = torch.compile(policy, fullgraph=True, mode=mode)
        loss_function = torch.compile(loss_function, fullgraph=True, mode=mode)
        gae_function = torch.compile(gae_function, fullgraph=True)

    return policy, loss_function, gae_function


def save_checkpoint(path, algorithm, args, agent, optimizer, step, **extra):
    payload = {
        "format": "goddard-leanrl-v1",
        "algorithm": algorithm,
        "config": vars(args),
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        **extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, algorithm, args, agent, optimizer):
    # Resume only trusted local checkpoints: optimizer/RNG payloads use pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("format") != "goddard-leanrl-v1" or payload.get("algorithm") != algorithm:
        raise ValueError("resume requires a checkpoint from this implementation and algorithm")

    keys = ["team_size", "hidden_size", "layers", "frameskip", "gamma"]

    if algorithm == "gaifo":
        keys += ["trajectory_length", "discriminator_hidden", "factorize", "near_distance"]

    for key in keys:
        if payload["config"][key] != getattr(args, key):
            raise ValueError(f"checkpoint configuration differs: {key}")

    agent.load_state_dict(payload["agent"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])

    return payload


def run_directory(args, algorithm):
    name = args.run_name or f"{algorithm}-{datetime.now():%Y%m%d-%H%M%S}-{args.seed}"
    path = args.output_dir / name
    path.mkdir(parents=True, exist_ok=True)

    return path


def log_metrics(writer, step, metrics):
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)

    print(
        f"step={step:,} " + " ".join(f"{key}={value:.4g}" for key, value in metrics.items()),
        flush=True,
    )


def add_ppo_arguments(parser):
    environment = parser.add_argument_group("CARL")
    environment.add_argument("--n-sim", type=int, default=8192)
    environment.add_argument("--team-size", type=int, choices=(1, 2, 3), default=1)
    environment.add_argument("--frameskip", type=int, default=4)
    environment.add_argument("--max-ticks", type=int, default=36000)
    environment.add_argument("--no-touch-timeout", type=float, default=30.0)

    training = parser.add_argument_group("PPO")
    training.add_argument("--timesteps", type=int, default=100_000_000)
    training.add_argument("--rollout", type=int, default=32)
    training.add_argument("--minibatch-size", type=int, default=16_384)
    training.add_argument("--update-epochs", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--learning-rate-end", type=float, default=1e-5)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--gae-lambda", type=float, default=0.95)
    training.add_argument("--clip-coef", type=float, default=0.2)
    training.add_argument("--clip-value-loss", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--ent-coef", type=float, default=0.01)
    training.add_argument("--vf-coef", type=float, default=0.5)
    training.add_argument("--max-grad-norm", type=float, default=0.5)
    training.add_argument("--target-kl", type=float)
    training.add_argument("--hidden-size", type=int, default=256)
    training.add_argument("--layers", type=int, default=2)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=1)
    runtime.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=False)
    runtime.add_argument("--output-dir", type=Path, default=Path("runs"))
    runtime.add_argument("--run-name")
    runtime.add_argument("--resume", type=Path)
    runtime.add_argument("--save-every", type=int, default=100)


def train(args):
    env, agent, optimizer, rollout = prepare(args)
    writer = None

    try:
        step = 0

        if args.resume:
            step = load_checkpoint(args.resume, "ppo", args, agent, optimizer)["step"]

        policy, loss_function, gae_function = compile_functions(agent, args)
        directory = run_directory(args, "ppo")
        writer = SummaryWriter(directory)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        iteration = 0

        while step < args.timesteps:
            started = time.perf_counter()
            progress = min(step / args.timesteps, 1.0)
            optimizer.param_groups[0]["lr"] = args.learning_rate + progress * (
                args.learning_rate_end - args.learning_rate
            )
            observation, episode_returns = collect(env, agent, policy, rollout, observation, args)

            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation)

            gae_function(rollout, next_value, args.gamma, args.gae_lambda)
            metrics = update_ppo(agent, optimizer, loss_function, rollout, args)
            step += rollout.rewards.numel()
            iteration += 1
            torch.cuda.synchronize(env.device)
            metrics["transitions_per_second"] = rollout.rewards.numel() / (
                time.perf_counter() - started
            )
            metrics["reward"] = rollout.rewards.mean().item()

            if episode_returns:
                metrics["episode_return"] = float(np.mean(episode_returns))

            log_metrics(writer, step, metrics)

            if iteration % args.save_every == 0 or step >= args.timesteps:
                save_checkpoint(directory / "latest.pt", "ppo", args, agent, optimizer, step)
    finally:
        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean, GPU-resident PPO for CARL.")
    add_ppo_arguments(parser)
    args = parser.parse_args()
    train(args)
