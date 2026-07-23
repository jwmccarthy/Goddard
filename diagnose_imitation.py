import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect.runner import _reset_state
from jarl.envs import DatasetResetSampler

from imitation_dataset import load_expert_dataset
from replay_states import load_replay_dataset
from watch_checkpoints import load_actor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure expert and CARL observation separability"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--expert-dataset",
        type=Path,
        default=Path("data/ballchasing-ssl-1v1/expert_dataset"),
    )
    parser.add_argument(
        "--replay-dataset",
        type=Path,
        default=Path("data/ballchasing-ssl-1v1/reset_dataset"),
    )
    parser.add_argument("--num-simulations", type=int, default=256)
    parser.add_argument("--sequence-count", type=int, default=16_384)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--replay-reset-probability", type=float, default=0.7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--max-probe-examples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def project(observation: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (observation[..., :9], observation[..., 9:25], observation[..., 30:46]),
        dim=-1,
    )


@torch.inference_mode()
def collect_policy_sequences(
    environment: CARLTorchVectorEnv,
    actor,
    count: int,
    sequence_length: int,
) -> torch.Tensor:
    observation = environment.reset()
    state = actor.initial_state(environment.n_envs)
    sequences = []
    collected = 0

    while collected < count:
        steps = []
        valid = torch.ones(
            environment.n_envs,
            dtype=torch.bool,
            device=environment.device,
        )
        for _ in range(sequence_length):
            steps.append(observation)
            output = actor.act(observation, state)
            observation, _, terminated, truncated, _ = environment.step(output.action)
            done = terminated.bool() | truncated.bool()
            valid &= ~done
            state = _reset_state(output.next_state, done)

        sequence = torch.stack(steps, dim=1)[valid]
        remaining = count - collected
        sequence = sequence[:remaining].cpu()
        sequences.append(sequence)
        collected += len(sequence)

    return torch.cat(sequences)


def add_noise(
    value: torch.Tensor,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not std:
        return value
    return value + torch.randn(value.shape, generator=generator) * std


def sequence_summary(sequence: torch.Tensor) -> torch.Tensor:
    delta = sequence[:, 1:] - sequence[:, :-1]
    return torch.cat(
        (
            sequence.mean(dim=1),
            sequence.std(dim=1),
            delta.mean(dim=1),
            delta.std(dim=1),
            delta.abs().mean(dim=1),
        ),
        dim=1,
    )


def shuffle_time(
    sequence: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    order = torch.rand(sequence.shape[:2], generator=generator).argsort(dim=1)
    return sequence.gather(1, order[..., None].expand_as(sequence))


def subsample(
    value: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if len(value) <= count:
        return value
    return value[torch.randperm(len(value), generator=generator)[:count]]


def probe_accuracy(
    expert: torch.Tensor,
    agent: torch.Tensor,
    steps: int,
    max_examples: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    count = min(len(expert), len(agent), max_examples)
    expert = subsample(expert, count, generator)
    agent = subsample(agent, count, generator)
    train_count = int(count * 0.8)
    expert_order = torch.randperm(count, generator=generator)
    agent_order = torch.randperm(count, generator=generator)
    train = torch.cat(
        (expert[expert_order[:train_count]], agent[agent_order[:train_count]])
    )
    train_target = torch.cat(
        (torch.zeros(train_count), torch.ones(train_count))
    )
    test = torch.cat(
        (expert[expert_order[train_count:]], agent[agent_order[train_count:]])
    )
    test_target = torch.cat(
        (torch.zeros(count - train_count), torch.ones(count - train_count))
    )

    mean = train.mean(dim=0)
    std = train.std(dim=0).clamp_min(1e-5)
    train = ((train - mean) / std).to(device)
    train_target = train_target.to(device)
    test = ((test - mean) / std).to(device)
    test_target = test_target.to(device)
    model = nn.Sequential(
        nn.Linear(train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        indices = torch.randint(len(train), (min(8192, len(train)),), device=device)
        loss = F.binary_cross_entropy_with_logits(
            model(train[indices]).squeeze(1), train_target[indices]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        prediction = model(test).squeeze(1) >= 0
        return prediction.eq(test_target.bool()).float().mean().item()


def effect_summary(expert: torch.Tensor, agent: torch.Tensor) -> str:
    pooled_std = ((expert.var(0) + agent.var(0)) / 2).sqrt().clamp_min(1e-6)
    effect = (expert.mean(0) - agent.mean(0)).abs() / pooled_std
    quantiles = torch.quantile(effect, torch.tensor([0.5, 0.95, 1.0]))
    return "median={:.3f} p95={:.3f} max={:.3f}".format(*quantiles.tolist())


def main() -> None:
    arguments = parse_arguments()
    if min(
        arguments.num_simulations,
        arguments.sequence_count,
        arguments.sequence_length,
        arguments.frameskip,
        arguments.probe_steps,
        arguments.max_probe_examples,
    ) < 1:
        raise ValueError("counts, lengths, and frameskip must be positive")
    if arguments.noise_std < 0:
        raise ValueError("noise standard deviation cannot be negative")
    if not 0 <= arguments.replay_reset_probability <= 1:
        raise ValueError("replay reset probability must be between zero and one")

    torch.manual_seed(arguments.seed)
    generator = torch.Generator().manual_seed(arguments.seed)
    reset_dataset = load_replay_dataset(arguments.replay_dataset, device="cuda:0")
    reset_sampler = DatasetResetSampler(
        reset_dataset,
        probability=arguments.replay_reset_probability,
        seed=arguments.seed,
    )
    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=1,
        n_orange=1,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=14_400,
        synchronize=False,
        reset_state_provider=reset_sampler,
        normalize=True,
    )
    try:
        actor = load_actor(arguments.checkpoint, environment)
        agent = collect_policy_sequences(
            environment,
            actor,
            arguments.sequence_count,
            arguments.sequence_length,
        )
    finally:
        environment.close()

    expert_dataset = load_expert_dataset(
        arguments.expert_dataset,
        arguments.frameskip,
        arguments.sequence_length,
    )
    expert_indices = torch.randint(
        len(expert_dataset),
        (arguments.sequence_count,),
        generator=generator,
    )
    expert = expert_dataset.data["observation"][expert_indices].float()
    expert = add_noise(project(expert), arguments.noise_std, generator)
    agent = add_noise(project(agent.float()), arguments.noise_std, generator)

    expert_frame = expert.flatten(0, 1)
    agent_frame = agent.flatten(0, 1)
    expert_delta = (expert[:, 1:] - expert[:, :-1]).flatten(0, 1)
    agent_delta = (agent[:, 1:] - agent[:, :-1]).flatten(0, 1)
    device = torch.device("cuda:0")

    print(f"sequences={len(expert):,} length={arguments.sequence_length}")
    print(f"noise_std={arguments.noise_std:.4f}")
    print(f"frame_effect {effect_summary(expert_frame, agent_frame)}")
    print(f"delta_effect {effect_summary(expert_delta, agent_delta)}")
    probes = (
        ("single_frame", expert_frame, agent_frame),
        ("one_step_delta", expert_delta, agent_delta),
        ("ordered_sequence", sequence_summary(expert), sequence_summary(agent)),
        (
            "shuffled_sequence",
            sequence_summary(shuffle_time(expert, generator)),
            sequence_summary(shuffle_time(agent, generator)),
        ),
    )
    for name, expert_features, agent_features in probes:
        accuracy = probe_accuracy(
            expert_features,
            agent_features,
            arguments.probe_steps,
            arguments.max_probe_examples,
            generator,
            device,
        )
        print(f"{name}_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
