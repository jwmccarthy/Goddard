from __future__ import annotations

import random
from argparse import Namespace
from typing import Any, Callable

import numpy as np
import torch
from carl.gymnasium import CARLTorchVectorEnv

from training.checkpoints import read_checkpoint
from training.models import Agent
from training.ppo import ppo_loss
from training.residual import attach_residual_actor
from training.rollout import Rollout, compute_gae
from training.tensors import compilation_mode, configure_compilation_warnings


def build_env(
    args:         Namespace,
    reward_funcs: Callable[..., torch.Tensor] | None = None
) -> CARLTorchVectorEnv:
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
        reward_funcs=reward_funcs,
        previous_controls=args.previous_controls,
        masked_resets=True,
        defer_episode_reporting=True
    )


def prepare(
    args: Namespace
) -> tuple[CARLTorchVectorEnv, Agent, torch.optim.Optimizer, Rollout]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires CUDA")

    torch.set_float32_matmul_precision("high")
    env = build_env(args)

    try:
        agent = Agent(
            env.single_observation_space.shape[0],
            env.single_action_space.nvec,
            env.action_codec,
            args.hidden_size,
            args.layers,
            policy_hidden=args.policy_hidden_size,
            policy_layers=args.policy_layers,
            critic_hidden=args.critic_hidden_size,
            critic_layers=args.critic_layers,
            mlp_critic=args.mlp_critic,
            critic_window_length=args.critic_window_length,
            critic_relative=args.critic_relative,
            n_cars=env.n_cars,
            joint_actions=args.joint_actions,
            gru_policy=args.gru_policy,
            gru_critic=args.gru_critic
        )

        if getattr(args, "residual_policy", False):
            attach_residual_actor(
                agent, env.single_observation_space.shape[0], vars(args)
            )

            if not args.resume:
                payload = read_checkpoint(args.base_checkpoint)
                actor = {
                    key.removeprefix("actor."): value
                    for key, value in payload["agent"].items()
                    if key.startswith("actor.")
                }
                agent.actor.base.load_state_dict(actor, strict=True)
                read_checkpoint.cache_clear()

        agent = agent.to(env.device)

        optimizer = torch.optim.Adam(
            [
                parameter for parameter in agent.parameters()
                if parameter.requires_grad
            ],
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            fused=True
        )

        rollout = Rollout.allocate(
            args.rollout,
            env.n_envs,
            env.single_observation_space.shape[0],
            len(env.single_action_space.nvec),
            env.device
        )

        if args.gru_policy:
            rollout.enable_policy_memory(
                args.policy_hidden_size, args.gru_sequence_length
            )

        if args.gru_critic:
            rollout.enable_critic_memory(
                args.critic_hidden_size, args.gru_sequence_length
            )

        if args.mlp_critic:
            rollout.enable_critic_history(
                args.critic_window_length,
                args.critic_decision_stride
            )
    except BaseException:
        env.close()
        raise

    return env, agent, optimizer, rollout


def compile_functions(
    agent: Agent,
    args:  Namespace
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    policy = (
        agent.get_recurrent_action_and_value
        if args.gru_policy
        else agent.get_action_and_value
    )

    if args.gru_critic:
        policy = agent.get_recurrent_critic_action_and_value
    loss_function = ppo_loss
    gae_function = compute_gae

    if args.compile:
        configure_compilation_warnings()
        mode = compilation_mode(cuda_graphs=args.cuda_graphs)
        policy = torch.compile(policy, fullgraph=True, mode=mode)
        loss_function = torch.compile(loss_function, fullgraph=True, mode=mode)
        gae_function = torch.compile(gae_function, fullgraph=True)

    return policy, loss_function, gae_function
