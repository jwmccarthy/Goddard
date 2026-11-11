import random
import warnings

import numpy as np
import torch

from training.latents import LatentObservationEnv
from training.models import Agent, critic_hidden_size, critic_layers, policy_window_length
from training.ppo import ppo_loss
from training.rollout import Rollout, compute_gae


def build_env(args, reward_funcs=None):
    from carl.gymnasium import CARLTorchVectorEnv

    env = CARLTorchVectorEnv(
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
    )
    if getattr(args, "info", False):
        try:
            return LatentObservationEnv(env, args.info_codes)
        except BaseException:
            env.close()
            raise
    return env


def prepare(args, reward_funcs=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires CUDA")

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use --no-bf16")

    env = build_env(args, reward_funcs)

    try:
        agent = Agent(
            env.single_observation_space.shape[0],
            env.single_action_space.nvec,
            env.action_codec,
            args.hidden_size,
            args.layers,
            gru_critic=getattr(args, "gru_critic", False),
            gru_policy=getattr(args, "gru_policy", False),
            info_codes=args.info_codes if getattr(args, "info", False) else 0,
            policy_window_length=policy_window_length(args),
            critic_hidden_size=critic_hidden_size(args),
            critic_layers=critic_layers(args),
            mlp_policy=getattr(args, "mlp_policy", False),
            mlp_critic=getattr(args, "mlp_critic", False)
        ).to(env.device)
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=args.learning_rate, eps=args.adam_eps, fused=True
        )
        rollout = Rollout.allocate(
            args.rollout,
            env.n_envs,
            env.single_observation_space.shape[0],
            len(env.single_action_space.nvec),
            env.device
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
        if getattr(args, "gru_critic", False) or getattr(args, "gru_policy", False):
            warnings.warn("GRU PPO loss runs uncompiled; PyTorch does not support full-graph GRU compilation", stacklevel=2)
        else:
            loss_function = torch.compile(loss_function, fullgraph=True, mode=mode)
        gae_function = torch.compile(gae_function, fullgraph=True)

    return policy, loss_function, gae_function
