from __future__ import annotations

import random
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from training.models import Agent
from training.ppo import ppo_loss
from training.rollout import Rollout, compute_gae

if TYPE_CHECKING:
    from carl.gymnasium import CARLTorchVectorEnv


def build_env(
    args:         Namespace,
    reward_funcs: Callable[..., torch.Tensor] | None = None
) -> CARLTorchVectorEnv:
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
            mlp_critic=args.mlp_critic,
            critic_window_length=args.critic_window_length,
            critic_relative=args.critic_relative,
            n_cars=env.n_cars,
            joint_actions=args.joint_actions
        ).to(env.device)

        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=args.learning_rate,
            eps=args.adam_eps,
            fused=True
        )

        rollout = Rollout.allocate(
            args.rollout,
            env.n_envs,
            env.single_observation_space.shape[0],
            len(env.single_action_space.nvec),
            env.device
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


def compilation_mode(*, cuda_graphs: bool = True) -> str:
    return "max-autotune" if cuda_graphs else "max-autotune-no-cudagraphs"


def compile_functions(
    agent: Agent,
    args:  Namespace
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    policy = agent.get_action_and_value
    loss_function = ppo_loss
    gae_function = compute_gae

    if args.compile:
        configure_compilation_warnings()
        mode = compilation_mode()
        policy = torch.compile(policy, fullgraph=True, mode=mode)

        loss_function = torch.compile(loss_function, fullgraph=True, mode=mode)

        gae_function = torch.compile(gae_function, fullgraph=True)

    return policy, loss_function, gae_function


def configure_compilation_warnings() -> None:
    import logging
    import warnings

    from torch._inductor import config

    config.autotune_num_choices_displayed = 0
    config.max_autotune_report_choices_stats = False
    logger = logging.getLogger("torch._inductor.select_algorithm")

    if _autotune_log_filter not in logger.filters:
        logger.addFilter(_autotune_log_filter)

    warnings.filterwarnings(
        "ignore",
        message=r"TypedStorage is deprecated\..*",
        category=UserWarning,
        module=r"torch\._inductor\.select_algorithm"
    )

    # This is Inductor choosing a correct split reduction, not a graph break,
    # recompile or numerical error. Keep all other compiler warnings visible.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"(?s).*Online softmax is disabled on the fly since Inductor "
            r"decides to.*split the reduction.*"
        ),
        category=UserWarning,
        module=r"torch\._inductor\.lowering"
    )


def _autotune_log_filter(record: Any) -> bool:
    # Inductor logs rejected candidates as errors while continuing the search.
    return not (
        "Ignoring this choice." in record.getMessage()
        or record.msg == "Exception %s for benchmark choice %s"
    )
