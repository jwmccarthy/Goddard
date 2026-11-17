import inspect
import random

import numpy as np
import torch

from training.arguments import entropy_coefficients, resolve_critic_arguments, resolve_joint_actions
from training.models import Agent
from training.observations import observation_size, resolve_observation_arguments
from training.ppo import ppo_loss
from training.rollout import Rollout, compute_gae


def build_env(args, reward_funcs=None):
    import carl
    from carl.gymnasium import CARLTorchVectorEnv

    schema, previous_controls = resolve_observation_arguments(args)
    version = getattr(carl, "OBS_CONTROLLER_VERSION", None)
    options = {}
    if schema == "ego-controller-v3":
        if (version != 3 or getattr(carl, "OBS_EGO_CONTROLLER", None) != 5
                or getattr(carl, "OBS_PREVIOUS_CONTROLS", None) != 8):
            raise RuntimeError("ego-controller-v3 requires merged CARL master; run uv sync and "
                               "remove any legacy CARL PYTHONPATH override")
        options["previous_controls"] = previous_controls
    elif schema == "ego-controller-v1":
        if version != 1:
            raise RuntimeError(
                "ego-controller-v1 checkpoints require the legacy v1 staged CARL package; "
                "merged CARL cannot reconstruct removed observations. Select that package "
                "with PYTHONPATH or start a new v3 run."
            )
        options["ego_controller"] = True
    elif version not in (None, 1):
        raise RuntimeError("base-v1 checkpoints require a legacy CARL package; merged CARL "
                           "always includes controller state")
    experimental = getattr(args, "experimental_jump_phase", False)
    if experimental and not hasattr(carl.Env, "reset_selected"):
        raise RuntimeError("--experimental-jump-phase requires CARL's reset_selected API, "
                           "which is only available in the legacy events-enabled build")

    parameters = inspect.signature(CARLTorchVectorEnv).parameters
    for name in ("masked_resets", "defer_episode_reporting"):
        if name in parameters:
            options[name] = getattr(args, name, True) and not experimental

    env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=args.team_size,
        n_orange=args.team_size,
        seed=args.seed,
        frameskip=1 if experimental else args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        discrete_actions=True,
        reward_funcs=None if experimental else reward_funcs,
        **options,
    )
    expected = observation_size(schema, env.n_cars, previous_controls)
    if env.single_observation_space.shape != (expected,):
        env.close()
        raise RuntimeError(f"CARL observation layout does not match {schema}: expected {expected} inputs")
    if experimental:
        from training.jump_phase import JumpPhaseEnv

        env = JumpPhaseEnv(env, args.frameskip)
        for function in reward_funcs or ():
            env.register_reward(function)
    return env


def prepare(args, reward_funcs=None, *, agent_class=Agent):
    resolve_joint_actions(args)
    resolve_critic_arguments(args)
    entropy_coefficients(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires CUDA")

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use --no-bf16")

    torch.set_float32_matmul_precision(args.matmul_precision)
    env = build_env(args, reward_funcs)

    try:
        agent = agent_class(
            env.single_observation_space.shape[0],
            env.single_action_space.nvec,
            env.action_codec,
            args.hidden_size,
            args.layers,
            mlp_critic=args.mlp_critic,
            critic_window_length=args.critic_window_length,
            critic_relative=args.critic_relative,
            n_cars=env.n_cars,
            joint_actions=args.joint_actions,
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
        if args.mlp_critic:
            rollout.enable_critic_history(args.critic_window_length, args.critic_decision_stride)
    except BaseException:
        env.close()
        raise

    return env, agent, optimizer, rollout


def compilation_mode(args, *, cuda_graphs=True):
    graphs = cuda_graphs and args.cuda_graphs
    if getattr(args, "compile_autotune", False):
        return "max-autotune" if graphs else "max-autotune-no-cudagraphs"
    return "reduce-overhead" if graphs else "default"


def compile_functions(agent, args):
    policy = agent.get_action_and_value
    loss_function = ppo_loss
    gae_function = compute_gae

    if args.compile:
        configure_compilation_warnings()
        mode = compilation_mode(args)
        policy = torch.compile(policy, fullgraph=True, mode=mode)
        loss_function = torch.compile(loss_function, fullgraph=True, mode=mode)
        gae_function = torch.compile(gae_function, fullgraph=True)

    return policy, loss_function, gae_function


def configure_compilation_warnings():
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
        module=r"torch\._inductor\.select_algorithm",
    )

    # This is Inductor choosing a correct split reduction, not a graph break,
    # recompile or numerical error. Keep all other compiler warnings visible.
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*Online softmax is disabled on the fly since Inductor decides to.*split the reduction.*",
        category=UserWarning,
        module=r"torch\._inductor\.lowering",
    )


def _autotune_log_filter(record):
    # Inductor logs rejected candidates as errors while continuing the search.
    return not (
        "Ignoring this choice." in record.getMessage()
        or record.msg == "Exception %s for benchmark choice %s"
    )
