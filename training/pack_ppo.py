import random

import numpy as np
import torch
import torch.nn.functional as F
from carl.gymnasium import CARLTorchVectorEnv
from carl.training import (
    CompiledTrainingPack,
    TrainingPackResetProvider,
    TrainingPackRuntime,
    TrainingPackTerminationManager,
)

from training.models import Agent
from training.rollout import Rollout
from training.tensors import plain_tensor
from training.wall_flip_reset import wall_flip_reset_pack


class PackActionCodec:

    def __init__(self, codec, observation_size):
        self.codec = codec
        self.observation_size = observation_size

    def mask(self, observation):
        return self.codec.mask(observation[..., :self.observation_size])


class StageRewardManager(TrainingPackTerminationManager):

    def __init__(self, runtime, provider, base, multiplier):
        super().__init__(runtime, provider)

        if not np.isfinite(base) or not np.isfinite(multiplier) or base <= 0 or multiplier < 1:
            raise ValueError("stage reward base must be positive and multiplier must be at least 1")

        self.base = base
        self.multiplier = multiplier
        self.counts = torch.zeros(runtime.pack.kind.shape[1], device=runtime.active.device, dtype=torch.int64)
        self.repeat_count = torch.zeros((), device=runtime.active.device, dtype=torch.int64)
        self.last_reward = torch.zeros(runtime.n_sim, device=runtime.active.device)
        self.transition_stage = None
        self.transition_remaining = None
        self.transition_exercise = None
        self.transition_repeats = None
        self.transition_total_matches = None

    def step(self, events, frameskip, state=None):
        stage = self.runtime.stage_index.clone()
        result = super().step(events, frameskip, state=state)

        self.transition_stage = self.runtime.stage_index.clone()
        self.transition_remaining = self.runtime.ticks_remaining.clone()
        self.transition_exercise = self.runtime.exercise_index.clone()
        self.transition_repeats = (self.runtime.repeat_count.clone()
                                   if hasattr(self.runtime, "repeat_count") else None)
        self.transition_total_matches = (self.runtime.total_matches.clone()
                                         if hasattr(self.runtime, "total_matches") else None)
        self.last_reward = result.advanced.float() * self.base * self.multiplier ** stage.float()
        if result.repeated is not None:
            self.repeat_count += result.repeated.sum()
        self.counts.scatter_add_(0, stage, result.advanced.long())

        return result

    def reward(self, context):
        return self.last_reward[:, None]

    def diagnostic_metrics(self):
        counts = self.counts.cpu().tolist()
        repeats = self.repeat_count.item()
        self.counts.zero_()
        self.repeat_count.zero_()

        return {**{f"exercise/stage_{index + 1}_passed": count for index, count in enumerate(counts)},
                "exercise/repeated_touches": repeats}


class PackObservationEnv:

    def __init__(self, env, runtime, include_repeat_count=False):
        self.env = env
        self.runtime = runtime
        self.n_stages = runtime.pack.kind.shape[1]
        self.include_repeat_count = include_repeat_count

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation(self, value, stage, remaining, exercise=None, repeats=None):
        value = plain_tensor(value)
        stage_features = F.one_hot(stage, num_classes=self.n_stages).to(value.dtype)
        exercise = self.runtime.exercise_index if exercise is None else exercise
        deadlines = self.runtime.pack.deadline[exercise, stage]
        fraction = torch.where(deadlines > 0, remaining.clamp_min(0) / deadlines.clamp_min(1), 0)

        if self.include_repeat_count:
            repeats = self.runtime.total_matches if repeats is None else repeats
            limit = self.runtime.pack.max_total_matches[exercise, stage]
            allowance = torch.where(limit >= 0, (limit - repeats).clamp_min(0) / limit.clamp_min(1), 0)
            value = torch.cat((value, allowance.to(value.dtype)[:, None]), dim=-1)

        return torch.cat((value, stage_features, fraction[:, None]), dim=-1)

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation, self.runtime.stage_index, self.runtime.ticks_remaining)

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)

        if "final_obs" in info:
            manager = self.env.termination_manager
            stage = manager.transition_stage
            remaining = manager.transition_remaining
            exercise = manager.transition_exercise
            value = plain_tensor(info["final_obs"])
            info["final_obs"] = self.observation(value, stage, remaining, exercise,
                                                 getattr(manager, "transition_total_matches", None))

        observation = self.observation(observation, self.runtime.stage_index, self.runtime.ticks_remaining)
        return observation, reward, terminated, truncated, info


def prepare_pack_ppo(args):
    if args.team_size != 1:
        raise ValueError("wall flip-reset PPO requires --team-size 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires CUDA")

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use --no-bf16")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pack = wall_flip_reset_pack()
    provider = TrainingPackResetProvider(pack, n_sim=args.n_sim, seed=args.seed)
    compiled = CompiledTrainingPack.compile(pack)
    runtime = TrainingPackRuntime(compiled, n_sim=args.n_sim)
    manager = StageRewardManager(runtime, provider, args.stage_reward_base, args.stage_reward_multiplier)
    env = CARLTorchVectorEnv(
        n_sim                    = args.n_sim,
        n_blue                   = 1,
        n_orange                 = 0,
        seed                     = args.seed,
        frameskip                = args.frameskip,
        max_ticks                = args.max_ticks,
        no_touch_timeout_seconds = args.no_touch_timeout,
        normalize                = True,
        discrete_actions         = True,
        capture_events           = True,
        capture_ball_touches     = compiled.needs_touch,
        reset_state_provider     = provider,
        termination_manager      = manager,
        reward_funcs             = [manager.reward],
    )

    try:
        observation_size = env.single_observation_space.shape[0] + runtime.pack.kind.shape[1] + 1
        action_codec = PackActionCodec(env.action_codec, env.single_observation_space.shape[0])
        agent = Agent(
            observation_size, env.single_action_space.nvec, action_codec,
            args.hidden_size, args.layers,
        ).to(env.device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps, fused=True)
        rollout = Rollout.allocate(args.rollout, env.n_envs, observation_size, len(env.single_action_space.nvec), env.device)
    except BaseException:
        env.close()
        raise

    return PackObservationEnv(env, runtime), agent, optimizer, rollout, manager
