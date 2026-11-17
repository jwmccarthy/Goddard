"""Opt-in real-tick discriminator views; policy decisions retain their cadence."""
from types import SimpleNamespace

import torch

from training.imitation import TransitionRollout
from training.tensors import plain_tensor


class JumpPhaseEnv:
    """Aggregate one-tick CARL steps into one policy/reward transition.

    Replay resets happen once at the decision boundary. After a terminal tick,
    remaining substeps use neutral controls in the native kickoff; those ticks
    never become part of the ending episode's reward or discriminator history.
    """

    def __init__(self, env, frameskip):
        if env._env.frameskip != 1 or frameskip < 2:
            raise ValueError("jump-phase adapter requires one-tick CARL and a multi-tick hold")
        self.env = env
        self.frameskip = frameskip
        self.scene_size = 9 + 21 * env.n_cars
        self.reset_state_provider = None
        self.reward_funcs = []
        self._returns = torch.zeros(env.n_envs, device=env.device)
        self._lengths = torch.zeros(env.n_envs, dtype=torch.long, device=env.device)
        self._tick_capture = None
        self._tick_indices = torch.arange(frameskip, device=env.device)[:, None]
        self._sim_indices = torch.arange(env.n_sim, device=env.device)
        env.register_reward(self._capture_context)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def register_reward(self, function):
        self.reward_funcs.append(function)
        return function

    def _apply_resets(self, mask):
        self.env.reset_state_provider = self.reset_state_provider
        try:
            self.env._apply_reset_state(mask)
        finally:
            self.env.reset_state_provider = None

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self._apply_resets(torch.ones(self.n_sim, dtype=torch.bool, device=self.device))
        self._returns.zero_()
        self._lengths.zero_()
        observation = self.env._observe()
        self.events = getattr(self.env, "events", None)
        return observation

    def _capture_context(self, context):
        active = ~self._ended
        self._raw = torch.where(active[:, None], context.current.raw, self._raw)
        self._observation = torch.where(
            active.repeat_interleave(self.n_cars)[:, None],
            plain_tensor(context.current_observation), self._observation,
        )
        self._touches |= context.current.car_ball_touches & active[:, None]
        self._score += context.events.score_delta * active
        self._terminated |= context.events.terminated & active
        self._truncated |= context.events.truncated & active
        self._repeated_done |= self._ended & context.events.done
        self._first_done = torch.where(
            active & context.events.done, self._tick, self._first_done
        )
        self._ended |= context.events.done
        for name in ("score_difference", "episode_ticks", "overtime"):
            setattr(self, "_" + name, torch.where(active, getattr(context, name), getattr(self, "_" + name)))
        return context.current.raw.new_zeros((self.n_sim, self.n_cars))

    def _step_tick(self, actions):
        """Collect native transition data without Gym's per-tick bookkeeping.

        In particular, avoid its done.nonzero() host synchronization, duplicate
        state/observation clones, unused rewards and episode-return accounting.
        Native same-step autoresets still occur; replay resets remain macro-only.
        """
        env = self.env
        # Exercise managers can impose extra terminations beyond native dones.
        if getattr(env, "termination_manager", None) is not None:
            return env.step(actions.reshape(self.n_envs, -1))[0]
        native = env._env
        env._sync()
        if env.discrete_actions:
            native.step(actions)
        else:
            native.step_continuous(actions)
        env._sync()
        if hasattr(native, "events"):
            native.events.acquire(torch.cuda.current_stream(self.device).cuda_stream)

        def tensor(capsule):
            return env._tensor(capsule, copy=False)

        score = tensor(native.get_rewards())
        done = tensor(native.get_dones())
        capture = self._tick_capture
        capture["raw"][self._tick].copy_(tensor(native.get_transition_state()))
        capture["observation"][self._tick].copy_(
            tensor(native.get_transition_obs()).view(self.n_envs, native.obs_dim)
        )
        capture["score"][self._tick].copy_(score)
        capture["done"][self._tick].copy_(done)
        for name in ("score_difference", "episode_ticks", "overtime"):
            capture[name][self._tick].copy_(tensor(getattr(native, "get_transition_" + name)()))
        self._ended |= done
        # Borrowed native storage is consumed/copied before the next step.
        return tensor(native.get_obs()).view(self.n_envs, native.obs_dim)

    def _finish_capture(self):
        """Reduce captured ticks once per hold, retaining true first endpoints."""
        import carl

        capture = self._tick_capture
        done, score = capture["done"], capture["score"]
        first = torch.where(done, self._tick_indices, self.frameskip - 1).amin(0)
        active = self._tick_indices <= first[None]
        self._first_done = first
        self._score = (score * active).sum(0)
        self._terminated = (done & score.ne(0) & active).any(0)
        self._truncated = (done & score.eq(0) & active).any(0)
        self._repeated_done = (done & ~active).any(0)
        raw = capture["raw"]
        car_end = carl.OBS_BALL + carl.STATE_PER_CAR * self.n_cars
        cars = raw[..., carl.OBS_BALL:car_end].view(self.frameskip, self.n_sim, self.n_cars, carl.STATE_PER_CAR)
        self._touches = (cars[..., carl.STATE_BALL_TOUCH].bool() & active[..., None]).any(0)
        self._raw = raw[first, self._sim_indices]
        observations = capture["observation"].view(self.frameskip, self.n_sim, self.n_cars, -1)
        self._observation = observations[first, self._sim_indices].flatten(0, 1)
        for name in ("score_difference", "episode_ticks", "overtime"):
            setattr(self, "_" + name, capture[name][first, self._sim_indices])

    @torch.no_grad()
    def step(self, actions):
        import carl
        from carl.gymnasium import state as state_views
        from carl.gymnasium.state import CARLObservation, CarlState, RewardResult

        self.env._check_open()
        dtype = torch.int32 if self.discrete_actions else torch.float32
        actions = torch.as_tensor(actions, device=self.device, dtype=dtype).contiguous()
        if actions.shape != (self.n_envs, len(carl.ACTION_NVECS)):
            raise ValueError(f"Expected actions shaped {(self.n_envs, len(carl.ACTION_NVECS))}")
        native_actions = actions.view(self.n_sim, self.n_cars, -1)
        previous = self.env._state
        previous_observation = self.env._observation
        self._raw = previous.raw.clone()
        self._observation = plain_tensor(previous_observation).clone()
        self._touches = torch.zeros((self.n_sim, self.n_cars), dtype=torch.bool, device=self.device)
        self._score = torch.zeros(self.n_sim, device=self.device)
        self._ended = torch.zeros(self.n_sim, dtype=torch.bool, device=self.device)
        self._terminated = self._ended.clone()
        self._truncated = self._ended.clone()
        self._repeated_done = self._ended.clone()
        self._first_done = torch.full((self.n_sim,), self.frameskip - 1, device=self.device, dtype=torch.long)
        for name in ("score_difference", "episode_ticks", "overtime"):
            setattr(self, "_" + name, getattr(self.env, "_" + name).clone())
        batched_capture = getattr(self.env, "termination_manager", None) is None
        if batched_capture and self._tick_capture is None:
            def tick_buffer(tensor):
                # A strided per-tick destination uses a device copy kernel,
                # avoiding many small cudaMemcpyAsync driver submissions.
                return tensor.new_empty((len(tensor), self.frameskip, *tensor.shape[1:])).transpose(0, 1)

            self._tick_capture = {
                "raw": tick_buffer(previous.raw),
                "observation": tick_buffer(self._observation),
                "score": tick_buffer(self._score),
                "done": tick_buffer(self._ended),
                **{name: tick_buffer(getattr(self, "_" + name))
                   for name in ("score_difference", "episode_ticks", "overtime")},
            }
        starts = self._observation.new_empty((self.frameskip, self.n_envs, self.scene_size))
        boundaries = torch.empty((self.frameskip, self.n_envs), dtype=torch.bool, device=self.device)
        observation = previous_observation
        for self._tick in range(self.frameskip):
            starts[self._tick].copy_(plain_tensor(observation)[:, :self.scene_size])
            held = native_actions.masked_fill(self._ended[:, None, None], 0)
            observation = self._step_tick(held)
            boundaries[self._tick].copy_(self._ended[:, None].expand(-1, self.n_cars).reshape(-1))
        if batched_capture:
            self._finish_capture()
        repeated, any_done = torch.stack((self._repeated_done.any(), self._ended.any())).tolist()
        if repeated:
            raise RuntimeError("multiple episode boundaries within an experimental jump-phase decision")

        current = CarlState.from_raw(self._raw, self.n_cars, self._boost_pad_positions, self._team_sign)
        current.car_values[..., carl.STATE_BALL_TOUCH] = self._touches
        self.events = SimpleNamespace(
            score_delta=self._score, done=self._ended,
            terminated=self._terminated, truncated=self._truncated,
        )
        context_type = getattr(state_views, "StepContext", None) or state_views.RewardContext
        context = context_type(
            current=current, previous=previous,
            current_observation=CARLObservation.from_tensor(self._observation, self.n_cars),
            previous_observation=previous_observation, events=self.events, actions=actions,
            score_difference=self._score_difference, episode_ticks=self._episode_ticks,
            overtime=self._overtime,
        )
        reward = self._score[:, None] * self._team_sign
        reward_info = {}
        if self.reward_funcs:
            reward = current.raw.new_zeros((self.n_sim, self.n_cars))
            for function in self.reward_funcs:
                result = function(context)
                if isinstance(result, RewardResult):
                    reward_info.update(result.info)
                    result = result.reward
                if result.shape != reward.shape:
                    raise ValueError(f"Reward must have shape {tuple(reward.shape)}")
                reward += result
        reward = reward.reshape(self.n_envs) * self.env.reward_scale
        self._returns += reward
        self._lengths += 1
        done = self._ended.repeat_interleave(self.n_cars)
        info = {
            "reward": [], "length": [],
            **reward_info,
            "jump_phase_starts": starts,
            "jump_phase_boundaries": boundaries,
            "jump_phase_endpoint": self._first_done.repeat_interleave(self.n_cars),
        }
        if any_done:
            info.update(
                reward=self._returns[done].cpu().tolist(), length=self._lengths[done].cpu().tolist(),
                final_obs=CARLObservation.from_tensor(self._observation, self.n_cars), _final_obs=done,
            )
            self._returns[done] = 0
            self._lengths[done] = 0
            # Discard the quarantined kickoff substeps before applying one
            # normal replay-reset draw. No leftover old action reaches it.
            self.env._env.reset_selected(self._ended.contiguous())
            self.env._clear_sim_stats(self._ended)
            self._apply_resets(self._ended)
        # Only refresh the wrapped environment's persistent state once per hold.
        observation = self.env._observe()
        return (
            observation, reward, self._terminated.repeat_interleave(self.n_cars),
            self._truncated.repeat_interleave(self.n_cars), info,
        )


class JumpPhaseRollout(TransitionRollout):
    """Real tick starts plus true macro endpoints, without storing all phase windows."""

    phase_pairs = True

    def __init__(self, observation, steps, scene_size, window_length, decision_stride, frameskip):
        super().__init__(observation, steps, scene_size, window_length, decision_stride)
        self.hold = frameskip
        self.spacing = frameskip * decision_stride
        self.tick_prefix_size = (window_length - 1) * self.spacing + frameskip
        shape = (steps * frameskip, len(observation))
        # Keep history contiguous from the outset: assembling it lazily used to
        # copy the entire (large) physics-tick rollout on every iteration.
        total = self.tick_prefix_size + shape[0]
        self.tick_frames = observation.new_empty((total + 1, len(observation), scene_size))
        self.tick_frames[-1].zero_()
        self.tick_boundaries = torch.zeros((total, len(observation)), dtype=torch.bool, device=observation.device)
        self.tick_prefix = self.tick_frames[:self.tick_prefix_size]
        self.tick_prefix.fill_(float("nan"))
        self.ticks = self.tick_frames[self.tick_prefix_size:-1]
        self.done_prefix = self.tick_boundaries[:self.tick_prefix_size]
        self.tick_dones = self.tick_boundaries[self.tick_prefix_size:]
        self.done_prefix[-1] = True
        self.endpoints = torch.empty(self.scenes.shape[:2], dtype=torch.long, device=observation.device)
        self.last_boundaries = None
        self.phase_offsets = torch.arange(1 - window_length, 0, device=observation.device) * self.spacing
        self.tick_indices = torch.arange(total, device=observation.device)[:, None]
        self.started_ticks = False

    def record_ticks(self, step, info):
        if step == 0:
            self.last_boundaries = None
            if self.started_ticks:
                count = self.tick_prefix_size
                if count <= len(self.ticks):
                    self.tick_prefix.copy_(self.ticks[-count:])
                    self.done_prefix.copy_(self.tick_dones[-count:])
                else:
                    self.tick_prefix.copy_(torch.cat((self.tick_prefix, self.ticks))[-count:])
                    self.done_prefix.copy_(torch.cat((self.done_prefix, self.tick_dones))[-count:])
        start = step * self.hold
        self.ticks[start:start + self.hold].copy_(info["jump_phase_starts"])
        self.tick_dones[start:start + self.hold].copy_(info["jump_phase_boundaries"])
        self.endpoints[step].copy_(info["jump_phase_endpoint"])
        self.started_ticks = True

    @property
    def transition_ticks(self):
        return self.endpoints + 1

    def _phase_windows(self, indices, views):
        if self.last_boundaries is None:
            # One scan per rollout, rather than a dense lookback per sampled
            # window/view. Includes every intervening tick, not just frame times.
            self.last_boundaries = torch.where(
                self.tick_boundaries, self.tick_indices, -1,
            ).cummax(0).values
        agents = self.scenes.shape[1]
        step, actor = indices // agents, indices % agents
        anchor = self.tick_prefix_size + step * self.hold + self.endpoints[step, actor]
        # Do not rephase across a recent reset, even if this episode has fewer
        # than `hold` ticks. Initial missing history is handled as padding.
        last = self.last_boundaries[anchor - 1, actor]
        available = (anchor - last).clamp(1, self.hold)
        phase = (torch.rand((len(indices), views), device=indices.device) * available[:, None]).long()
        end = anchor[:, None] - phase
        times = end[..., None] + 1 + self.phase_offsets
        history = self.tick_frames[times, actor[:, None, None]]
        last_reset = self.last_boundaries[end - 1, actor[:, None]] + 1
        history = history.masked_fill((times < last_reset[..., None])[..., None], float("nan"))
        endpoint = self.tick_frames[end + 1, actor[:, None]]
        # At a macro boundary or a terminal tick the following start can be a
        # reset state. Always substitute the saved true transition endpoint.
        endpoint = torch.where(
            (phase == 0)[..., None], self.successors[step, actor][:, None], endpoint
        )
        return torch.cat((history, endpoint[..., None, :]), -2)

    def transition_pairs(self, indices):
        return self._phase_windows(indices, 2)

    def transitions(self, indices):
        return self._phase_windows(indices, 1)[:, 0]
