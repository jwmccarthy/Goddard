import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from training.discriminator import (
    add_motion_noise,
    balanced_window_batch,
    discriminator_rewards,
    mirror_scene_windows,
    polyak_update,
)
from training.hierarchy import discriminator_scales
from training.tensors import autocast, require_finite


class TransitionRollout:
    def __init__(self, observation, steps, scene_size, window_length, decision_stride=1):
        self.scene_size = scene_size
        self.window_length = window_length
        self.stride = decision_stride
        self.history_length = (window_length - 1) * decision_stride + 1
        self.scenes = observation.new_empty((steps, len(observation), scene_size))
        self.successors = torch.empty_like(self.scenes)
        self.dones = torch.zeros(self.scenes.shape[:2], dtype=torch.bool, device=observation.device)
        self.terminated = torch.zeros_like(self.dones)
        self.score_deltas = observation.new_zeros(self.dones.shape)
        self.prefix = observation.new_full(
            (self.history_length - 2, len(observation), scene_size), float("nan")
        )
        self.prefix_dones = torch.zeros(
            self.prefix.shape[:2], dtype=torch.bool, device=observation.device
        )
        self.started = False
        self.frames = self.boundaries = None
        self.history_offsets = torch.arange(self.history_length - 1, device=observation.device)
        self.selected_offsets = self.history_offsets[:: self.stride]

    def record(
        self,
        step,
        observation,
        next_observation,
        final_observation,
        done,
        terminated,
        score_delta=None,
    ):
        if step == 0:
            self.frames = self.boundaries = None
            if self.started and len(self.prefix):
                if len(self.prefix) <= len(self.scenes):
                    # The carried history is usually much shorter than a rollout;
                    # do not concatenate/copy the entire rollout just to slice it.
                    self.prefix.copy_(self.scenes[-len(self.prefix) :])
                    self.prefix_dones.copy_(self.dones[-len(self.prefix) :])
                else:
                    self.prefix.copy_(torch.cat((self.prefix, self.scenes))[-len(self.prefix) :])
                    self.prefix_dones.copy_(
                        torch.cat((self.prefix_dones, self.dones))[-len(self.prefix) :]
                    )
            self.started = True
        self.scenes[step].copy_(observation[:, : self.scene_size])
        self.successors[step].copy_(
            torch.where(
                done[:, None],
                final_observation[:, : self.scene_size],
                next_observation[:, : self.scene_size],
            )
        )
        self.dones[step].copy_(done)
        self.terminated[step].copy_(terminated)
        if score_delta is None:
            self.score_deltas[step].zero_()
        else:
            self.score_deltas[step].copy_(score_delta)

    def transitions(self, indices):
        agents = self.scenes.shape[1]
        step, actor = indices // agents, indices % agents
        if self.frames is None:
            self.frames = torch.cat((self.prefix, self.scenes))
            self.boundaries = torch.cat((self.prefix_dones, self.dones))
        offsets = self.history_offsets
        times = step[:, None] + offsets
        previous_done = self.boundaries[times, actor[:, None]].clone()
        previous_done[:, -1] = False
        last_reset = torch.where(previous_done, offsets + 1, 0).amax(1)
        selected = self.selected_offsets
        history = self.frames[step[:, None] + selected, actor[:, None]]
        history = history.masked_fill(
            (selected[None] < last_reset[:, None])[..., None], float("nan")
        )
        return torch.cat((history, self.successors[step, actor][:, None]), 1)


def generated_training_indices(rollout, scenes):
    """All eligible learner scenarios train; no simulator slots are reserved."""
    valid = ~scenes.terminated.flatten()
    if rollout.learner_mask is not None:
        valid &= rollout.learner_mask.flatten()
    return valid.nonzero().flatten()


class PolicyWindowHistory:
    def __init__(self, capacity, add_size, interval, phase_pairs=False):
        self.capacity, self.add_size, self.interval = capacity, add_size, interval
        self.windows = None
        self.size = self.cursor = self.iteration = 0
        self.phase_pairs = phase_pairs

    def sample(self, count):
        return self.windows[torch.randint(self.size, (count,), device=self.windows.device)]

    def add(self, scenes, indices):
        self.iteration += 1
        count = min(self.add_size, self.capacity, len(indices))
        if not count or (self.iteration - 1) % self.interval:
            return 0
        selected = indices[torch.randperm(len(indices), device=indices.device)[:count]]
        windows = scenes.transition_pairs(selected) if self.phase_pairs else scenes.transitions(selected)
        if self.windows is None:
            self.windows = windows.new_empty((self.capacity, *windows.shape[1:]))
        slots = (torch.arange(count, device=indices.device) + self.cursor) % self.capacity
        self.windows[slots] = windows
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        return count

    def state_dict(self):
        return {
            "windows": self.windows[: self.size].cpu() if self.windows is not None else None,
            "size": self.size,
            "cursor": self.cursor,
            "iteration": self.iteration,
            "phase_pairs": self.phase_pairs,
        }

    def load_state_dict(self, state, device):
        if not self.capacity:
            self.windows = None
            self.size = self.cursor = 0
            self.iteration = state["iteration"]
            return
        if state.get("phase_pairs", False) != self.phase_pairs:
            warnings.warn("Jump-phase sampling changed: discarding incompatible generated history; "
                          "online/target/optimizer weights are retained.", stacklevel=2)
            self.windows = None
            self.size = self.cursor = 0
            self.iteration = state["iteration"]
            return
        self.size, self.cursor, self.iteration = state["size"], state["cursor"], state["iteration"]
        if self.size:
            saved = state["windows"].to(device)
            self.windows = saved.new_empty((self.capacity, *saved.shape[1:]))
            self.windows[: self.size].copy_(saved)


def discriminator_loss(discriminator, windows, labels, amp):
    loss = windows.new_zeros(())
    objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits
    for head in discriminator.heads:
        logits = head(windows).float()
        loss = loss + objective(logits, labels) * (1 if amp else 2) / len(discriminator.heads)
    return loss


def phase_discriminator_loss(discriminator, windows, paired, labels, amp):
    # Small batches are launch-bound: evaluate both real phase views together.
    # Large batches keep the lower-peak-memory head-at-a-time path below.
    if len(windows) <= 16_384:
        logits, other = discriminator(torch.cat((windows, paired))).float().chunk(2)
        objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits
        classification = objective(logits, labels[:, None].expand_as(logits)) * (1 if amp else 2)
        return classification + (logits - other).square().mean()
    loss = windows.new_zeros(())
    objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits
    for head in discriminator.heads:
        logits = head(windows).float()
        other = head(paired).float()
        classification = objective(logits, labels) * (1 if amp else 2)
        loss = loss + (classification + (logits - other).square().mean()) / len(discriminator.heads)
    return loss


def compile_discriminator_functions(discriminator, target, args):
    """Compile callables, never wrap checkpoint-owned model state/parameters."""
    loss = phase_discriminator_loss if getattr(args, "experimental_jump_phase", False) else discriminator_loss
    if getattr(args, "imitation_algorithm", "gaifo") == "difo":
        from training.diffusion import diffusion_loss, phase_diffusion_loss

        loss = phase_diffusion_loss if args.experimental_jump_phase else diffusion_loss
    reward_model = target if target is not None else discriminator
    # Deep-copied/restored GRU weights otherwise trigger a cuDNN compaction
    # warning and may be repacked on every inference. Do this once at startup.
    for model in (discriminator, target):
        if model is not None:
            for module in model.modules():
                if isinstance(module, nn.RNNBase):
                    module.flatten_parameters()
    if args.compile and args.gru and not args.mlp_discriminator:
        # Dynamo deliberately does not trace nn.GRU. Keep this supported
        # architecture eager rather than generating graph-break warnings/errors.
        print("Discriminator GRU uses eager execution (torch.compile does not trace nn.GRU).", flush=True)
        return loss, reward_model
    if args.compile:
        from training.setup import compilation_mode, configure_compilation_warnings

        configure_compilation_warnings()
        if not args.mlp_discriminator or getattr(args, "discriminator_hierarchical", False):
            # The eval-only fused TransformerEncoderLayer operator has no fake
            # tensor implementation in this PyTorch build. Compile its regular
            # attention/FFN path instead (same weights, masks and computation).
            torch.backends.mha.set_fastpath_enabled(False)
        mode = compilation_mode(args)
        if args.compile_discriminator_loss:
            # Gradient accumulation spans several microbatches. CUDA graph trees
            # can overwrite saved backward tensors across those invocations;
            # use fused Inductor backward without graph replay for this callable.
            loss = torch.compile(loss, fullgraph=True,
                                 mode=compilation_mode(args, cuda_graphs=False), dynamic=True)
        reward_model = torch.compile(reward_model, fullgraph=True, mode=mode, dynamic=True)
    return loss, reward_model


def train_discriminator_batch(discriminator, optimizer, generated, expert, args, target,
                              loss_function=discriminator_loss):
    count = len(generated)
    phase = getattr(args, "experimental_jump_phase", False)
    if phase:
        if loss_function is discriminator_loss:
            loss_function = phase_discriminator_loss
        if generated.ndim != 4 or generated.shape != expert.shape or generated.shape[1] != 2:
            raise ValueError("experimental discriminator requires matching real phase pairs")
        padding = torch.isnan(generated[..., 0]).any(1) | torch.isnan(expert[..., 0]).any(1)
        windows = torch.cat((generated, expert)).masked_fill(
            torch.cat((padding, padding))[:, None, :, None], float("nan")
        )
        first = windows[:, 0]
        noisy = add_motion_noise(first, args.discriminator_motion_noise_scale)
        noise = torch.nan_to_num(noisy - first)
        windows = windows + noise[:, None]
    else:
        windows = add_motion_noise(
            balanced_window_batch(generated, expert), args.discriminator_motion_noise_scale
        )
    if getattr(args, "discriminator_mirror", False):
        windows = mirror_scene_windows(windows)
    labels = torch.cat((windows.new_ones(count), windows.new_zeros(count)))
    if args.amp:
        labels = 2 * labels - 1
    optimizer.zero_grad(set_to_none=True)
    total = windows.new_zeros(())
    for start in range(0, count, args.discriminator_update_microbatch_size):
        stop = min(start + args.discriminator_update_microbatch_size, count)
        indices = torch.cat(
            (
                torch.arange(start, stop, device=windows.device),
                torch.arange(count + start, count + stop, device=windows.device),
            )
        )
        with autocast(windows.device, args.bf16):
            torch.compiler.cudagraph_mark_step_begin()
            if phase:
                loss = loss_function(discriminator, windows[indices, 0], windows[indices, 1], labels[indices], args.amp)
            else:
                loss = loss_function(discriminator, windows[indices], labels[indices], args.amp)
            loss = loss * ((stop - start) / count)
        loss.backward()
        total += loss.detach()
    norm = nn.utils.clip_grad_norm_(
        discriminator.parameters(), args.max_grad_norm, error_if_nonfinite=False
    )
    require_finite(torch.stack((total, norm)), "discriminator loss/gradients")
    optimizer.step()
    if target is not None:
        polyak_update(target, discriminator, args.target_discriminator_tau)
    return total


def discriminator_batches(indices, args):
    for _ in range(args.discriminator_updates):
        order = indices[torch.randperm(len(indices), device=indices.device)]
        yield from order.split(args.discriminator_batch_size)


def update_discriminator(
    discriminator, optimizer, experts, scenes, train, args, history, target,
    loss_function=discriminator_loss,
):
    total = torch.zeros((), device=train.device)
    updates = 0

    for indices in discriminator_batches(train, args):
        if not len(indices):
            continue

        phase = getattr(args, "experimental_jump_phase", False)
        generated = scenes.transition_pairs(indices) if phase else scenes.transitions(indices)
        historical = (
            int(len(indices) * args.discriminator_history_mix_fraction) if history.size else 0
        )

        if historical:
            generated = torch.cat((generated[historical:], history.sample(historical)))

        total += train_discriminator_batch(
            discriminator, optimizer, generated,
            experts.sample_pairs(len(indices)) if phase else experts.sample(len(indices)), args, target,
            loss_function,
        )
        updates += 1

    added = history.add(scenes, train)
    metrics = {
        "discriminator/train_loss": (total / max(updates, 1)).item(),
        "discriminator/minibatches": updates,
        "discriminator/history_windows": history.size,
        "discriminator/history_added": added,
        "discriminator/motion_noise_scale": args.discriminator_motion_noise_scale,
        "discriminator/scene_mirroring": float(getattr(args, "discriminator_mirror", False)),
        "discriminator/flags": float(getattr(args, "discriminator_flags", True)),
        "discriminator/boost_amount": float(getattr(args, "discriminator_boost", True)),
        "discriminator/boost_rounding": float(getattr(args, "discriminator_round_boost", False)),
        "discriminator/endpoint_scalars": float(getattr(args, "discriminator_endpoint_scalars", False)),
        "discriminator/experimental_jump_phase": float(getattr(args, "experimental_jump_phase", False)),
        "discriminator/phase_consistency_weight": float(getattr(args, "experimental_jump_phase", False)),
    }
    metrics.update(experts.diagnostic_metrics())
    if getattr(args, "discriminator_hierarchical", False):
        metrics["discriminator/hierarchical"] = 1.0
        for name, length in discriminator_scales(args):
            metrics[f"discriminator/{name}_span_ticks"] = (
                (length - 1) * args.frameskip * args.discriminator_decision_stride
            )
    if getattr(args, "imitation_algorithm", "gaifo") == "difo":
        metrics.update({
            f"difo/{name.removeprefix('diffusion_')}": float(getattr(args, name))
            for name in (
                "diffusion_steps", "diffusion_mse_weight", "diffusion_bce_weight",
                "diffusion_logit_scale", "diffusion_reward_samples",
                "diffusion_reward_t_min", "diffusion_reward_t_max",
            )
        })
    if target is not None:
        metrics.update(
            {
                "discriminator/target_updates": updates,
                "discriminator/target_tau": args.target_discriminator_tau,
            }
        )
    return metrics


class EpisodeImitationStatistics:
    """Undiscounted imitation-only totals for fully observed completed episodes."""

    def __init__(self, agents, device, frameskip=1):
        self.frameskip = frameskip
        self.sums = torch.zeros(agents, device=device)
        self.lengths = torch.zeros(agents, dtype=torch.long, device=device)

    @torch.no_grad()
    def observe(self, rewards, dones, ticks=None):
        if len(rewards) < 3:
            totals = rewards.new_zeros(4)
            for index, (reward, done) in enumerate(zip(rewards, dones)):
                self.sums += reward
                self.lengths += self.frameskip if ticks is None else ticks[index]
                totals[0] += self.sums.abs().masked_fill(~done, 0).sum()
                totals[1] += self.sums.masked_fill(~done, 0).sum()
                totals[2] += self.lengths.masked_fill(~done, 0).sum()
                totals[3] += done.sum()
                self.sums.masked_fill_(done, 0)
                self.lengths.masked_fill_(done, 0)
        else:
            # Segmented prefix sums replace a Python loop and hundreds of GPU
            # launches. A prior episode ends immediately before its next frame.
            times = torch.arange(len(rewards), device=rewards.device)[:, None]
            ends = torch.where(dones, times, -1).cummax(0).values
            previous = torch.cat((torch.full_like(ends[:1], -1), ends[:-1]), 0)
            indices = previous + 1
            cumulative = rewards.cumsum(0)
            durations = ((times + 1) * self.frameskip).expand_as(dones) if ticks is None else ticks.cumsum(0)
            baseline = torch.cat((torch.zeros_like(cumulative[:1]), cumulative), 0).gather(0, indices)
            time_baseline = torch.cat((torch.zeros_like(durations[:1]), durations), 0).gather(0, indices)
            sums = cumulative - baseline + self.sums[None] * (previous < 0)
            lengths = durations - time_baseline + self.lengths[None] * (previous < 0)
            totals = torch.stack((
                sums.abs().masked_fill(~dones, 0).sum(), sums.masked_fill(~dones, 0).sum(),
                lengths.masked_fill(~dones, 0).sum(), dones.sum(),
            ))
            self.sums.copy_(sums[-1].masked_fill(dones[-1], 0))
            self.lengths.copy_(lengths[-1].masked_fill(dones[-1], 0))

        absolute, signed, length, count = totals.tolist()
        if not count:
            return {}

        return {
            "imitation/episode_abs_sum_mean": absolute / count,
            "imitation/episode_sum_mean": signed / count,
            "imitation/episode_length_ticks_mean": length / count,
            "imitation/episode_count": count,
        }


@torch.no_grad()
def imitation_rewards(discriminator, target, scenes, rollout, args, episodes=None, reward_model=None):
    if reward_model is None:
        reward_model = target if target is not None else discriminator
    total = rollout.rewards.numel()
    rewards = rollout.rewards.flatten()
    imitation_values = torch.empty_like(rewards)
    rewards.mul_(args.gameplay_reward_weight)
    reward_sum = rewards.new_zeros(())
    hierarchical = getattr(args, "discriminator_hierarchical", False)
    scale_sums = rewards.new_zeros(3) if hierarchical else None
    forward_kl = getattr(args, "discriminator_forward_kl_reward", False)
    capped = rewards.new_zeros(()) if forward_kl else None
    live_heads = rewards.new_zeros(()) if forward_kl else None
    for start in range(0, total, args.discriminator_inference_microbatch_size):
        indices = torch.arange(
            start,
            min(start + args.discriminator_inference_microbatch_size, total),
            device=rewards.device,
        )
        windows = scenes.transitions(indices)
        with autocast(rewards.device, args.bf16):
            torch.compiler.cudagraph_mark_step_begin()
            logits = reward_model(windows)
            values = discriminator_rewards(
                logits.unflatten(-1, (3, -1)) if hierarchical else logits,
                amp=args.amp,
                discriminator_log_reward=args.discriminator_log_reward,
                discriminator_forward_kl_reward=getattr(args, "discriminator_forward_kl_reward", False),
                discriminator_ratio_cap=getattr(args, "discriminator_ratio_cap", 10.0),
            )
            imitation = values.mean(-1) if hierarchical else values
        live = ~scenes.terminated.flatten()[indices]
        if forward_kl:
            capped += ((-logits.float() > math.log(args.discriminator_ratio_cap)) & live[:, None]).sum()
            live_heads += live.sum() * logits.shape[-1]
        if hierarchical:
            scale_sums += values.masked_fill(~live[:, None], 0).sum(0)
        imitation = imitation.masked_fill(~live, 0)
        rewards[indices] += imitation
        imitation_values[indices] = imitation
        reward_sum += imitation.sum()
    require_finite(rewards, "GAIfO rewards")
    metrics = {
        "imitation/reward_per_tick": (reward_sum / (total * args.frameskip)).item(),
    }
    if forward_kl:
        metrics["imitation/forward_kl_reward"] = 1.0
        metrics["imitation/forward_kl_baseline"] = 1.0
        metrics["imitation/ratio_cap"] = args.discriminator_ratio_cap
        metrics["imitation/ratio_capped_fraction"] = (capped / live_heads.clamp_min(1)).item()
    if hierarchical:
        metrics.update({
            f"imitation/{name}_reward_per_tick": value
            for (name, _), value in zip(
                discriminator_scales(args), (scale_sums / (total * args.frameskip)).tolist(),
            )
        })

    if episodes is not None:
        metrics.update(episodes.observe(
            imitation_values.view_as(rollout.rewards), rollout.dones,
            ticks=getattr(scenes, "transition_ticks", None),
        ))

    goal_weight = getattr(args, "goal_reward_weight", None) or 0.0
    goal_rewards = scenes.score_deltas * goal_weight
    rollout.rewards.add_(goal_rewards)
    require_finite(rollout.rewards, "GAIfO rewards including goals")
    metrics["goal_reward/weight"] = goal_weight
    metrics["goal_reward/abs_per_tick"] = goal_rewards.abs().mean().item() / args.frameskip

    return metrics
