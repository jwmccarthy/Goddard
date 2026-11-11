import torch
from torch import nn

from imitation.models import build_discriminator, relative_window
from imitation.rewards import discriminator_rewards
from imitation.rollout import TransitionRollout, split_generated
from training.tensors import require_finite


class AbsorbingDiscriminator(nn.Module):
    """Window discriminator with an explicit, learned absorbing-frame embedding."""

    def __init__(self, args, n_cars):
        super().__init__()
        self.base = build_discriminator(args, n_cars)
        self.factorized = self.base.factorized
        self.head_name = getattr(self.base, "head_name", "global")
        self.relative = self.base.relative
        self.supports_frame_cache = not self.relative
        count = 2 if self.factorized else 1
        self.absorbing = nn.ParameterList(
            nn.Parameter(torch.zeros(args.frame_embedding)) for _ in range(count)
        )

    def encode_frames(self, frames):
        indicator = torch.nan_to_num(frames[..., -1:])
        encoded = self.base.encode_frames(frames[..., :-1])
        return tuple((1 - indicator) * value + indicator * embedding
                     for value, embedding in zip(encoded, self.absorbing))

    def score_encoded(self, encoded, padding):
        return self.base.score_encoded(encoded, padding)

    def forward(self, windows):
        if self.relative:
            windows = torch.cat((relative_window(windows[..., :-1]), windows[..., -1:]), dim=-1)
        return self.score_encoded(self.encode_frames(windows), torch.isnan(windows[..., 0]))

    def head_score(self, windows, head):
        model = (self.base.ego_discriminator, self.base.global_discriminator)[head]
        if self.relative:
            windows = torch.cat((relative_window(windows[..., :-1]), windows[..., -1:]), dim=-1)
        frames = torch.nan_to_num(windows[..., :-1])
        indicator = torch.nan_to_num(windows[..., -1:])
        encoded = model.head.projection(model._encode(frames))
        encoded = (1 - indicator) * encoded + indicator * self.absorbing[head]
        return model.head.forward_encoded(encoded, torch.isnan(windows[..., 0]))


class AbsorbingRollout(TransitionRollout):
    """Append goal-to-absorbing transitions and virtual absorbing self-loops."""

    def all_frames(self):
        if self.cached_frames is None:
            frames = torch.cat((self.prefix, self.scenes, self.successors), dim=0)
            indicator = torch.zeros_like(frames[..., :1])
            indicator = indicator.masked_fill(torch.isnan(frames[..., :1]), float("nan"))
            frames = torch.cat((frames, indicator), dim=-1)
            successor = frames[-len(self.successors):]
            successor[self.terminated] = 0
            successor[..., -1][self.terminated] = 1
            self.cached_frames = frames.flatten(0, 1)
        return self.cached_frames

    def transitions(self, indices):
        total = self.scenes.shape[0] * self.scenes.shape[1]
        absorbing = indices >= total
        window_indices, padding = self.window_indices(indices.remainder(total))
        frames = self.all_frames()[window_indices].masked_fill(padding[..., None], float("nan"))
        loop = self.scenes.new_zeros((1, 1, self.scene_size + 1))
        loop[..., -1] = 1
        return torch.where(absorbing[:, None, None], loop, frames)


def split_absorbing_generated(scenes, heldout_sim, heldout_size, n_cars):
    train, heldout, heldout_sim = split_generated(scenes.dones, heldout_sim, heldout_size, n_cars)
    total = scenes.dones.numel()
    terminated = scenes.terminated.flatten()
    train = torch.cat((train, train[terminated[train]] + total))
    heldout = torch.cat((heldout, heldout[terminated[heldout]] + total))
    return train, heldout, heldout_sim


class AbsorbingValue(nn.Module):

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))


def update_absorbing_value(value, optimizer, reward, gamma, updates):
    require_finite(reward, "absorbing reward")
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        target = reward.detach() + gamma * value.value.detach()
        loss = (value.value - target).square()
        require_finite(loss, "absorbing value loss")
        loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    require_finite(value.value, "absorbing value")
    return loss.detach()


@torch.no_grad()
def absorbing_reward(discriminator, args, device, scene_size):
    window = torch.zeros(1, args.discriminator_window_length, scene_size + 1, device=device)
    window[..., -1] = 1
    logits = discriminator(window)
    return discriminator_rewards(
        logits, args.exp_log_odds_reward, args.multiplicative_reward, args.discriminator_log_reward,
        amp=getattr(args, "amp", False)
    ).squeeze(0)


@torch.no_grad()
def absorbing_diagnostics(discriminator, args, device, scene_size, reward, value,
                          scenes, train_indices, heldout_indices, experts):
    window = torch.zeros(1, args.discriminator_window_length, scene_size + 1, device=device)
    window[..., -1] = 1
    logits = discriminator(window).float().reshape(-1)
    require_finite(logits, "absorbing logits")
    fixed_point = reward / (1 - args.reward_gamma)
    metrics = {
        "absorbing/value_fixed_point": fixed_point.item(),
        "absorbing/value_error": (value.value - fixed_point).item(),
    }
    names = ("ego", "global") if discriminator.factorized else (getattr(discriminator, "head_name", "global"),)
    for name, logit in zip(names, logits):
        metrics[f"absorbing/{name}_logit"] = logit.item()
    total = scenes.dones.numel()
    for name, indices, kinds in (
        ("train", train_indices, experts.train_kinds),
        ("heldout", heldout_indices, experts.heldout_kinds),
    ):
        if indices.numel():
            metrics[f"absorbing/{name}_generated_pool_loop_fraction"] = (indices >= total).float().mean().item()
        if kinds.numel():
            metrics[f"absorbing/{name}_expert_pool_loop_fraction"] = (kinds == 2).float().mean().item()
    return metrics
