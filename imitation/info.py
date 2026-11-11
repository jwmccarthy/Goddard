"""State-only categorical InfoGAIL for GAIfO and DAC-style PPO."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from imitation.models import WindowHead, relative_window
from training.tensors import autocast, require_finite


class TransitionPosterior(nn.Module):
    """Q(z | causal state window); never receives actions or the policy code."""

    def __init__(self, args, scene_size):
        super().__init__()
        self.relative = getattr(args, "discriminator_relative", False)
        self.head = WindowHead(scene_size, args.discriminator_hidden, args.frame_embedding,
                               args.discriminator_window_length, output_size=args.info_codes, mlp=True)

    def forward(self, frames):
        padding = torch.isnan(frames[..., 0])
        if self.relative:
            frames = relative_window(frames)
        return self.head(torch.nan_to_num(frames), padding).float()


class InfoGAIL:

    def __init__(self, args, scene_size, device):
        self.args = args
        self.posterior = TransitionPosterior(args, scene_size).to(device)
        self.optimizer = torch.optim.Adam(self.posterior.parameters(), lr=args.info_learning_rate,
                                          fused=torch.device(device).type == "cuda")

    def checkpoint(self):
        return {"info_posterior": self.posterior.state_dict(), "info_optimizer": self.optimizer.state_dict()}

    def load_checkpoint(self, payload):
        self.posterior.load_state_dict(payload["info_posterior"])
        self.optimizer.load_state_dict(payload["info_optimizer"])
        self.optimizer.param_groups[0]["lr"] = self.args.info_learning_rate

    def batches(self, scenes, indices):
        # DAC's virtual loops and goal-to-absorbing transitions are NOT behaviors.
        indices = indices[indices < scenes.dones.numel()]
        indices = indices[~scenes.terminated.flatten()[indices]]
        for batch in indices.split(self.args.info_batch_size):
            frames = scenes.transitions(batch)[..., :scenes.scene_size]
            valid = (~torch.isnan(frames[..., 0])).sum(dim=1) >= 2
            if valid.any().item():
                yield batch[valid], frames[valid]

    @torch.no_grad()
    def score(self, scenes, rollout, labels):
        rewards = rollout.rewards.flatten()
        indices = torch.arange(len(rewards), device=rewards.device)
        log_q_sum, squared_sum = rewards.new_zeros(()), rewards.new_zeros(())
        count = 0
        for batch, frames in self.batches(scenes, indices):
            with autocast(rewards.device, self.args.bf16):
                logits = self.posterior(frames)
            require_finite(logits, "info posterior logits")
            log_q = logits.log_softmax(-1).gather(-1, labels[batch, None]).squeeze(-1)
            rewards[batch] += self.args.info_weight * log_q
            log_q_sum += log_q.sum()
            squared_sum += log_q.square().sum()
            count += len(batch)
        mean_log_q = (log_q_sum / max(count, 1)).item()
        return {
            "info/log_q": mean_log_q,
            "info/mi_lower_bound": math.log(self.args.info_codes) + mean_log_q if count else 0.0,
            "info/reward": self.args.info_weight * (log_q_sum / len(rewards)).item(),
            "info/reward_rms": self.args.info_weight * (squared_sum / len(rewards)).sqrt().item(),
            "info/valid_fraction": count / len(rewards),
        }

    @torch.no_grad()
    def evaluate(self, scenes, indices, labels):
        device = scenes.scenes.device
        statistics = torch.zeros(2, device=device)
        count = 0
        for batch, frames in self.batches(scenes, indices):
            with autocast(device, self.args.bf16):
                logits = self.posterior(frames)
            require_finite(logits, "info heldout logits")
            statistics[0] += (logits.argmax(-1) == labels[batch]).sum()
            statistics[1] += logits.log_softmax(-1).gather(-1, labels[batch, None]).sum()
            count += len(batch)
        if not count:
            return {"info/heldout_windows": 0}
        accuracy, log_q = (statistics / count).cpu().tolist()
        return {"info/heldout_windows": count, "info/heldout_accuracy": accuracy,
                "info/heldout_log_q": log_q, "info/heldout_mi_lower_bound": math.log(self.args.info_codes) + log_q}

    def update(self, scenes, rollout, train_indices, heldout_indices):
        labels = rollout.observations[..., -self.args.info_codes:].argmax(-1).flatten()
        # Freeze Q for reward scoring BEFORE fitting this rollout's labels. This
        # avoids an immediate bonus for memorizing the very samples being scored.
        metrics = self.score(scenes, rollout, labels)
        loss_sum = rollout.rewards.new_zeros(())
        count = 0
        for _ in range(self.args.info_epochs):
            permutation = torch.randperm(len(train_indices), device=train_indices.device)
            for batch, frames in self.batches(scenes, train_indices[permutation]):
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(frames.device, self.args.bf16):
                    logits = self.posterior(frames)
                    loss = F.cross_entropy(logits, labels[batch])
                require_finite(loss, "info posterior loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.posterior.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                loss_sum += loss.detach() * len(batch)
                count += len(batch)
        metrics["info/posterior_loss"] = (loss_sum / max(count, 1)).item()
        metrics["info/train_windows"] = count
        metrics.update(self.evaluate(scenes, heldout_indices, labels))
        probabilities = torch.bincount(labels, minlength=self.args.info_codes).float() / len(labels)
        metrics["info/code_entropy"] = -(probabilities * probabilities.clamp_min(1e-8).log()).sum().item()
        metrics["info/codes_used"] = (probabilities > 0).sum().item()
        require_finite(rollout.rewards, "info-augmented rewards")
        return metrics
