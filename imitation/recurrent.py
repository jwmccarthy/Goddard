from dataclasses import dataclass

import torch
from torch import nn

from imitation import features
from imitation.rewards import discriminator_rewards
from imitation.update import gradient_penalty_weights
from training.tensors import autocast


@dataclass
class RecurrentBatch:

    frames: torch.Tensor
    hidden: torch.Tensor
    reset: torch.Tensor
    valid: torch.Tensor

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, indices):
        return RecurrentBatch(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))

    @classmethod
    def concatenate(cls, batches):
        return cls(*(torch.cat([getattr(batch, name) for batch in batches]) for name in cls.__dataclass_fields__))

    def detach(self):
        return RecurrentBatch(*(getattr(self, name).detach() for name in self.__dataclass_fields__))


def zero_hidden(model, count, device):
    head = model.car_head if model.factorized else model.head
    return torch.zeros(count, 3 if model.factorized else 1, head.gru.hidden_size, device=device)


def forward_batch(model, batch):
    scores, states = model.recurrent(batch.frames, batch.hidden, batch.reset)
    return scores.float().unsqueeze(-1) if not model.factorized else scores.float(), states


class ExpertStreams:

    def __init__(self, experts, model, count, length, heldout=False):
        self.frames = experts.frames
        self.episodes = experts.heldout_episodes if heldout else experts.train_episodes
        self.generator = experts.heldout_generator if heldout else experts.train_generator
        self.length = length
        self.hidden = zero_hidden(model, count, self.frames.device)
        self.position = torch.zeros(count, dtype=torch.long, device=self.frames.device)
        self.end = torch.zeros_like(self.position)
        self.age = torch.zeros_like(self.position)

    def sample(self, count):
        position, end, age = self.position[:count], self.end[:count], self.age[:count]
        indices, resets, valid = [], [], []

        for _ in range(self.length):
            reset = position >= end
            choices = torch.randint(len(self.episodes), (count,), device=position.device, generator=self.generator)
            chosen = self.episodes[choices]
            position.copy_(torch.where(reset, chosen[:, 0], position))
            end.copy_(torch.where(reset, chosen[:, 1], end))
            age.masked_fill_(reset, 0)
            indices.append(position.clone())
            resets.append(reset)
            valid.append(age >= features.DISCRIMINATOR_WARMUP)
            position.add_(1)
            age.add_(1)

        return RecurrentBatch(
            self.frames[torch.stack(indices, 1)], self.hidden[:count].clone(),
            torch.stack(resets, 1), torch.stack(valid, 1)
        )

    @torch.no_grad()
    def advance(self, model, batch, args):
        microbatch = max(1, args.discriminator_inference_microbatch_size // self.length)
        for start in range(0, len(batch), microbatch):
            stop = min(start + microbatch, len(batch))
            with autocast(batch.frames.device, args.bf16):
                _, states = forward_batch(model, batch[start:stop])
            self.hidden[start:stop].copy_(states[:, -1])


class RecurrentHistory:

    def __init__(self, capacity, seed, device):
        self.capacity = capacity
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.data = None
        self.size = self.start = 0

    def add(self, batch, count):
        count = min(count, len(batch), self.capacity)
        if not count:
            return
        order = torch.randperm(len(batch), device=batch.frames.device, generator=self.generator)[:count]
        selected = batch[order].detach()
        if self.data is None:
            self.data = RecurrentBatch(*(
                value.new_empty((self.capacity, *value.shape[1:]))
                for value in vars(selected).values()
            ))
        indices = (torch.arange(count, device=batch.frames.device) + self.start) % self.capacity
        for name in vars(selected):
            getattr(self.data, name)[indices] = getattr(selected, name)
        self.start = (self.start + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count):
        indices = torch.randint(self.size, (count,), device=self.data.frames.device, generator=self.generator)
        return self.data[indices]


class EpisodeRollout:

    def __init__(self, scenes, model, args):
        self.scenes = scenes
        self.hidden = zero_hidden(model, scenes.scenes.shape[1], scenes.scenes.device)
        self.length = args.trajectory_length
        self.batch = None

    @torch.no_grad()
    def prepare(self, model, args):
        scenes = self.scenes
        frames = scenes.scenes.transpose(0, 1)
        reset = (scenes.ages == 0).T
        valid = (scenes.ages >= features.DISCRIMINATOR_WARMUP).T
        batches = []
        hidden = self.hidden
        microbatch = max(1, args.discriminator_inference_microbatch_size // self.length)
        for start in range(0, frames.shape[1], self.length):
            stop = min(start + self.length, frames.shape[1])
            batch = RecurrentBatch(frames[:, start:stop], hidden.clone(), reset[:, start:stop], valid[:, start:stop])
            # Keep a fixed chunk shape, masking padding out of every loss and metric.
            if stop - start < self.length:
                pad = self.length - (stop - start)
                batch.frames = torch.cat((batch.frames, batch.frames[:, -1:].expand(-1, pad, -1)), 1)
                batch.reset = torch.cat((batch.reset, torch.zeros_like(batch.reset[:, :1]).expand(-1, pad)), 1)
                batch.valid = torch.cat((batch.valid, torch.zeros_like(batch.valid[:, :1]).expand(-1, pad)), 1)
            parts = []
            for selected in range(0, len(batch), microbatch):
                with autocast(frames.device, args.bf16):
                    _, states = forward_batch(model, batch[selected:selected + microbatch])
                parts.append(states[:, stop - start - 1])
            hidden = torch.cat(parts).detach()
            batches.append(batch)
        self.batch = RecurrentBatch.concatenate(batches)

    @torch.no_grad()
    def rewards(self, model, args):
        scenes = self.scenes
        agents = scenes.scenes.shape[1]
        terminal = scenes.dones.T.clone()
        terminal[:, -1] = True
        scores, terminal_hidden = [], []
        microbatch = max(1, args.discriminator_inference_microbatch_size // scenes.scenes.shape[0])
        for start in range(0, agents, microbatch):
            stop = min(start + microbatch, agents)
            batch = RecurrentBatch(scenes.scenes[:, start:stop].transpose(0, 1), self.hidden[start:stop],
                                   (scenes.ages[:, start:stop] == 0).T, (scenes.ages[:, start:stop] >= features.DISCRIMINATOR_WARMUP).T)
            with autocast(scenes.scenes.device, args.bf16):
                values, trace = forward_batch(model, batch)
            scores.append(values)
            a, t = terminal[start:stop].nonzero(as_tuple=True)
            terminal_hidden.append(trace[a, t])
            self.hidden[start:stop].copy_(trace[:, -1])
        scores, terminal_hidden = torch.cat(scores), torch.cat(terminal_hidden)
        # A terminal observation is not the reset observation of the next episode.
        actor, step = terminal.nonzero(as_tuple=True)
        successor = torch.empty((len(actor), scores.shape[-1]), device=scores.device)
        for start in range(0, len(actor), args.discriminator_inference_microbatch_size):
            stop = start + args.discriminator_inference_microbatch_size
            a, t = actor[start:stop], step[start:stop]
            frames = scenes.successors[t, a, None]
            batch = RecurrentBatch(frames, terminal_hidden[start:stop], torch.zeros((len(a), 1), dtype=torch.bool, device=scores.device),
                                   torch.ones((len(a), 1), dtype=torch.bool, device=scores.device))
            with autocast(scores.device, args.bf16):
                value, _ = forward_batch(model, batch)
            successor[start:stop] = value[:, 0]
        next_scores = torch.cat((scores[:, 1:], scores[:, -1:]), 1)
        next_scores[actor, step] = successor
        near = features.ball_distances(scenes.scenes.transpose(0, 1)) <= args.near_distance
        next_near = features.ball_distances(scenes.successors.transpose(0, 1)) <= args.near_distance
        pairs = torch.stack((scores, next_scores), 2).flatten(0, 1)
        gates = torch.stack((near, next_near), 2).flatten(0, 1)
        values = discriminator_rewards(pairs, gates if args.differential else gates[:, -1], args)
        values = values.view(agents, -1).T
        values = values.masked_fill(scenes.ages < features.DISCRIMINATOR_WARMUP, 0)
        # Cache BEFORE the next rollout's initial observation; re-score that boundary
        # with updated weights instead of reusing a stale scalar potential.
        return values


def recurrent_loss(model, batch, labels, near, weights, class_weights):
    scores, _ = forward_batch(model, batch)
    errors = 2 * (scores - (labels[:, None, None] * 2 - 1)).square()
    metrics = {}
    if model.factorized:
        head_losses = (errors[..., :2] * weights).sum((0, 1))
        global_loss = (errors[..., 2] * class_weights).sum()
        loss = features.SPECIALIST_WEIGHT * head_losses.sum() + features.GLOBAL_WEIGHT * global_loss
        selected = (features.SPECIALIST_WEIGHT * torch.where(near, scores[..., 1], scores[..., 0])
                    + features.GLOBAL_WEIGHT * scores[..., 2])
        metrics.update(far_loss=head_losses[0].detach(), near_loss=head_losses[1].detach(),
                       global_loss=global_loss.detach())
    else:
        loss = (errors[..., 0] * class_weights).sum()
        selected = scores[..., 0]
    for label, name in ((0, "agent"), (1, "expert")):
        normalized = 2 * class_weights * (labels[:, None] == label)
        metrics[f"{name}_score"] = (selected.detach() * normalized).sum()
        metrics[f"{name}_accuracy"] = (((selected.detach() > 0) == bool(label)) * normalized).sum()
    return loss, metrics


def recurrent_penalty(model, batch, weights, steps):
    with torch.autocast(batch.frames.device.type, enabled=False), torch.backends.cudnn.flags(enabled=False):
        penalty = batch.frames.new_zeros((), dtype=torch.float32)
        for head in range(weights.shape[1]):
            active = weights[:, head] != 0
            if not active.any():
                continue
            selected = batch[active]
            frames = selected.frames.detach().float().requires_grad_(True)
            hidden = selected.hidden.detach().float()
            if model.factorized:
                scores, _ = model.recurrent_head(frames, hidden[:, head], selected.reset, head)
            else:
                scores, _ = model.recurrent(frames, hidden, selected.reset)
            values = scores[torch.arange(len(frames), device=frames.device), steps[active]]
            gradient, = torch.autograd.grad(values.sum(), frames, create_graph=True)
            penalty = penalty + (gradient.flatten(1).square().sum(1) * weights[active, head]).sum() / len(batch)
    return penalty


def train_recurrent_batch(model, optimizer, generated, expert, args):
    batch = RecurrentBatch.concatenate((generated, expert))
    count = len(generated)
    labels = torch.cat((batch.frames.new_zeros(count), batch.frames.new_ones(count)))
    near = features.ball_distances(batch.frames) <= args.near_distance
    valid = batch.valid
    timestep_labels = labels[:, None].expand_as(valid)
    class_weights = batch.frames.new_zeros(valid.shape)
    for label in (0, 1):
        mask = valid & (timestep_labels == label)
        class_weights[mask] = 1 / (2 * mask.sum().clamp_min(1))
    weights = batch.frames.new_zeros((*valid.shape, 2))
    if model.factorized and valid.any():
        weights[valid] = features.balanced_proximity_weights(near[valid], timestep_labels[valid]) / valid.sum()
    positions = expert.valid.nonzero()
    penalty_batch = penalty_weights = penalty_steps = None
    if args.discriminator_gradient_penalty and len(positions):
        n = min(count, args.discriminator_gradient_penalty_samples or count)
        choices = torch.randint(len(positions), (n,), device=batch.frames.device)
        rows, penalty_steps = positions[choices].unbind(1)
        penalty_batch = expert[rows]
        penalty_weights = gradient_penalty_weights(near[count:][expert.valid], model.factorized)[choices]

    optimizer.zero_grad(set_to_none=True)
    loss_total = batch.frames.new_zeros(())
    penalty_total = batch.frames.new_zeros(())
    metrics = {}
    microbatch = max(1, args.discriminator_update_microbatch_size // expert.frames.shape[1])
    for start in range(0, count, microbatch):
        stop = min(start + microbatch, count)
        indices = torch.cat((torch.arange(start, stop, device=batch.frames.device),
                             torch.arange(count + start, count + stop, device=batch.frames.device)))
        with autocast(batch.frames.device, args.bf16):
            loss, chunk_metrics = recurrent_loss(model, batch[indices], labels[indices], near[indices], weights[indices], class_weights[indices])
        loss.backward()
        loss_total += loss.detach()
        for name, value in chunk_metrics.items():
            metrics[name] = metrics.get(name, 0.0) + value

    if penalty_batch is not None:
        for start in range(0, len(penalty_batch), microbatch):
            stop = min(start + microbatch, len(penalty_batch))
            penalty = recurrent_penalty(model, penalty_batch[start:stop], penalty_weights[start:stop], penalty_steps[start:stop])
            fraction = (stop - start) / len(penalty_batch)
            (0.5 * args.discriminator_gradient_penalty * penalty * fraction).backward()
            penalty_total += penalty.detach() * fraction
    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    metrics.update(classification_loss=loss_total, gradient_penalty=penalty_total,
                   loss=loss_total + 0.5 * args.discriminator_gradient_penalty * penalty_total,
                   near_agent_fraction=(near[:count] & valid[:count]).sum() / valid[:count].sum().clamp_min(1),
                   near_expert_fraction=(near[count:] & valid[count:]).sum() / valid[count:].sum().clamp_min(1))
    return {name: value.item() for name, value in metrics.items()}


@torch.no_grad()
def evaluate_recurrent(model, generated, expert, args):
    batch = RecurrentBatch.concatenate((generated, expert))
    count = len(generated)
    parts = []
    microbatch = max(1, args.discriminator_inference_microbatch_size // batch.frames.shape[1])
    for start in range(0, len(batch), microbatch):
        with autocast(batch.frames.device, args.bf16):
            values, _ = forward_batch(model, batch[start:start + microbatch])
        parts.append(values)
    scores = torch.cat(parts)
    near = features.ball_distances(batch.frames) <= args.near_distance
    metrics, accuracies = {}, []
    labels = torch.arange(len(batch), device=batch.frames.device) >= count
    correct = (scores > 0) == labels[:, None, None]
    masks = (~near, near, torch.ones_like(near)) if model.factorized else (torch.ones_like(near),)
    names = ("far", "near", "global") if model.factorized else ("global",)
    for head, (name, mask) in enumerate(zip(names, masks)):
        mask = mask & batch.valid
        gen, exp = mask[:count], mask[count:]
        if gen.any() and exp.any():
            accuracy = 0.5 * (correct[:count, :, head][gen].float().mean() + correct[count:, :, head][exp].float().mean())
            accuracies.append(accuracy.item())
            metrics[f"discriminator/{name}_heldout_accuracy"] = accuracy.item()
        else:
            metrics[f"discriminator/{name}_heldout_accuracy"] = 0.0
    metrics["discriminator/heldout_accuracy"] = min(accuracies) if accuracies else 0.0
    selected = (features.SPECIALIST_WEIGHT * torch.where(near, scores[..., 1], scores[..., 0])
                + features.GLOBAL_WEIGHT * scores[..., 2]) if model.factorized else scores[..., 0]
    for name, values, mask, target in (("agent", selected[:count], batch.valid[:count], False),
                                       ("expert", selected[count:], batch.valid[count:], True)):
        if mask.any():
            metrics[f"discriminator/{name}_score"] = values[mask].mean().item()
            metrics[f"discriminator/{name}_accuracy"] = ((values[mask] > 0) == target).float().mean().item()
    return metrics


def update_recurrent(model, optimizer, episode, streams, validation_streams, history, heldout_sim, n_cars, args, scheduled):
    batch = episode.batch
    agents = episode.scenes.scenes.shape[1]
    heldout_actor = (heldout_sim[:, None].expand(-1, n_cars).flatten() if heldout_sim is not None
                     else torch.zeros(agents, dtype=torch.bool, device=batch.frames.device))
    heldout = heldout_actor.repeat(len(batch) // agents)
    train_indices = (~heldout & batch.valid.any(1)).nonzero().flatten()
    validation_indices = (heldout & batch.valid.any(1)).nonzero().flatten()
    sequence_count = max(1, args.discriminator_batch_size // episode.length)
    metrics = {"discriminator/updated": 0.0, "discriminator/minibatches": 0.0,
               "discriminator/scheduled": float(scheduled)}
    validation = None
    if validation_streams is not None and len(validation_indices):
        count = min(len(validation_indices), max(1, args.discriminator_heldout_size // episode.length))
        choices = torch.randperm(len(validation_indices), device=batch.frames.device)[:count]
        validation = (batch[validation_indices[choices]], validation_streams.sample(count))
    totals, updates = {}, 0
    if scheduled:
        for _ in range(args.discriminator_updates):
            # Shuffle streams, not timesteps: each chunk is chronological and its
            # detached seed was obtained by walking the preceding rollout chunks.
            order = train_indices[torch.randperm(len(train_indices), device=batch.frames.device)]
            for indices in order.split(sequence_count):
                count = len(indices)
                history_count = min(int(count * args.history_mix_fraction), history.size)
                generated = batch[indices[:count - history_count]]
                if history_count:
                    generated = RecurrentBatch.concatenate((generated, history.sample(history_count)))
                expert = streams.sample(count)
                result = train_recurrent_batch(model, optimizer, generated, expert, args)
                streams.advance(model, expert, args)
                for name, value in result.items():
                    totals[name] = totals.get(name, 0.0) + value
                updates += 1
    if validation is not None:
        metrics.update(evaluate_recurrent(model, *validation, args))
        validation_streams.advance(model, validation[1], args)
    metrics.update({f"discriminator/train_{name}": value / updates for name, value in totals.items()})
    metrics["discriminator/updated"] = float(updates > 0)
    metrics["discriminator/minibatches"] = float(updates)
    history.add(batch[train_indices], max(1, args.history_add_size // episode.length) if args.history_add_size else 0)
    return metrics
