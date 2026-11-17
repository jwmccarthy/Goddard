"""Episode-persistent historical opponents; only live-policy cars train PPO."""
import copy

import torch


def validate_self_play_arguments(args):
    if args.opponent_pool_size < 0:
        raise ValueError("opponent pool size must be nonnegative")
    if args.opponent_snapshot_interval < 1:
        raise ValueError("opponent snapshot interval must be positive")
    if not 0 <= args.historical_opponent_fraction <= 1:
        raise ValueError("historical opponent fraction must be between 0 and 1")


class OpponentPool:
    def __init__(self, agent, args, device, step=0):
        validate_self_play_arguments(args)
        if not args.opponent_pool_size:
            raise ValueError("construct an opponent pool only when its size is positive")
        self.agent = agent
        self.capacity = args.opponent_pool_size
        self.interval = args.opponent_snapshot_interval
        self.fraction = args.historical_opponent_fraction
        self.team_size = args.team_size
        self.n_cars = 2 * self.team_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed + 9173)
        self.assignments = torch.full((args.n_sim,), -1, dtype=torch.long, device=self.device)
        self.learner_sides = torch.zeros(args.n_sim, dtype=torch.long, device=self.device)
        self.snapshots = {}
        self.published = []
        self.next_id = 0
        self.last_snapshot_step = step
        self.add_snapshot(step)

    def add_snapshot(self, step):
        actor = copy.deepcopy(self.agent.actor).eval().requires_grad_(False)
        identity = self.next_id
        self.next_id += 1
        self.snapshots[identity] = {"actor": actor, "step": step}
        self.published.append(identity)
        self.published = self.published[-self.capacity:]
        self.last_snapshot_step = step
        # Removing a snapshot from future matchmaking must never change an
        # opponent in an unfinished episode. Release retired models only when
        # no simulation still references them.
        occupied = set(self.assignments.cpu().tolist())
        for old in list(self.snapshots):
            if old not in self.published and old not in occupied:
                del self.snapshots[old]

    def maybe_snapshot(self, step):
        if step - self.last_snapshot_step >= self.interval:
            self.add_snapshot(step)

    @torch.no_grad()
    def resample(self, reset_mask):
        simulations = reset_mask.nonzero().flatten()
        count = len(simulations)
        if not count:
            return
        historical = torch.rand(count, device=self.device, generator=self.generator) < self.fraction
        choices = torch.randint(len(self.published), (count,), device=self.device, generator=self.generator)
        identities = torch.tensor(self.published, device=self.device)[choices]
        self.assignments[simulations] = torch.where(historical, identities, -1)
        self.learner_sides[simulations] = torch.randint(2, (count,), device=self.device, generator=self.generator)

    def learner_mask(self):
        teams = torch.arange(self.n_cars, device=self.device) // self.team_size
        return ((self.assignments[:, None] < 0) | (teams[None] == self.learner_sides[:, None])).flatten()

    @torch.no_grad()
    def apply_actions(self, observations, actions):
        opponents = ~self.learner_mask()
        identities = self.assignments.repeat_interleave(self.n_cars)
        for identity, snapshot in self.snapshots.items():
            indices = ((identities == identity) & opponents).nonzero().flatten()
            if not len(indices):
                continue
            batch = observations[indices]
            logits = snapshot["actor"](batch).float()
            actions[indices] = self.agent.sample_action_logits(logits, batch)[0]

    def state_dict(self):
        # Simulator episodes are not checkpointed. Resume starts fresh episodes,
        # so only published snapshots (not retired in-flight models) are needed.
        return {"format": "historical-opponents-v1", "next_id": self.next_id,
                "last_snapshot_step": self.last_snapshot_step, "rng": self.generator.get_state(),
                "snapshots": [{"id": identity, "step": self.snapshots[identity]["step"],
                               "actor": {key: value.detach().cpu().clone()
                                         for key, value in self.snapshots[identity]["actor"].state_dict().items()}}
                              for identity in self.published]}

    def load_state_dict(self, state):
        if state.get("format") != "historical-opponents-v1" or not state.get("snapshots"):
            raise ValueError("invalid historical opponent pool checkpoint")
        self.snapshots.clear()
        self.published.clear()
        for snapshot in state["snapshots"][-self.capacity:]:
            actor = copy.deepcopy(self.agent.actor).eval().requires_grad_(False)
            actor.load_state_dict(snapshot["actor"], strict=True)
            identity = snapshot["id"]
            self.snapshots[identity] = {"actor": actor, "step": snapshot["step"]}
            self.published.append(identity)
        self.next_id = state["next_id"]
        if self.next_id <= max(self.published):
            raise ValueError("opponent pool snapshot IDs are inconsistent")
        self.last_snapshot_step = state["last_snapshot_step"]
        self.generator.set_state(state["rng"])
        self.assignments.fill_(-1)

    def diagnostic_metrics(self):
        return {"self_play/pool_size": len(self.published),
                "self_play/resident_snapshots": len(self.snapshots),
                "self_play/historical_match_fraction": (self.assignments >= 0).float().mean().item(),
                "self_play/latest_snapshot_step": self.last_snapshot_step}
