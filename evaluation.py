import json
from pathlib import Path

import torch
import trueskill

from carl.gymnasium import CARLTorchVectorEnv


class TrueSkillEvaluator:
    def __init__(
        self,
        policy,
        opponent_pool,
        logger,
        checkpoint_dir: Path,
        interval:       int,
        n_simulations:  int,
        n_blue:         int,
        n_orange:       int,
        frameskip:      int,
        max_ticks:      int,
        opponents:      int,
        draw_probability: float,
        seed:            int,
    ) -> None:
        if interval < 1 or n_simulations < 1 or opponents < 1:
            raise ValueError("TrueSkill settings must be positive")
        if not 0.0 <= draw_probability < 1.0:
            raise ValueError("draw probability must be between zero and one")

        self.policy = policy
        self.opponent_pool = opponent_pool
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.n_simulations = n_simulations
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.frameskip = frameskip
        self.max_ticks = max_ticks
        self.opponents = opponents
        self.seed = seed
        self.next_evaluation = interval
        self.current_step = 0
        self.evaluation_count = 0
        self.rating_system = trueskill.TrueSkill(
            draw_probability=draw_probability
        )
        self.current_rating = self.rating_system.create_rating()
        self.snapshot_ratings = {}
        self.rating_games = {}
        self.env = None

    def ready(self, step: int) -> bool:
        self.current_step = step
        if step < self.next_evaluation:
            return False
        while self.next_evaluation <= step:
            self.next_evaluation += self.interval
        return True

    @torch.no_grad()
    def run(self) -> None:
        if self.env is None:
            self.env = CARLTorchVectorEnv(
                n_sim=self.n_simulations,
                n_blue=self.n_blue,
                n_orange=self.n_orange,
                seed=self.seed + 1,
                frameskip=self.frameskip,
                max_ticks=self.max_ticks,
                synchronize=False,
            )

        snapshot_ids = self.opponent_pool.ids[-self.opponents :]
        wins = draws = games = 0
        devices = [self.policy.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.evaluation_count)
            for snapshot_id in snapshot_ids:
                opponent = self.opponent_pool.policy(
                    snapshot_id, self.policy.device
                )
                outcomes = torch.cat(
                    (
                        self._play(opponent, current_is_blue=True),
                        self._play(opponent, current_is_blue=False),
                    )
                ).cpu()
                wins += int(outcomes.gt(0).sum())
                draws += int(outcomes.eq(0).sum())
                games += len(outcomes)
                self._rate(snapshot_id, outcomes)

        self.evaluation_count += 1
        rating = self.current_rating
        self.logger.update(
            {
                "TrueSkill": {
                    "mu":        rating.mu,
                    "sigma":     rating.sigma,
                    "skill":     rating.mu - 3.0 * rating.sigma,
                    "games":      games,
                    "win_rate":   wins / games,
                    "draw_rate":  draws / games,
                    "opponents":  len(snapshot_ids),
                }
            },
            step=self.current_step,
        )
        self._write_ratings()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _play(self, opponent, current_is_blue: bool) -> torch.Tensor:
        observation = self.env.reset()
        current_state = self.policy.initial_state(
            self.n_simulations
            * (self.n_blue if current_is_blue else self.n_orange)
        )
        opponent_state = opponent.initial_state(
            self.n_simulations
            * (self.n_orange if current_is_blue else self.n_blue)
        )
        active = torch.ones(
            self.n_simulations, dtype=torch.bool, device=self.policy.device
        )
        outcomes = torch.zeros(
            self.n_simulations, dtype=torch.float32, device=self.policy.device
        )
        max_steps = (self.max_ticks + self.frameskip - 1) // self.frameskip

        for _ in range(max_steps):
            grouped = observation.view(
                self.n_simulations, self.n_blue + self.n_orange, -1
            )
            blue = grouped[:, : self.n_blue].flatten(0, 1)
            orange = grouped[:, self.n_blue :].flatten(0, 1)
            if current_is_blue:
                blue_output = self.policy.act(blue, current_state)
                orange_output = opponent.act(orange, opponent_state)
                current_state = blue_output.next_state
                opponent_state = orange_output.next_state
            else:
                blue_output = opponent.act(blue, opponent_state)
                orange_output = self.policy.act(orange, current_state)
                opponent_state = blue_output.next_state
                current_state = orange_output.next_state

            action = torch.cat(
                (
                    blue_output.action.view(
                        self.n_simulations, self.n_blue, -1
                    ),
                    orange_output.action.view(
                        self.n_simulations, self.n_orange, -1
                    ),
                ),
                dim=1,
            ).flatten(0, 1)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated | truncated).view(
                self.n_simulations, self.n_blue + self.n_orange
            ).any(dim=-1)
            finished = active & done
            blue_result = reward.view(
                self.n_simulations, self.n_blue + self.n_orange
            )[:, 0]
            outcomes[finished] = (
                blue_result[finished]
                if current_is_blue
                else -blue_result[finished]
            )
            active &= ~done
            if not active.any():
                break

        return outcomes

    def _rate(self, snapshot_id: int, outcomes: torch.Tensor) -> None:
        opponent_rating = self.snapshot_ratings.setdefault(
            snapshot_id,
            trueskill.Rating(
                mu=self.current_rating.mu,
                sigma=self.current_rating.sigma,
            ),
        )
        rated_games = 0
        for outcome in outcomes.tolist():
            if outcome > 0:
                self.current_rating, opponent_rating = self.rating_system.rate_1vs1(
                    self.current_rating, opponent_rating
                )
                rated_games += 1
            elif outcome < 0:
                opponent_rating, self.current_rating = self.rating_system.rate_1vs1(
                    opponent_rating, self.current_rating
                )
                rated_games += 1
        self.snapshot_ratings[snapshot_id] = opponent_rating
        self.rating_games[snapshot_id] = (
            self.rating_games.get(snapshot_id, 0) + rated_games
        )

    def _write_ratings(self) -> None:
        snapshots = {
            str(snapshot_id): {
                "mu":    rating.mu,
                "sigma": rating.sigma,
                "skill": rating.mu - 3.0 * rating.sigma,
                "games": self.rating_games.get(snapshot_id, 0),
            }
            for snapshot_id, rating in self.snapshot_ratings.items()
        }
        payload = {
            "current": {
                "mu":    self.current_rating.mu,
                "sigma": self.current_rating.sigma,
                "skill": self.current_rating.mu - 3.0 * self.current_rating.sigma,
            },
            "snapshots": snapshots,
        }
        (self.checkpoint_dir / "trueskill_ratings.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


__all__ = ["TrueSkillEvaluator"]
