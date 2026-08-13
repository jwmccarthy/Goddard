import numpy as np
import subtr_actor

from typing import Tuple
from pathlib import Path
from rich.progress import track

from utils import clip_vector_norm, quaternion_to_forward_up


PLAYER_START = 9
PLAYER_WIDTH = 17
POSITION_SCALE = np.array([4108.0, 6000.0, 2076.0])
BALL_MAX_SPEED = 6000.0
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_SPEED = 5.5
BOOST_MAX = 100.0


class ReplayParser:

    def __init__(
        self,
        fps:   float = 15.0,
        dtype: str = "float32",
        normalize: bool = True
    ) -> None:
        self.fps = fps
        self.dtype = dtype
        self.normalize = normalize

    def _parse(self, replay_file: str) -> Tuple[dict, np.ndarray]:
        return subtr_actor.get_ndarray_with_info_from_replay_filepath(
            replay_file,
            global_feature_adders=[
                "BallRigidBody",
                "SecondsRemaining",
                "ReplicatedStateName",
                "ReplicatedGameStateTimeRemaining"
            ],
            player_feature_adders=[
                "PlayerRigidBodyQuaternionVelocities",
                "PlayerBoost",
                "PlayerDemolishedBy"
            ],
            fps=self.fps,
            dtype=self.dtype
        )
    
    
    def _filter(
        self,
        meta: dict,
        states: np.ndarray,
        match_info: dict
    ) -> np.ndarray:
        headers = meta["column_headers"]["global_headers"]

        game_state_idx = headers.index("game state")
        countdown_idx = headers.index("kickoff countdown")

        # Keep the first kickoff row and active gameplay before the goal sequence.
        kickoff_start = (
            (states[:, game_state_idx] == 28)
            & (states[:, countdown_idx] == 0)
        )
        gameplay = states[:, game_state_idx] == 30
        active = (
            kickoff_start | gameplay
        )

        active_indices = np.flatnonzero(active)

        goal_frames = match_info["goal_frames"]
        if len(goal_frames):
            goal_samples = np.sort(
                goal_frames * self.fps / match_info["record_fps"]
            )
            next_goal = np.searchsorted(
                goal_samples,
                active_indices,
                side="left"
            )
            has_goal = next_goal < len(goal_samples)
            goal_distance = np.full_like(
                active_indices,
                np.inf,
                dtype=np.float32
            )
            goal_distance[has_goal] = (
                goal_samples[next_goal[has_goal]]
                - active_indices[has_goal]
            )
            active_indices = active_indices[goal_distance >= 5.0 * self.fps]

        # Remove filtering-only columns
        states = np.delete(
            states[active_indices],
            [game_state_idx, countdown_idx],
            axis=1
        )

        return states

    def _get_match_info(self, replay_file: str) -> dict:
        meta = subtr_actor.get_replay_meta(replay_file)["replay_meta"]
        headers = dict(meta["all_headers"])

        scores = (
            headers.get("Team0Score", 0),
            headers.get("Team1Score", 0),
        )

        return {
            "winning_team": int(scores[1] > scores[0]),
            "team_sizes": (
                len(meta["team_zero"]),
                len(meta["team_one"]),
            ),
            "goal_frames": np.array(
                [goal["frame"] for goal in headers.get("Goals", [])],
                dtype=np.float32
            ),
            "record_fps": float(headers.get("RecordFPS", self.fps))
        }


    def _format(
        self,
        states: np.ndarray,
        match_info: dict,
    ) -> np.ndarray:
        # Ball position, linear velocity, angular velocity
        ball = np.concatenate(
            (
                states[:, 0:3],
                states[:, 6:9],
                states[:, 9:12],
            ),
            axis=-1
        )

        players = states[:, 13:].reshape(len(states), -1, 15)

        # Put the winning team first
        blue_count, orange_count = match_info["team_sizes"]

        if match_info["winning_team"] == 0:
            order = np.arange(blue_count + orange_count)
        else:
            order = np.concatenate(
                (
                    np.arange(blue_count, blue_count + orange_count),
                    np.arange(blue_count),
                )
            )

        players = players[:, order]

        forward, up = quaternion_to_forward_up(
            players[..., 3:7]
        )

        players = np.concatenate(
            (
                players[..., 0:3],
                players[..., 7:10],
                np.deg2rad(players[..., 10:13]),
                forward, up,
                players[..., 13:14] * (100.0 / 255.0),
                (players[..., 14:15] >= 0).astype(states.dtype)
            ),
            axis=-1
        )

        if match_info["winning_team"] == 1:
            for start in range(0, 9, 3):
                ball[..., start:start + 2] *= -1
            for start in range(0, 15, 3):
                players[..., start:start + 2] *= -1

        return np.concatenate(
            (
                ball,
                players.reshape(len(states), -1),
            ),
            axis=-1
        )

    def _normalize(self, states: np.ndarray) -> np.ndarray:
        clip_vector_norm(states[:, 3:6], BALL_MAX_SPEED)
        clip_vector_norm(states[:, 6:9], BALL_MAX_ANG_SPEED)

        states[:, 0:3] /= POSITION_SCALE
        states[:, 3:6] /= BALL_MAX_SPEED
        states[:, 6:9] /= BALL_MAX_ANG_SPEED

        players = states[:, PLAYER_START:].reshape(
            len(states), -1, PLAYER_WIDTH
        )

        clip_vector_norm(players[..., 3:6], CAR_MAX_SPEED)
        clip_vector_norm(players[..., 6:9], CAR_MAX_ANG_SPEED)

        players[..., 0:3] /= POSITION_SCALE
        players[..., 3:6] /= CAR_MAX_SPEED
        players[..., 6:9] /= CAR_MAX_ANG_SPEED
        players[..., 15]  /= BOOST_MAX

        return states

    def _group_player_properties(
        self, states: np.ndarray
    ) -> dict[str, np.ndarray]:
        player_count = (states.shape[1] - PLAYER_START) // PLAYER_WIDTH

        starts = [
            PLAYER_START + player * PLAYER_WIDTH
            for player in range(player_count)
        ]

        property_offsets = {
            "position":         np.arange(0, 3),
            "velocity":         np.arange(3, 6),
            "angular_velocity": np.arange(6, 9),
            "forward":          np.arange(9, 12),
            "up":               np.arange(12, 15),
            "boost":            np.arange(15, 16),
            "demoed":           np.arange(16, 17),
        }

        grouped = {
            name: np.stack(
                [states[:, start + offsets] for start in starts],
                axis=1,
            )
            for name, offsets in property_offsets.items()
        }
        grouped["demoed"] = grouped["demoed"].astype(bool)

        return grouped

    def _save(self, replay_file: str, states: np.ndarray) -> None:
        output_file = Path(replay_file).with_suffix(".npy")
        np.save(output_file, states)

    def parse_replays(self, replay_dir: str, output_dir: str) -> None:
        replay_dir = Path(replay_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        replay_files = list(replay_dir.glob("*.replay"))

        for replay_file in track(
            replay_files,
            description="Parsing replays..."
        ):
            output_file = output_dir / f"{replay_file.stem}.npy"

            try:
                match_info = self._get_match_info(str(replay_file))
                meta, states = self._parse(str(replay_file))
                states = self._filter(meta, states, match_info)

                if not len(states):
                    continue

                states = self._format(states, match_info)
                if self.normalize:
                    states = self._normalize(states)
                self._save(output_file, states)

            except Exception as e:
                print(f"Failed {replay_file.name}: {e}")


if __name__ == "__main__":
    parser = ReplayParser()

    replay_dir = "./ballchasing_replays/replays"
    output_dir = "./ballchasing_replays/parsed_replays"

    parser.parse_replays(replay_dir, output_dir)
