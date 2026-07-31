import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pro_replay_dataset import pro_duel, uses_octane_hitbox


class ProReplayDatasetTests(unittest.TestCase):
    def test_pro_duel_only_requires_one_player_per_team(self):
        replay = {
            "blue": {"players": [{"name": "pro"}]},
            "orange": {"players": [{"name": "opponent"}]},
        }

        self.assertTrue(pro_duel(replay))
        replay["orange"]["players"].append({"name": "extra"})
        self.assertFalse(pro_duel(replay))

    def test_octane_filter_requires_both_players(self):
        subtr_actor = Mock()
        subtr_actor.get_replay_meta.return_value = {
            "replay_meta": {
                "team_zero": [{"car_hitbox_family": "Octane"}],
                "team_one": [{"car_hitbox_family": "Dominus"}],
            }
        }

        with patch.dict(sys.modules, {"subtr_actor": subtr_actor}):
            self.assertFalse(uses_octane_hitbox(Path("replay.replay")))

        subtr_actor.get_replay_meta.return_value["replay_meta"]["team_one"][0][
            "car_hitbox_family"
        ] = "Octane"
        with patch.dict(sys.modules, {"subtr_actor": subtr_actor}):
            self.assertTrue(uses_octane_hitbox(Path("replay.replay")))


if __name__ == "__main__":
    unittest.main()
