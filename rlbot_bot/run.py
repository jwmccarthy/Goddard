from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.managers import MatchManager


def main() -> None:
    manager = MatchManager()
    manager.start_match(Path(__file__).with_name("rlbot.toml"))
    while (
        manager.packet is None
        or manager.packet.match_info.match_phase != flat.MatchPhase.Ended
    ):
        sleep(0.1)
    manager.shut_down()


if __name__ == "__main__":
    main()
