from pathlib import Path
from time import sleep

from rlbot.flat import MatchPhase
from rlbot.managers import MatchManager

if __name__ == "__main__":
    manager = MatchManager()
    try:
        manager.start_match(Path(__file__).resolve().parent / "match.toml")
        while manager.packet is None or manager.packet.match_info.match_phase != MatchPhase.Ended:
            sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        manager.shut_down()
