from pathlib import Path

REPLAYS_DIR = Path(__file__).resolve().parent

RAW_DIR = REPLAYS_DIR / "raw"
PARSED_DIR = REPLAYS_DIR / "parsed_replays"

__all__ = ["PARSED_DIR", "RAW_DIR", "REPLAYS_DIR"]
