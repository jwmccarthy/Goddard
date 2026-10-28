from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"


__all__ = ["DATA_DIR", "RAW_DIR", "PARSED_DIR"]
