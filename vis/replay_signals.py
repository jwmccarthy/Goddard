"""Read recorded jump holds from each car's canonical POV, without simulation."""
from pathlib import Path

import numpy as np

from replays.layout import team_observation_size


def replay_jump_holds(path, rows, team_size):
    """Return frame × car int8 holds; -1 means that POV was not available."""
    path = Path(path)
    holds = np.full((len(rows), 2 * team_size), -1, dtype=np.int8)
    column = team_observation_size(team_size) + 5  # Recorded last-jump / holding latch.
    holds[:, 0] = rows[:, column] > .5
    parts = path.stem.split("-", 2)
    if len(parts) != 3:
        return holds
    for other_path in path.parent.glob(f"*-{parts[1]}-{parts[2]}.npy"):
        if other_path == path:
            continue
        other = np.load(other_path, mmap_mode="r", allow_pickle=False)
        try:
            if other.shape != rows.shape:
                continue
            for car in range(1, 2 * team_size):
                expected = np.array(rows[:, 9 + 21 * car:30 + 21 * car], copy=True)
                if car >= team_size:
                    for field in (0, 3, 6, 9, 12):
                        expected[:, field:field + 2] *= -1
                matched = np.isclose(expected, other[:, 9:30], rtol=1e-4, atol=1e-4).all(-1)
                holds[matched, car] = other[matched, column] > .5
        finally:
            if getattr(other, "_mmap", None) is not None:
                other._mmap.close()
    return holds


def hold_ranges(holds):
    """Compact half-open source-frame intervals, stable under arbitrary scrubbing."""
    result = []
    for car in holds.T:
        edges = np.diff(np.r_[False, car > 0, False].astype(np.int8))
        result.append(np.column_stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))).tolist())
    return result
