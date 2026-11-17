"""Match/sequence-balanced sampling of original, contiguous expert windows."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from replays import layout
from replays.maneuvers import FORMAT, STRATA
from replays.utils import file_sha256


class ManeuverIndex:
    def __init__(
        self,
        path,
        directory,
        team_size,
    ):
        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        manifest = json.loads(self.path.read_text())
        if manifest.get("format") != FORMAT or manifest.get("team_size") != team_size:
            raise ValueError("incompatible expert maneuver index")
        self.directory = Path(directory)
        self.files = manifest["files"]
        self.windows = []

    def add_file(
        self,
        path,
        rows,
        count,
        candidates,
        lengths,
        offset,
        source_frameskip,
    ):
        record = self.files.get(path.relative_to(self.directory).as_posix())
        if record is None:
            return
        sidecar = path.with_suffix(".json")
        metadata_sha = file_sha256(sidecar) if sidecar.is_file() else None
        if (
            record["rows"] != len(rows) or record["frameskip"] != source_frameskip
            or record["sha256"] != file_sha256(path)
            or record["metadata_sha256"] != metadata_sha
            or record["match_key"] != layout.replay_match_key(path)
        ):
            raise ValueError(f"expert maneuver source changed: {path}")
        for sequence in record["sequences"]:
            begin, takeoff, landing, recovery, end = (
                sequence[key] for key in ("start", "takeoff", "landing", "recovery_end", "stop")
            )
            if (
                not all(isinstance(i, int) for i in (begin, takeoff, landing, recovery, end))
                or not 0 <= begin < takeoff < landing < recovery <= end <= len(rows)
                or sequence["stratum"] not in STRATA
            ):
                raise ValueError(f"invalid maneuver interval: {path}")
            # A frame cap may omit an entire sequence, never retain a partial core.
            if end > count:
                continue
            left, right = np.searchsorted(candidates, [begin, end])
            if right > left:
                self.windows.append((
                    sequence["stratum"], record["match_key"],
                    np.stack((candidates[left:right] + offset, lengths[left:right]), axis=-1),
                ))

    def partition(
        self,
        match_keys,
        device,
    ):
        groups = defaultdict(lambda: defaultdict(list))
        for stratum, key, windows in self.windows:
            if key in match_keys:
                groups[stratum][key].append(windows)
        return ManeuverSampler(groups, device) if groups else None


class ManeuverSampler:
    def __init__(
        self,
        groups,
        device,
    ):
        windows, sequences, matches, strata = [], [], [], []
        window_offset = 0
        self.sequence_counts = {}
        for stratum in sorted(groups):
            match_offset = len(matches)
            count = 0
            for key in sorted(groups[stratum]):
                sequence_offset = len(sequences)
                for values in groups[stratum][key]:
                    windows.append(values)
                    sequences.append((window_offset, len(values)))
                    window_offset += len(values)
                    count += 1
                matches.append((sequence_offset, len(sequences) - sequence_offset))
            strata.append((match_offset, len(matches) - match_offset))
            self.sequence_counts[stratum] = count
        self.windows = torch.as_tensor(np.concatenate(windows), device=device)
        self.sequences = torch.tensor(sequences, device=device)
        self.matches = torch.tensor(matches, device=device)
        self.strata = torch.tensor(strata, device=device)

    def sample(
        self,
        count,
        generator,
    ):
        choices = torch.randint(len(self.strata), (count,), device=self.windows.device, generator=generator)
        for ranges in (self.strata, self.matches, self.sequences):
            selected = ranges[choices]
            choices = selected[:, 0] + (
                torch.rand(count, device=self.windows.device, generator=generator) * selected[:, 1]
            ).long()
        return self.windows[choices].unbind(-1)
