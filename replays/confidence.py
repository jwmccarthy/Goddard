"""Hash-linked reconstruction diagnostics; never discriminator input features."""

from pathlib import Path

import numpy as np

from replays.reconstruction import VERSION
from replays.timing import Status
from replays.utils import file_sha256

FORMAT = 1

PER_ROW_FIELDS = ("status", "confidence", "source_ticks", "unsafe")


def confidence_path(path: Path) -> Path:
    return path.with_suffix(".quality.npz")


def write_confidence(
    path:    Path,
    quality: dict[str, np.ndarray],
    version: str = VERSION
) -> dict:
    np.savez_compressed(
        confidence_path(path),
        format=np.asarray(FORMAT),
        reconstruction=np.asarray(version),
        source_sha256=np.asarray(file_sha256(path)),
        **quality
    )

    status = quality["status"]

    return {
        "version":        version,
        "quality_format": FORMAT,
        "status_bits":    {item.name.lower(): int(item) for item in Status},
        "status_counts":  {
            item.name.lower(): int(((status & item) != 0).sum())
            for item in Status
        },
        "unsafe_rows":     int(quality["unsafe"].any(1).sum()),
        "entity_ids":      quality["entity_ids"].tolist(),
        "fresh_counts":    quality["fresh_counts"].tolist(),
        "native_rows":     int(quality["native_rows"]),
        "confidence_kind": "evidence score, not calibrated probability"
    }


def load_confidence(
    path:    Path,
    count:   int,
    n_cars:  int,
    version: str | None = None
) -> dict[str, np.ndarray]:
    target = confidence_path(path)

    with np.load(target, allow_pickle=False) as archive:
        stored = archive["reconstruction"].item()

        if (
            archive["format"].item() != FORMAT
            or stored != VERSION
            or (version is not None and stored != version)
        ):
            raise ValueError(f"unsupported replay reconstruction quality: {target}")

        if archive["source_sha256"].item() != file_sha256(path):
            raise ValueError(
                f"reconstruction quality belongs to a different state export: {target}"
            )

        fields = {
            key: archive[key]
            for key in ("row_ticks", "entity_ids", *PER_ROW_FIELDS)
        }

    shape = (count, n_cars + 1)

    if (
        fields["row_ticks"].shape != (count,)
        or fields["entity_ids"].shape != (n_cars + 1,)
        or any(fields[key].shape != shape for key in PER_ROW_FIELDS)
        or not np.isfinite(fields["row_ticks"]).all()
        or (np.diff(fields["row_ticks"]) <= 0).any()
        or not np.isfinite(fields["confidence"]).all()
        or not np.isfinite(fields["source_ticks"]).all()
        or ((fields["confidence"] < 0) | (fields["confidence"] > 1)).any()
        or fields["unsafe"].dtype != np.bool_
        or fields["status"].dtype != np.uint16
    ):
        raise ValueError(f"invalid replay reconstruction quality: {target}")

    return fields
