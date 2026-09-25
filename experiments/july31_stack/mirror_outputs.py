"""Mirror fast, RAM-backed July training outputs to the persistent RunPod volume."""

import argparse
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def mirror(source: Path, destination: Path) -> None:
    common = ["rsync", "-a", "--exclude=*.tmp"]
    subprocess.run(
        [*common, "--exclude=events.out.tfevents.*", f"{source}/", f"{destination}/"],
        check=True,
    )
    # The event log only grows: transfer the appended bytes, not the full file
    # on every backup. Model/checkpoint files above can be replaced atomically.
    subprocess.run(
        [
            "rsync", "-a", "--append", "--include=*/",
            "--include=events.out.tfevents.*", "--exclude=*",
            f"{source}/", f"{destination}/",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--interval", type=int, default=120)
    options = parser.parse_args()
    if options.interval < 1 or options.pid < 1:
        parser.error("interval and pid must be positive")
    if not options.source.is_dir() or not options.destination.is_dir():
        parser.error("source and destination directories must already exist")

    while True:
        try:
            mirror(options.source, options.destination)
            print(f"Mirrored at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
        except subprocess.CalledProcessError as error:
            print(f"Mirror failed (will retry): {error}", flush=True)

        if not running(options.pid):
            break
        if (
            shutil.disk_usage(options.source).free < 2 * 1024**3
            or shutil.disk_usage("/").free < 512 * 1024**2
        ):
            print("Stopping training before local output storage fills", flush=True)
            os.kill(options.pid, signal.SIGTERM)
        time.sleep(options.interval)


if __name__ == "__main__":
    main()
