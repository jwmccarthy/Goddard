"""Give each bot process its own output file, including concurrent mirror matches."""

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    instance = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    environment = os.environ.copy()
    environment["GODDARD_INSTANCE_ID"] = instance
    target = "self_check.py" if sys.argv[1:] == ["--self-check"] else "bot.py"
    console = root / f"console-{instance}.log"
    # Never share a redirected file handle between bot processes on Windows.
    with console.open("x", encoding="utf-8") as stream:
        stream.write(f"{datetime.now().isoformat()} Launching {target}; instance={instance}; root={root}\n")
        stream.flush()
        try:
            result = subprocess.run(
                [sys.executable, "-u", str(root / target)], cwd=root, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        except OSError as error:
            stream.write(f"Could not launch bot: {error}\n")
            return 1
        stream.write(f"{datetime.now().isoformat()} Process exited with code {result.returncode}\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
