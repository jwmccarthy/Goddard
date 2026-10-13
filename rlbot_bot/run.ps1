$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$Root\.venv\Scripts\python.exe" "$Root\run.py"
