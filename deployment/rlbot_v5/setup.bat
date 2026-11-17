@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv venv
if errorlevel 1 (
  echo Install Python 3.12 for Windows from python.org, including the Python launcher.
  pause
  exit /b 1
)
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check the error above.
  pause
  exit /b 1
)
venv\Scripts\python.exe self_check.py
if errorlevel 1 (
  echo Self-check failed. Do not launch the bot until the error is resolved.
  pause
  exit /b 1
)
echo Ready. Add bot.toml to RLBot V5, or run play.bat to play against Goddard.
pause
