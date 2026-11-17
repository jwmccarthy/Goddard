@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0venv\Scripts\python.exe" (
  echo Python environment missing. Run setup.bat first.
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" -u "%~dp0launch.py" %*
exit /b %errorlevel%
