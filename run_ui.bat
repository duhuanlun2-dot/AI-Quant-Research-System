@echo off
set "ROOT=%~dp0"
cd /d "%ROOT%"
if exist "%ROOT%\.venv\Scripts\pythonw.exe" (
  "%ROOT%\.venv\Scripts\pythonw.exe" "%ROOT%\run_ui.py"
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\run_ui.py"
) else (
  python "%ROOT%\run_ui.py"
)
if errorlevel 1 (
  echo.
  echo Failed to start AI Quant Research System UI.
  echo Try running run_once.bat first to create the virtual environment.
  pause
)
