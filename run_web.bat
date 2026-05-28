@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat or setup_cpu.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo.
echo  Starting Project Challenger Web GUI...
echo  Open http://localhost:8765 in your browser.
echo  Press Ctrl+C to stop.
echo.
python web_app.py %*
if errorlevel 1 pause
