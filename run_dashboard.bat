@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo  [ERROR] Virtual environment not found.
    echo         Run setup.bat first to create it.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo  Starting Project Challenger Dashboard monitor...
echo  Open this in a second terminal while the bot is running.
echo  Ctrl+C to stop.
echo.

python dashboard.py %*
if errorlevel 1 (
    echo.
    echo  [ERROR] dashboard.py exited with an error ^(see above^).
    echo.
    pause
    exit /b 1
)
endlocal
