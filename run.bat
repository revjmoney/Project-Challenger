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
echo  Starting Project Challenger (headless mode)...
echo  Ctrl+C to stop.
echo.

python main.py %*
if errorlevel 1 (
    echo.
    echo  [ERROR] main.py exited with an error ^(see above^).
    echo.
    pause
    exit /b 1
)
endlocal
