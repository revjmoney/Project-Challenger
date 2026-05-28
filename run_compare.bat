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
echo  Running model comparison report...
echo.

python compare_models.py %*
if errorlevel 1 (
    echo.
    echo  [ERROR] compare_models.py exited with an error ^(see above^).
    echo.
)
pause
endlocal
