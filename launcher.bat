@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ================================================================
echo   PROJECT CHALLENGER  ^|  WEB GUI LAUNCHER
echo  ================================================================
echo.

REM ── Step 1: Setup — only runs when venv is absent ───────────────────────────

if not exist ".venv\Scripts\activate.bat" (
    echo  [SETUP] First run — creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo  [ERROR] Python 3.10+ is required.
        echo  Download from:  https://www.python.org/downloads/
        pause & exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo  [SETUP] Installing dependencies (this may take a few minutes^)...
    pip install -r requirements.txt

    echo.
    echo  [SETUP] Installing PyTorch (CPU^)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    echo.
    echo  ----------------------------------------------------------------
    echo   Setup complete. If you have an NVIDIA GPU and want CUDA support,
    echo   manually run the appropriate pip install torch command later.
    echo  ----------------------------------------------------------------
    echo.
) else (
    call .venv\Scripts\activate.bat
)

REM ── Step 2: Ensure runtime directories exist ────────────────────────────────

if not exist "data"   mkdir data
if not exist "models" mkdir models
if not exist "logs"   mkdir logs

REM ── Step 3: Initialise DB and coin cache ────────────────────────────────────

echo  [INFO] Initialising database and coin cache...
python -c "from database import init_db; init_db(); from coin_manager import refresh_available_coins; refresh_available_coins()"

REM ── Step 4: Write the live status monitor script and launch it ───────────────
REM   The PS1 starts the web server, opens the browser after 4 s, then shows a
REM   status panel (last 20 log lines) that refreshes every 3 s.  Press q to quit.

set "CMON=%TEMP%\challenger_mon_%RANDOM%.ps1"

REM Disable delayed expansion so ! inside echo lines reach the PS1 file intact
setlocal disabledelayedexpansion

> "%CMON%" echo # Project Challenger — web-server monitor
>> "%CMON%" echo $logFile = 'logs\web_app.log'
>> "%CMON%" echo $logMaxBytes = 4404019
>> "%CMON%" echo.
>> "%CMON%" echo # Pre-launch cleanup: kill port 8765 holders and orphaned web_app.py workers
>> "%CMON%" echo $portPids = @(netstat -ano ^| Select-String ':8765\s' ^| ForEach-Object { ($_ -split '\s+')[-1] } ^| Where-Object { $_ -match '^\d+$' -and $_ -ne '0' } ^| Sort-Object -Unique)
>> "%CMON%" echo foreach ($procId in $portPids) {
>> "%CMON%" echo     Write-Host "  [WARN] Killing port 8765 holder: PID $procId"
>> "%CMON%" echo     taskkill /F /PID $procId 2^>$null ^| Out-Null
>> "%CMON%" echo }
>> "%CMON%" echo $workers = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue ^| Where-Object { $_.CommandLine -like '*web_app.py*' }
>> "%CMON%" echo foreach ($w in $workers) {
>> "%CMON%" echo     Write-Host "  [WARN] Killing orphaned worker: PID $($w.ProcessId)"
>> "%CMON%" echo     Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue
>> "%CMON%" echo }
>> "%CMON%" echo if ($portPids.Count -gt 0 -or ($workers -and $workers.Count -gt 0)) { Start-Sleep 1 }
>> "%CMON%" echo.
>> "%CMON%" echo # Background log-size watcher — trims to half when limit is hit
>> "%CMON%" echo $null = Start-Job -ScriptBlock {
>> "%CMON%" echo     param($lf, $lmax)
>> "%CMON%" echo     while ($true) {
>> "%CMON%" echo         Start-Sleep 10
>> "%CMON%" echo         if (Test-Path $lf) {
>> "%CMON%" echo             $sz = (Get-Item $lf).Length
>> "%CMON%" echo             if ($sz -gt $lmax) {
>> "%CMON%" echo                 $keep = [int]($lmax / 2)
>> "%CMON%" echo                 $bytes = [System.IO.File]::ReadAllBytes($lf)
>> "%CMON%" echo                 $trimmed = $bytes[($bytes.Length - $keep)..($bytes.Length - 1)]
>> "%CMON%" echo                 [System.IO.File]::WriteAllBytes($lf, $trimmed)
>> "%CMON%" echo                 Add-Content $lf "--- [log trimmed at $(Get-Date)] ---"
>> "%CMON%" echo             }
>> "%CMON%" echo         }
>> "%CMON%" echo     }
>> "%CMON%" echo } -ArgumentList $logFile, $logMaxBytes
>> "%CMON%" echo.
>> "%CMON%" echo # Launch web server — cmd.exe merges stdout+stderr into the log file
>> "%CMON%" echo $psi = New-Object System.Diagnostics.ProcessStartInfo
>> "%CMON%" echo $psi.FileName = 'cmd.exe'
>> "%CMON%" echo $psi.Arguments = "/c python web_app.py >> $logFile 2>&1"
>> "%CMON%" echo $psi.UseShellExecute = $false
>> "%CMON%" echo $psi.CreateNoWindow  = $true
>> "%CMON%" echo $p = [System.Diagnostics.Process]::Start($psi)
>> "%CMON%" echo.
>> "%CMON%" echo # Open browser 4 seconds after server starts
>> "%CMON%" echo Start-Process powershell -ArgumentList "-WindowStyle Hidden -NoProfile -Command ""Start-Sleep 4; Start-Process 'http://localhost:8765'""" -WindowStyle Hidden
>> "%CMON%" echo.
>> "%CMON%" echo function Stop-Server {
>> "%CMON%" echo     Write-Host ''
>> "%CMON%" echo     Write-Host '  Stopping server...'
>> "%CMON%" echo     taskkill /F /T /PID $p.Id 2^>$null ^| Out-Null
>> "%CMON%" echo     $null = $p.WaitForExit(5000)
>> "%CMON%" echo     Write-Host "  Server stopped. Logs saved to $logFile"
>> "%CMON%" echo     Write-Host ''
>> "%CMON%" echo }
>> "%CMON%" echo.
>> "%CMON%" echo function Show-Status {
>> "%CMON%" echo     Clear-Host
>> "%CMON%" echo     $kb     = if (Test-Path $logFile) { [int]((Get-Item $logFile).Length / 1KB) } else { 0 }
>> "%CMON%" echo     $status = if (-not $p.HasExited) { "RUNNING  (PID $($p.Id))" } else { 'STOPPED' }
>> "%CMON%" echo     Write-Host ''
>> "%CMON%" echo     Write-Host ' ================================================================'
>> "%CMON%" echo     Write-Host '  PROJECT CHALLENGER  ^|  WEB GUI'
>> "%CMON%" echo     Write-Host ' ================================================================'
>> "%CMON%" echo     Write-Host "  Status:   $status"
>> "%CMON%" echo     Write-Host '  Web:      http://localhost:8765'
>> "%CMON%" echo     Write-Host "  Log:      $logFile  ($kb KB / 4200 KB max)"
>> "%CMON%" echo     Write-Host ' ----------------------------------------------------------------'
>> "%CMON%" echo     Write-Host '  Recent activity:'
>> "%CMON%" echo     Write-Host ' ----------------------------------------------------------------'
>> "%CMON%" echo     if (Test-Path $logFile) { Get-Content $logFile -Tail 20 ^| ForEach-Object { "  $_" } }
>> "%CMON%" echo     Write-Host ''
>> "%CMON%" echo     Write-Host ' ================================================================'
>> "%CMON%" echo     Write-Host '  q = quit'
>> "%CMON%" echo     Write-Host ' ================================================================'
>> "%CMON%" echo     Write-Host -NoNewline '  > '
>> "%CMON%" echo }
>> "%CMON%" echo.
>> "%CMON%" echo while ($true) {
>> "%CMON%" echo     Show-Status
>> "%CMON%" echo     $deadline = [DateTime]::Now.AddSeconds(3)
>> "%CMON%" echo     while ([DateTime]::Now -lt $deadline) {
>> "%CMON%" echo         if ([Console]::KeyAvailable) {
>> "%CMON%" echo             $key = [Console]::ReadKey($true)
>> "%CMON%" echo             if ($key.Key -eq 'Q') {
>> "%CMON%" echo                 Clear-Host
>> "%CMON%" echo                 Stop-Server
>> "%CMON%" echo                 exit 0
>> "%CMON%" echo             }
>> "%CMON%" echo             break
>> "%CMON%" echo         }
>> "%CMON%" echo         Start-Sleep -Milliseconds 100
>> "%CMON%" echo     }
>> "%CMON%" echo     if ($p.HasExited) {
>> "%CMON%" echo         Clear-Host
>> "%CMON%" echo         Write-Host ''
>> "%CMON%" echo         Write-Host '  [ERROR] Server process stopped unexpectedly.'
>> "%CMON%" echo         Write-Host "  Check $logFile for details."
>> "%CMON%" echo         Write-Host ''
>> "%CMON%" echo         exit 1
>> "%CMON%" echo     }
>> "%CMON%" echo }

powershell -NoProfile -ExecutionPolicy Bypass -File "%CMON%"
set EXIT_CODE=%ERRORLEVEL%
del "%CMON%" 2>nul

endlocal & endlocal & set "EXIT_CODE=%EXIT_CODE%"

if %EXIT_CODE% neq 0 (
    echo.
    echo  [ERROR] Web server exited with an error. Check logs\web_app.log for details.
    echo.
    pause
)
