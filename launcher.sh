#!/usr/bin/env bash
# Project Challenger — Web GUI launcher
# Handles first-run setup, shows status, and confirms before acting.
# Run:  bash launcher.sh
cd "$(dirname "$0")"

# ── Helpers ───────────────────────────────────────────────────────────────────

confirm() {
    local prompt="${1:-Continue?}"
    printf "\n  %s [Y/n] " "$prompt"
    read -r reply
    case "$reply" in
        ""|[Yy]*) return 0 ;;
        *) echo "  Aborted."; exit 0 ;;
    esac
}

status_ok()   { echo "  [  OK  ] $1"; }
status_warn() { echo "  [ WARN ] $1"; }
status_info() { echo "  [ INFO ] $1"; }
status_fail() { echo "  [ FAIL ] $1"; }

divider() { echo "  ────────────────────────────────────────────────────────"; }

# ── Header ────────────────────────────────────────────────────────────────────

echo ""
echo " ================================================================"
echo "  ⚡  PROJECT CHALLENGER  |  WEB GUI LAUNCHER"
echo " ================================================================"
echo ""

# ── Step 1: System status report ─────────────────────────────────────────────

divider
echo "  SYSTEM STATUS"
divider

# Python
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    status_ok "Python:        $PY_VER"
else
    status_fail "Python:        python3 not found in PATH"
    echo ""
    echo "  Cannot continue without Python 3. Please install it first."
    exit 1
fi

# Virtual environment
if [ -f ".venv/bin/activate" ]; then
    status_ok "Virtual env:   .venv present"
    FIRST_RUN=false
else
    status_warn "Virtual env:   NOT found (first-run setup required)"
    FIRST_RUN=true
fi

# requirements.txt
if [ -f "requirements.txt" ]; then
    REQ_COUNT=$(grep -c '.' requirements.txt 2>/dev/null || echo "?")
    status_ok "requirements:  requirements.txt found ($REQ_COUNT packages listed)"
else
    status_warn "requirements:  requirements.txt not found"
fi

# Data / models dirs
DATA_STATUS="present"
MODELS_STATUS="present"
[ ! -d "data" ]   && DATA_STATUS="missing (will create)"
[ ! -d "models" ] && MODELS_STATUS="missing (will create)"
status_info "Data dir:      $DATA_STATUS"
status_info "Models dir:    $MODELS_STATUS"

# web_app.py
if [ -f "web_app.py" ]; then
    status_ok "web_app.py:    found"
else
    status_fail "web_app.py:    NOT found — cannot launch server"
    exit 1
fi

echo ""

# ── Step 2: First-run setup ───────────────────────────────────────────────────

if [ "$FIRST_RUN" = true ]; then
    divider
    echo "  FIRST-RUN SETUP"
    divider
    echo ""
    echo "  This will:"
    echo "    1. Create a Python virtual environment (.venv)"
    echo "    2. Install all packages from requirements.txt"
    echo "    3. Install PyTorch (CPU build, may be ~200 MB)"
    echo ""
    confirm "Run first-time setup now?"

    set -e
    echo ""
    echo "  [SETUP] Creating virtual environment..."
    python3 -m venv .venv
    # shellcheck source=/dev/null
    source .venv/bin/activate

    echo "  [SETUP] Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt

    echo "  [SETUP] Installing PyTorch (CPU)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    echo ""
    status_ok "Setup complete."
    set +e
else
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

# ── Step 3: Ensure runtime directories ───────────────────────────────────────

mkdir -p data models logs

# ── Step 4: Confirm launch ────────────────────────────────────────────────────

LOG_FILE="logs/web_app.log"
LOG_MAX_BYTES=4404019   # 4.20 MB exactly (4.20 * 1024 * 1024)

# Show existing log info if any
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    LOG_SIZE_KB=$(( LOG_SIZE / 1024 ))
    status_info "Log file:      $LOG_FILE (${LOG_SIZE_KB} KB used of 4200 KB max)"
else
    status_info "Log file:      $LOG_FILE (new)"
fi

divider
echo "  READY TO LAUNCH"
divider
echo ""
echo "  Web server:  http://localhost:8765"
echo "  Log file:    $LOG_FILE  (max 4.20 MB, then oldest lines are trimmed)"
echo "  Browser opens automatically after 4 seconds."
echo ""
echo "  Type  q + Enter  to stop the server cleanly."
echo ""
confirm "Launch the web server now?"

# ── Step 4.5: Clear port 8765 and stale workers ──────────────────────────────

echo ""
divider
echo "  PRE-LAUNCH CLEANUP"
divider

PORT_PIDS=$(lsof -ti:8765 2>/dev/null)
if [ -n "$PORT_PIDS" ]; then
    status_warn "Port 8765 in use — stopping: $(echo "$PORT_PIDS" | tr '\n' ' ')"
    echo "$PORT_PIDS" | xargs kill -9 2>/dev/null
    sleep 1
    status_ok "Port 8765 cleared."
else
    status_ok "Port 8765 is free."
fi

ORPHAN_PIDS=$(pgrep -f "python.*web_app\.py" 2>/dev/null)
if [ -n "$ORPHAN_PIDS" ]; then
    status_warn "Orphaned workers found — stopping: $(echo "$ORPHAN_PIDS" | tr '\n' ' ')"
    echo "$ORPHAN_PIDS" | xargs kill -9 2>/dev/null
    sleep 1
fi

echo ""

# ── Step 5: Clean-exit handler ────────────────────────────────────────────────

cleanup() {
    clear
    echo ""
    echo "  Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    kill "$WATCHER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    echo "  Server stopped. Logs saved to $LOG_FILE"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM

# ── Step 6: Open browser + start server (logs → file) ────────────────────────

echo ""
echo "  Starting web server..."
echo ""

# Ensure DB is initialised and available coins are cached before starting
python -c "from database import init_db; init_db(); from coin_manager import refresh_available_coins; refresh_available_coins()" >> "$LOG_FILE" 2>&1

(sleep 4 && (
    open "http://localhost:8765" 2>/dev/null ||
    xdg-open "http://localhost:8765" 2>/dev/null ||
    sensible-browser "http://localhost:8765" 2>/dev/null ||
    true
)) &

python web_app.py >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Background log-size watcher: trims file to last 2.10 MB when limit is hit
(
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        if [ -f "$LOG_FILE" ]; then
            SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
            if [ "$SIZE" -gt "$LOG_MAX_BYTES" ]; then
                KEEP=$(( LOG_MAX_BYTES / 2 ))
                tail -c "$KEEP" "$LOG_FILE" > "$LOG_FILE.tmp" \
                    && mv "$LOG_FILE.tmp" "$LOG_FILE" \
                    && echo "--- [log trimmed at $(date)] ---" >> "$LOG_FILE"
            fi
        fi
        sleep 10
    done
) &
WATCHER_PID=$!

# ── Step 7: Live status display (refreshes every 3 s) ────────────────────────

show_status() {
    LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    LOG_SIZE_KB=$(( LOG_SIZE / 1024 ))
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        SRV_STATUS="RUNNING  (PID $SERVER_PID)"
    else
        SRV_STATUS="STOPPED"
    fi
    clear
    echo ""
    echo " ================================================================"
    echo "  ⚡  PROJECT CHALLENGER  |  WEB GUI"
    echo " ================================================================"
    echo "  Status:   $SRV_STATUS"
    echo "  Web:      http://localhost:8765"
    echo "  Log:      $LOG_FILE  (${LOG_SIZE_KB} KB / 4200 KB max)"
    echo " ----------------------------------------------------------------"
    echo "  Recent activity:"
    echo " ----------------------------------------------------------------"
    tail -n 20 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
    echo ""
    echo " ================================================================"
    echo "  q + Enter = quit"
    echo " ================================================================"
    printf "  > "
}

while true; do
    show_status
    if read -t 3 -r cmd; then
        case "$cmd" in
            q|quit|exit|stop) cleanup ;;
        esac
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        clear
        echo ""
        echo "  [ERROR] Server process stopped unexpectedly."
        echo "  Check $LOG_FILE for details."
        echo ""
        exit 1
    fi
done
