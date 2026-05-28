#!/usr/bin/env bash
# Project Challenger — Web GUI launcher (Linux / macOS)
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found. Run ./setup.sh or ./setup_cpu.sh first."
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo ""
echo "  Starting Project Challenger Web GUI..."
echo "  Open http://localhost:8765 in your browser."
echo "  Press Ctrl+C to stop."
echo ""

exec python web_app.py "$@"
