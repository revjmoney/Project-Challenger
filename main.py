"""
Project Challenger — Headless mode (no UI).
Prefer controller.py for interactive use.

Copyright (c) 2026 Rev. J. Money.
Non-commercial learning/research use only. See LICENSE and NOTICE.

Usage:
  python main.py                  # train if needed, then run live
  python main.py --no-train       # skip training
  python main.py --train-only     # train and exit
  python main.py --retrain        # force retrain
"""
import argparse
import multiprocessing
import sys
import time

from config import CONFIG, is_demo_mode
from bot_manager import BotManager, models_are_trained
from database import init_db
from training import train_all_models
from startup import check_install, tui_auth


def main():
    parser = argparse.ArgumentParser(description="Project Challenger (headless)")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--no-train",   action="store_true")
    parser.add_argument("--retrain",    action="store_true")
    args = parser.parse_args()

    check_install()
    tui_auth()

    active = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]
    mode   = "DEMO" if is_demo_mode() else CONFIG["COINBASE"]["PRODUCT_ID"]

    print("=" * 60)
    print("  PROJECT CHALLENGER — Headless Mode")
    print(f"  {mode}  |  Models: {', '.join(active)}")
    print("=" * 60)

    init_db()

    if not args.no_train:
        if args.retrain or not models_are_trained():
            train_all_models()
        else:
            print("[MAIN] Models found — skipping training. Use --retrain to force.")

    if args.train_only:
        print("[MAIN] --train-only done. Exiting.")
        return

    if not models_are_trained():
        print("[MAIN] ERROR: No model files. Run without --no-train first.")
        sys.exit(1)

    manager = BotManager()
    manager.start()

    print("\n[MAIN] Running. Ctrl+C to stop.")
    print("[MAIN] Tip: run 'python controller.py' for the interactive TUI.\n")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
        manager.stop()
        print("[MAIN] Done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
