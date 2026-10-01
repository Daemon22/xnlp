#!/usr/bin/env python
"""Monitor training and automatically evaluate after each epoch.

Watches the training log for epoch completion markers and runs
evaluate_checkpoint.py on the latest best_model.pt when a new
checkpoint is saved.

Usage::

    python watch_and_eval.py                 # watch and evaluate
    python watch_and_eval.py --once          # evaluate once and exit
"""
import argparse
import json
import os
import sys
import time
import re
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

CHECKPOINT_DIR = "artifacts/xnlp_v2_tiny_tokenizer_fixed/model"
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
REPORT_DIR = "data/reports"


def get_ckpt_mtime():
    if os.path.exists(BEST_CKPT):
        return os.path.getmtime(BEST_CKPT)
    return 0


def run_evaluation():
    """Run evaluate_checkpoint.py and return the report."""
    print(f"\n=== Running evaluation on {BEST_CKPT} ===")
    result = subprocess.run(
        [sys.executable, "evaluate_checkpoint.py", BEST_CKPT],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Watch training and auto-evaluate")
    parser.add_argument("--once", action="store_true",
                        help="Evaluate once and exit")
    args = parser.parse_args()

    last_mtime = 0
    last_report_time = 0

    if args.once:
        if os.path.exists(BEST_CKPT):
            run_evaluation()
        else:
            print(f"No checkpoint found at {BEST_CKPT}")
        return

    print(f"Watching for checkpoint updates in {CHECKPOINT_DIR}/...")
    print(f"Baseline mtime: {get_ckpt_mtime()}")

    while True:
        current_mtime = get_ckpt_mtime()
        if current_mtime > last_mtime and current_mtime > last_report_time:
            # New checkpoint saved — evaluate
            print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoint detected "
                  f"(mtime: {time.ctime(current_mtime)})")
            time.sleep(5)  # Wait for write to complete
            success = run_evaluation()
            if success:
                last_report_time = current_mtime
                last_mtime = current_mtime
            else:
                print("Evaluation failed, will retry next cycle")
                last_mtime = current_mtime

        # Print summary every 5 minutes
        if int(time.time()) % 300 < 10:
            log_path = os.path.join(
                max(
                    (f for f in os.listdir("runs") if f.startswith("v2_")),
                    default="runs"
                ),
                "train.log"
            )
            # Just print a heartbeat
            pass

        time.sleep(30)


if __name__ == "__main__":
    main()
