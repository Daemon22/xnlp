#!/usr/bin/env python
"""Monitor an in-progress V2 training run.

Usage::

    python monitor_training.py                          # check latest run
    python monitor_training.py runs/v2_full_20240101_*  # check specific run

Prints:
  - Current epoch, step, loss
  - Best validation loss so far
  - Estimated time remaining
  - Whether the process is still alive
"""
from __future__ import annotations
import os
import sys
import time
import glob
import re
import json
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def find_latest_run():
    """Find the most recent run directory in runs/."""
    dirs = sorted(glob.glob("runs/v2_full_*") + glob.glob("runs/v2_run_*"))
    return dirs[-1] if dirs else None


def parse_log(log_path):
    """Extract training metrics from the log file."""
    if not os.path.exists(log_path):
        return {"lines": 0, "steps": [], "epochs": [], "val_losses": []}

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    steps = []
    epochs = []
    val_losses = []

    for line in lines:
        s = line.strip()
        # Parse step logs: "  Ep 1/20  step 50  loss 6.6601  avg 7.2597  lr 1.50e-04"
        m = re.search(
            r"Ep\s+(\d+)/(\d+)\s+step\s+(\d+)\s+loss\s+([\d.]+)\s+avg\s+([\d.]+)\s+lr\s+([\d.e+-]+)",
            s,
        )
        if m:
            steps.append({
                "epoch": int(m.group(1)),
                "max_epoch": int(m.group(2)),
                "step": int(m.group(3)),
                "loss": float(m.group(4)),
                "avg_loss": float(m.group(5)),
                "lr": float(m.group(6)),
            })
            continue

        # Parse epoch completion: "[epoch 1/20]  val_loss=5.3607  ppl=212.8"
        m = re.search(
            r"\[epoch\s+(\d+)/(\d+)\]\s+val_loss=([\d.]+)\s+ppl=([\d.]+)",
            s,
        )
        if m:
            epochs.append({
                "epoch": int(m.group(1)),
                "max_epoch": int(m.group(2)),
                "val_loss": float(m.group(3)),
                "perplexity": float(m.group(4)),
            })
            val_losses.append(float(m.group(3)))

    # Parse last epoch val loss from training history if available
    return {
        "lines": len(lines),
        "steps": steps,
        "epochs": epochs,
        "val_losses": val_losses,
        "last_lines": [l.rstrip()[:100] for l in lines[-5:]],
    }


def check_pid_alive(pid):
    """Check if a process with the given PID is alive."""
    try:
        return psutil.pid_exists(pid) and any(
            p.info["pid"] == pid for p in psutil.process_iter(["pid"])
        )
    except Exception:
        return False


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    if not run_dir:
        print("No training runs found in runs/")
        sys.exit(1)

    log_path = os.path.join(run_dir, "train.log")
    pid_path = os.path.join(run_dir, "run.pid")

    pid = None
    if os.path.exists(pid_path):
        pid = int(open(pid_path).read().strip())

    print(f"=== Training Monitor ===")
    print(f"Run dir: {run_dir}")
    print(f"PID:     {pid}")
    if pid and check_pid_alive(pid):
        print(f"Status:  ALIVE")
    elif pid:
        print(f"Status:  NOT RUNNING (PID {pid} not found)")
    print()

    result = parse_log(log_path)
    print(f"Log lines: {result['lines']}")

    if result["steps"]:
        last = result["steps"][-1]
        print(f"\nLatest step: epoch {last['epoch']}/{last['max_epoch']}, "
              f"step {last['step']}, loss {last['loss']:.4f}, avg {last['avg_loss']:.4f}, "
              f"lr {last['lr']:.2e}")

        # Estimate time per step
        if len(result["steps"]) >= 2:
            step_times = []
            for i in range(1, min(len(result["steps"]), 10)):
                # We don't have timestamps, so estimate from log line count
                pass

    if result["epochs"]:
        last_epoch = result["epochs"][-1]
        print(f"Latest epoch: {last_epoch['epoch']}/{last_epoch['max_epoch']}, "
              f"val_loss={last_epoch['val_loss']:.4f}, ppl={last_epoch['perplexity']:.1f}")
        print(f"\nAll epochs:")
        for e in result["epochs"]:
            print(f"  Epoch {e['epoch']}: val_loss={e['val_loss']:.4f}, ppl={e['perplexity']:.1f}")

    # Check for completion marker
    if result["lines"] > 0:
        all_lines = open(log_path, encoding="utf-8").readlines()
        for line in reversed(all_lines):
            if "Training Complete" in line:
                print("\n*** TRAINING COMPLETE ***")
                break
            if "early-stop" in line and "Stopping" in line:
                print("\n*** EARLY STOPPING TRIGGERED ***")
                break

    print(f"\nLast log lines:")
    for line in result["last_lines"]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
