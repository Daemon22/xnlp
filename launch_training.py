#!/usr/bin/env python
"""Launch V2 training as a fully detached process.

Usage::

    python launch_training.py [extra args for start_v2_training.py]

This script:
1. Creates a run directory with timestamp.
2. Writes PID to ``runs/v2_full_<ts>/run.pid``.
3. Launches ``start_v2_training.py`` via ``CREATE_NEW_PROCESS_GROUP``
   so the training survives shell-session expiry.
4. Exits immediately, leaving the training process running.
"""
import os
import sys
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"v2_full_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train.log")
    pid_path = os.path.join(run_dir, "run.pid")

    # Build the argument list for start_v2_training.py
    extra_args = sys.argv[1:]
    args = [sys.executable, "-u", "start_v2_training.py"] + extra_args

    # Open log file for the subprocess
    log_file = open(log_path, "w", encoding="utf-8")

    # CREATE_NEW_PROCESS_GROUP = 0x00000200
    # This detaches the process from the current process group so it
    # survives when the parent shell session expires.
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        creationflags=creationflags,
    )

    with open(pid_path, "w") as f:
        f.write(str(proc.pid))

    log_file.close()

    print(f"Training launched as detached process.")
    print(f"  PID:       {proc.pid}")
    print(f"  Run dir:   {run_dir}")
    print(f"  Log file:  {log_path}")
    print(f"  PID file:  {pid_path}")


if __name__ == "__main__":
    main()
