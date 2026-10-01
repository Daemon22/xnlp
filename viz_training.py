#!/usr/bin/env python
"""Render training loss curves from a training log.

Parses a train.log file and produces a text-based loss chart
plus a JSON summary suitable for plotting.

Usage::

    python viz_training.py [run_dir]
    python viz_training.py runs/v2_run_20261001_171811
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_log(log_path: str) -> dict:
    """Parse a training log and extract step/loss data."""
    steps = []
    val_losses = []
    checkpoint_time = None

    step_re = re.compile(
        r"Ep (\d+)/(\d+)  step (\d+)  loss ([\d.]+)  avg ([\d.]+)  lr ([\d.e+-]+)"
    )
    val_re = re.compile(
        r"Epoch (\d+).*val_loss ([\d.]+)"
    )
    ckpt_re = re.compile(r"checkpoint saved")

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        m = step_re.search(line)
        if m:
            epoch = int(m.group(1))
            total_epochs = int(m.group(2))
            step = int(m.group(3))
            loss = float(m.group(4))
            avg_loss = float(m.group(5))
            lr = float(m.group(6))
            steps.append({
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "loss": loss,
                "avg_loss": avg_loss,
                "lr": lr,
                "line": line.strip(),
            })
            continue

        m = val_re.search(line)
        if m:
            epoch = int(m.group(1))
            val_loss = float(m.group(2))
            val_losses.append({"epoch": epoch, "val_loss": val_loss})

    # Extract start time
    start_match = re.search(r"Training started: (.+)", "".join(lines[:5]))
    start_time = start_match.group(1) if start_match else None

    # Check if checkpoint exists
    ckpt_path = os.path.join(
        Path(log_path).parent.parent,
        "artifacts", "xnlp_v2_tiny_tokenizer_fixed", "model",
        "best_model.pt"
    )
    checkpoint_saved = os.path.exists(ckpt_path)

    return {
        "steps": steps,
        "val_losses": val_losses,
        "start_time": start_time,
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_path": ckpt_path,
        "total_log_lines": len(lines),
    }


def text_chart(values, width=50, label=""):
    """Render a simple text-based chart."""
    if not values:
        return "  (no data)"
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        max_val = min_val + 1
    lines = []
    for i, v in enumerate(values):
        bar_len = int((v - min_val) / (max_val - min_val) * (width - 1))
        bar = "─" * bar_len + "●"
        lines.append(f"  {label}{i}: {v:.4f} {bar}")
    return "\n".join(lines)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if run_dir is None:
        # Find most recent run
        runs = [d for d in os.listdir("runs") if d.startswith("v2_run_")]
        if runs:
            run_dir = os.path.join("runs", sorted(runs)[-1])
        else:
            print("No run directory found in runs/")
            sys.exit(1)

    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        sys.exit(1)

    data = parse_log(log_path)
    steps = data["steps"]
    val_losses = data["val_losses"]

    print(f"=== Training Visualization ===")
    print(f"Run dir:    {run_dir}")
    print(f"Log:        {log_path}")
    print(f"Start time: {data['start_time']}")
    print(f"Log lines:  {data['total_log_lines']}")
    print()

    if not steps:
        print("No training steps found in log.")
        return

    # Summary stats
    latest = steps[-1]
    first = steps[0]
    print("=== Training Summary ===")
    if val_losses:
        for vl in val_losses:
            print(f"  Epoch {vl['epoch']}: val_loss = {vl['val_loss']:.4f}")
    print(f"  First step:   epoch={first['epoch']}, step={first['step']}, "
          f"loss={first['loss']:.4f}, lr={first['lr']:.2e}")
    print(f"  Latest step:  epoch={latest['epoch']}, step={latest['step']}, "
          f"loss={latest['loss']:.4f}, avg_loss={latest['avg_loss']:.4f}, "
          f"lr={latest['lr']:.2e}")
    print(f"  Loss delta:   {first['loss']:.4f} → {latest['loss']:.4f} "
          f"(Δ={latest['loss'] - first['loss']:.4f})")
    print(f"  Steps in epoch: {latest['step']}")
    if latest.get("total_epochs"):
        pct = latest["step"] / 1507 * 100 if latest["step"] < 1507 else 100
        print(f"  Epoch {latest['epoch']} progress: ~{pct:.0f}%")
    print()

    # Loss chart (every 10th step to keep it readable)
    step_losses = [s["loss"] for s in steps]
    avg_losses = [s["avg_loss"] for s in steps]
    sampled_steps = steps[::max(1, len(steps) // 20)]

    print("=== Step Loss Chart (sampled) ===")
    for s in sampled_steps:
        print(f"  step {s['step']:5d}  loss={s['loss']:.4f}  "
              f"avg={s['avg_loss']:.4f}  lr={s['lr']:.2e}")
    print()

    # Save JSON summary
    report = {
        "run_dir": run_dir,
        "start_time": data["start_time"],
        "checkpoint_saved": data["checkpoint_saved"],
        "first_step": {
            "epoch": first["epoch"], "step": first["step"],
            "loss": first["loss"], "lr": first["lr"],
        },
        "latest_step": {
            "epoch": latest["epoch"], "step": latest["step"],
            "loss": latest["loss"], "avg_loss": latest["avg_loss"],
            "lr": latest["lr"],
        },
        "loss_delta": latest["loss"] - first["loss"],
        "val_losses": val_losses,
        "total_steps_logged": len(steps),
    }

    report_path = os.path.join(run_dir, "training_viz.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
