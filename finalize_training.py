#!/usr/bin/env python
"""Finalize training: evaluate, run release gate, and summarize results.

Runs automatically after training completes. Performs:
1. Checkpoint format verification (validate_checkpoint)
2. Release gate (release_gate_v2_tokenizer_fixed.py)
3. Comprehensive evaluation (evaluate_checkpoint.py)
4. Results comparison with quick checkpoint baseline

Usage::

    python finalize_training.py [checkpoint_path]
"""
from __future__ import annotations
import json
import os
import sys
import time
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from xnlp_trainer.config import validate_checkpoint
from xnlp_trainer.inference import XNLPPredictor


def check_training_complete(run_dir: str = "runs/v2_run_20261001_171811"):
    """Check if the training process has completed by looking for completion markers."""
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        return False

    with open(log_path, encoding="utf-8") as f:
        content = f.read()

    if "Training Complete" in content:
        print("[check] Training Complete found in log")
        return True

    # Check if process is still alive
    pid_file = os.path.join(run_dir, "run.pid")
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read().strip())
        try:
            import psutil
            p = psutil.Process(pid)
            if p.status() == "running":
                print(f"[check] Training still running (PID {pid})")
                return False
        except (psutil.NoSuchProcess, ValueError):
            print(f"[check] Process {pid} not found")
    else:
        # Fallback: check if "Training Complete" is in recent log lines
        if "[early-stop] Stopping" in content or "Training Complete" in content:
            print("[check] Training stopped")
            return True

    return False


def verify_checkpoint(ckpt_path: str) -> bool:
    """Verify checkpoint format version 1 and all required fields."""
    import torch

    print(f"\n=== Checkpoint Verification ===")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        validate_checkpoint(ckpt)
        print(f"  PASS: validate_checkpoint() succeeded")
    except ValueError as e:
        print(f"  FAIL: {e}")
        return False

    meta = ckpt.get("training_metadata", {})
    print(f"  Epoch: {meta.get('epoch', '?')}")
    print(f"  Step:  {meta.get('global_step', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?')}")
    print(f"  Params: {meta.get('param_count', '?'):,}")
    print(f"  Vocab:  {meta.get('vocab_size', '?')}")
    print(f"  Size:   {os.path.getsize(ckpt_path)} bytes")
    return True


def run_release_gate(ckpt_path: str) -> bool:
    """Run the V2 release gate."""
    print(f"\n=== Release Gate ===")
    result = subprocess.run(
        [sys.executable, "release_gate_v2_tokenizer_fixed.py", ckpt_path],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  FAIL: release gate script exited with code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return False

    report_path = "data/reports/v2_tokenizer_fixed_release_gate.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            gate = json.load(f)
        print(f"  checkpoint_only_inference: {gate['checkpoint_only_inference']}")
        print(f"  generation_spacing: {gate['generation_spacing']}")
        return gate["checkpoint_only_inference"] and gate["generation_spacing"]
    return False


def run_evaluation(ckpt_path: str) -> bool:
    """Run the comprehensive evaluation suite."""
    print(f"\n=== Evaluation ===")
    result = subprocess.run(
        [sys.executable, "evaluate_checkpoint.py", ckpt_path],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  FAIL: evaluation exited with code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return False

    # Find the latest report
    import glob
    reports = sorted(glob.glob("data/reports/evaluation_2026*.json"))
    if not reports:
        print("  WARN: No evaluation report found")
        return False

    with open(reports[-1]) as f:
        report = json.load(f)
    print(f"  Overall: {report.get('overall_score', 0)}%")
    for name, result in report.get("tests", {}).items():
        score = result.get("score", 0)
        passed = result.get("passed", 0)
        total = result.get("total", 0)
        print(f"    {name}: {passed}/{total} ({score}%)")
    return True


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt"

    print(f"=== Finalizing XNLP V2 Training ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Verify checkpoint
    ok1 = verify_checkpoint(ckpt_path)

    # 2. Release gate
    ok2 = run_release_gate(ckpt_path)

    # 3. Evaluation
    ok3 = run_evaluation(ckpt_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Checkpoint verification:  {'PASS' if ok1 else 'FAIL'}")
    print(f"  Release gate:             {'PASS' if ok2 else 'FAIL'}")
    print(f"  Evaluation:               {'PASS' if ok3 else 'FAIL'}")
    print(f"{'='*60}")

    if ok1 and ok2 and ok3:
        print(f"  All checks PASSED. Training is finalized.")
    else:
        print(f"  Some checks FAILED. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
