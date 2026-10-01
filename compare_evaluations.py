#!/usr/bin/env python
"""Compare evaluation reports across training epochs.

Parses all evaluation_*.json reports and produces a comparison table
showing how each metric improved from the quick checkpoint to
successive epochs of full-corpus training.

Usage::

    python compare_evaluations.py
"""
import json
import os
import glob
from pathlib import Path

REPORT_DIR = "data/reports"


def load_reports():
    """Load all evaluation reports sorted by timestamp."""
    reports = []
    for path in sorted(glob.glob(f"{REPORT_DIR}/evaluation_2026*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            ts = os.path.basename(path).replace("evaluation_", "").replace(".json", "")
            data["_timestamp"] = ts
            data["_filename"] = os.path.basename(path)
            reports.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return reports


def print_comparison(reports):
    """Print a comparison table of evaluation scores across reports."""
    if not reports:
        print("No evaluation reports found.")
        return

    # Collect all test names
    test_names = set()
    for r in reports:
        for name in r.get("tests", {}):
            test_names.add(name)
    test_names = sorted(test_names)

    # Header
    print("\n" + "=" * 80)
    print("  EVALUATION COMPARISON ACROSS TRAINING CHECKPOINTS")
    print("=" * 80)

    # Column headers
    name_width = max(len("Test"), max(len(n) for n in test_names))
    header = f"{'Test':<{name_width}}"
    for r in reports:
        header += f"  {r['_timestamp'][:15]:>15s}"
    print(header)
    print("-" * len(header))

    # Rows
    for name in test_names:
        row = f"{name:<{name_width}}"
        for r in reports:
            tests = r.get("tests", {})
            if name in tests:
                score = tests[name].get("score", 0)
                row += f"  {score:>14.1f}%"
            else:
                row += f"  {'N/A':>15}"
        print(row)

    # Overall row
    row = f"{'Overall':<{name_width}}"
    for r in reports:
        overall = r.get("overall_score", 0)
        row += f"  {overall:>14.1f}%"
    print("-" * len(header))
    print(row)

    # Details per report
    print("\n" + "=" * 80)
    print("  REPORT DETAILS")
    print("=" * 80)
    for r in reports:
        print(f"\n  {r['_filename']} (overall: {r.get('overall_score', 0)}%)")
        print(f"  Checkpoint: {r.get('checkpoint', 'N/A')}")
        print(f"  Timestamp:  {r.get('timestamp', 'N/A')}")
        for name, result in r.get("tests", {}).items():
            print(f"    {name}: {result.get('passed',0)}/{result.get('total',0)} = {result.get('score',0)}%")

    # Trend analysis
    print("\n" + "=" * 80)
    print("  TREND ANALYSIS")
    print("=" * 80)
    if len(reports) >= 2:
        first = reports[0]
        last = reports[-1]
        for name in test_names:
            if name in first.get("tests", {}) and name in last.get("tests", {}):
                old = first["tests"][name].get("score", 0)
                new = last["tests"][name].get("score", 0)
                delta = new - old
                sign = "+" if delta >= 0 else ""
                print(f"  {name:.<30s} {old:.1f}% -> {new:.1f}% (delta={sign}{delta:.1f}%)")


def main():
    reports = load_reports()
    if not reports:
        print("No evaluation reports found in", REPORT_DIR)
        return

    print(f"Found {len(reports)} evaluation reports:")
    for r in reports:
        overall = r.get("overall_score", 0)
        print(f"  {r['_filename']}: {overall}%")

    print_comparison(reports)


if __name__ == "__main__":
    main()
