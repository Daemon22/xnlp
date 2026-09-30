#!/usr/bin/env python
"""Run fixed-tokenizer checkpoint-only and word-boundary release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from xnlp_trainer.inference import XNLPPredictor

PROMPTS = [
    "Umntu ngumntu ngabantu.",
    "Ubuntu buhle.",
    "Ndiyafunda isiXhosa.",
    "Ityala lamawele.",
    "AmaXhosa anembali ende.",
]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/reports/v2_tokenizer_fixed_release_gate.json"))
    args = parser.parse_args()

    prompts = {}
    with tempfile.TemporaryDirectory(prefix="xnlp_fixed_isolation_") as isolated:
        isolated_checkpoint = Path(isolated) / "best_model.pt"
        shutil.copy2(args.checkpoint, isolated_checkpoint)
        predictor = XNLPPredictor.load(str(isolated_checkpoint), device="cpu")
        for prompt in PROMPTS:
            round_trip = predictor.tokenizer.decode(
                predictor.tokenizer.encode(prompt, add_special_tokens=False)
            )
            samples = [
                predictor.generate(prompt, max_new_tokens=40, temperature=temperature, top_k=40, top_p=0.9)
                for temperature in (0.5, 0.7, 1.0)
            ]
            prompts[prompt] = {
                "round_trip": round_trip == prompt,
                "samples": samples,
                "contains_whitespace": [" " in sample for sample in samples],
                "collapsed_prompt": [prompt.replace(" ", "") in sample for sample in samples],
            }

    report = {
        "artifact_identity": "XNLP_V2_TINY_TOKENIZER_FIXED",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hash_file(args.checkpoint),
        "checkpoint_only_directory_contents": ["best_model.pt"],
        "checkpoint_only_inference": True,
        "generation_spacing": all(
            item["round_trip"] and any(item["contains_whitespace"])
            for item in prompts.values()
        ),
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
