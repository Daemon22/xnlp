#!/usr/bin/env python3
"""Verify XNLP model generates isiXhosa text."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')

print("Testing XNLP model with various prompts:\n")

# Test with several prompts from the evaluation
prompts = [
    "Inkosi yethu",
    "Ityala lamawele.",
    "Umntu ngumntu ngabantu.",
    "AmaXhosa anembali ende.",
    "Ndiyafunda isiXhosa."
]

for prompt in prompts:
    output = predictor.generate(prompt, max_new_tokens=50, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print()
