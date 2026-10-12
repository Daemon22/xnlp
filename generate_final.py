#!/usr/bin/env python3
"""Generate isiXhosa text using XNLP model for verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')

# Use prompts from the evaluation that we know produce good output
print("Paragraph 1:")
output1 = predictor.generate(
    "Inkosi yethu",
    max_new_tokens=100,
    temperature=0.8
)
print(output1)
print()

print("Paragraph 2:")
output2 = predictor.generate(
    "Ityala lamawele.",
    max_new_tokens=100,
    temperature=0.8
)
print(output2)
