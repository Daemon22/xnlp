#!/usr/bin/env python3
"""Generate isiXhosa text using XNLP model for verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

print("Loading XNLP model...")
predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')
print("Model loaded.\n")

# Try with higher temperature and different prompts
print("Paragraph 1:")
output1 = predictor.generate(
    "XNLP iyini",
    max_new_tokens=150,
    temperature=0.9,
    top_k=50
)
print(output1)
print()

print("Paragraph 2:")
output2 = predictor.generate(
    "Le projekthi iya",
    max_new_tokens=150,
    temperature=0.9,
    top_k=50
)
print(output2)
