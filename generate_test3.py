#!/usr/bin/env python3
"""Generate isiXhosa text using XNLP model for verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')

# Generate paragraphs using sentence-starting prompts that are likely to continue
print("Paragraph 1:")
output1 = predictor.generate(
    "Umntu",
    max_new_tokens=80,
    temperature=0.8,
    top_k=40
)
print(output1)
print()

print("Paragraph 2:")
output2 = predictor.generate(
    "AmaXhosa",
    max_new_tokens=80,
    temperature=0.8,
    top_k=40
)
print(output2)
