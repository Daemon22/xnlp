#!/usr/bin/env python3
"""Generate two paragraphs of isiXhosa text for verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')

# First paragraph - general statement about the project
print("Paragraph 1:")
output1 = predictor.generate(
    "Inkosi yethu",
    max_new_tokens=60,
    temperature=0.8
)
print(output1)
print()

# Second paragraph - continuation on a different topic
print("Paragraph 2:")
output2 = predictor.generate(
    "UMemeso ophiqo",
    max_new_tokens=60,
    temperature=0.8
)
print(output2)
