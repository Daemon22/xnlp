#!/usr/bin/env python3
"""Direct verification: Generate two paragraphs of isiXhosa."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')

print("XNLP DIRECT VERIFICATION")
print("="*60)
print("Generating two paragraphs of isiXhosa from the current model.\n")

# Paragraph 1
print("Paragraph 1:")
output1 = predictor.generate(
    "Umntu",
    max_new_tokens=100,
    temperature=0.8
)
print(output1)
print()

# Paragraph 2
print("Paragraph 2:")
output2 = predictor.generate(
    "Kodwa",
    max_new_tokens=100,
    temperature=0.8
)
print(output2)

print()
print("="*60)
print("END OF VERIFICATION")
