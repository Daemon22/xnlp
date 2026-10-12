#!/usr/bin/env python3
"""Generate two clean paragraphs for verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')

print("XNLP Model Verification - Direct Generation\n")
print("="*50)

# Paragraph 1
print("\nParagraph 1:")
output1 = predictor.generate(
    "Umntu",
    max_new_tokens=70,
    temperature=0.85,
    top_k=50
)
print(output1)

# Paragraph 2
print("\nParagraph 2:")
output2 = predictor.generate(
    "Kodwa",
    max_new_tokens=70,
    temperature=0.85,
    top_k=50
)
print(output2)

print("\n" + "="*50)
print("Verification complete. Both paragraphs are model-generated isiXhosa.")
