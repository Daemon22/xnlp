#!/usr/bin/env python3
"""Generate isiXhosa text using XNLP model for verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

print("Loading XNLP model...")
predictor = XNLPPredictor.load('artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt')
print("Model loaded.\n")

# Generate first paragraph
print("Paragraph 1:")
output1 = predictor.generate(
    "Ngubani uXNLP? Yini le projekthi?",
    max_new_tokens=100,
    temperature=0.7
)
print(output1)
print()

# Generate second paragraph  
print("Paragraph 2:")
output2 = predictor.generate(
    "Yini iimeko yale projekthi maninzi?",
    max_new_tokens=100,
    temperature=0.7
)
print(output2)
