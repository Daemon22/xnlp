#!/usr/bin/env python3
"""Debug generation issue."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load(
    r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt',
    device='cpu'
)

# Test different prompts
prompts = [
    "Molo",
    "Molo ",
    "Molo.",
    "Molo. ",
    "Ngitshele",
    "Ngitshele ",
    "",
]

for p in prompts:
    print(f"Prompt: '{p}'")
    output = predictor.generate(p, max_new_tokens=20, temperature=0.8, do_sample=True)
    print(f"Output: '{output}'")
    print(f"Generated chars: {len(output) - len(p) if output.startswith(p) else len(output)}")
    print()
