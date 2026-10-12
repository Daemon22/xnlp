#!/usr/bin/env python3
"""Test raw generation capability."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

print("Testing generation with different checkpoints...")
print()

# Try best_model.pt
print("=" * 80)
print("Testing best_model.pt (epoch 10):")
print("=" * 80)
predictor = XNLPPredictor.load(r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt', device='cpu')
output = predictor.generate("Molo. ", max_new_tokens=50, temperature=0.8, do_sample=True)
print(f"Output: {output}")
print(f"Length: {len(output)}")
print()

# Try last_model.pt
print("=" * 80)
print("Testing last_model.pt (epoch 15):")
print("=" * 80)
predictor = XNLPPredictor.load(r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\last_model.pt', device='cpu')
output = predictor.generate("Molo. ", max_new_tokens=50, temperature=0.8, do_sample=True)
print(f"Output: {output}")
print(f"Length: {len(output)}")
print()

# Try checkpoint_epoch15.pt
print("=" * 80)
print("Testing checkpoint_epoch15.pt:")
print("=" * 80)
predictor = XNLPPredictor.load(r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\checkpoint_epoch15.pt', device='cpu')
output = predictor.generate("Molo. ", max_new_tokens=50, temperature=0.8, do_sample=True)
print(f"Output: {output}")
print(f"Length: {len(output)}")
print()

# Try without sampling
print("=" * 80)
print("Testing best_model.pt without sampling (greedy):")
print("=" * 80)
predictor = XNLPPredictor.load(r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt', device='cpu')
output = predictor.generate("Molo. ", max_new_tokens=50, temperature=1.0, do_sample=False)
print(f"Output: {output}")
print(f"Length: {len(output)}")
