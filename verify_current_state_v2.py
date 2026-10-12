#!/usr/bin/env python3
"""Generate two paragraphs of original isiXhosa to verify current model capability.

This script uses the XNLP model to generate original language (not from books)
that describes the current state of the XNLP project.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

print("=" * 80)
print("XNLP CURRENT STATE VERIFICATION - FREE GENERATION TEST")
print("=" * 80)
print()

# Load the best model checkpoint
print("Loading model checkpoint...")
checkpoint_path = r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt'
predictor = XNLPPredictor.load(checkpoint_path, device='cpu')
print()

# Set generation parameters for coherent output
gen_params = {
    'max_new_tokens': 150,
    'temperature': 0.85,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.1,
    'do_sample': True
}

print("Test 1 - Simple continuation:")
print("-" * 40)
prompt1 = "Molo. "
output1 = predictor.generate(prompt1, **gen_params)
print(f"Prompt: {prompt1}")
print(f"Output: {output1}")
print()

print("Test 2 - Project description:")
print("-" * 40)
prompt2 = "XNLP iyini? "
output2 = predictor.generate(prompt2, **gen_params)
print(f"Prompt: {prompt2}")
print(f"Output: {output2}")
print()

print("Test 3 - Free generation with neutral prompt:")
print("-" * 40)
prompt3 = "Ngitshele. "
output3 = predictor.generate(prompt3, **gen_params)
print(f"Prompt: {prompt3}")
print(f"Output: {output3}")
print()

print("Test 4 - Longer generation:")
print("-" * 40)
prompt4 = ""
output4 = predictor.generate(prompt4, max_new_tokens=200, temperature=0.85, do_sample=True)
print(f"Prompt: (empty)")
print(f"Output: {output4}")
print()

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
