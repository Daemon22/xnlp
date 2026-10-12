#!/usr/bin/env python3
"""
Generate two paragraphs of isiXhosa describing the XNLP project state.
The prompts are NOT from the books - they are neutral requests for the model
to speak about the project itself.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

print("XNLP PROJECT STATE - VERIFICATION GENERATION")
print("=" * 80)
print()

# Load best checkpoint
predictor = XNLPPredictor.load(
    r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt',
    device='cpu'
)
print()

# Generation parameters
gen_params = {
    'max_new_tokens': 120,
    'temperature': 0.75,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.2,
    'do_sample': True
}

print("Paragraph 1 - General project state:")
print("-" * 80)
prompt1 = "Ndiyakwazi ukuthetha ngesiXhosa. XNLP yinto"
output1 = predictor.generate(prompt1, **gen_params)
print(output1)
print()

print("Paragraph 2 - Current capabilities:")
print("-" * 80)
prompt2 = "Lo msebenzi wesiXhosa saseXNLP ube"
output2 = predictor.generate(prompt2, **gen_params)
print(output2)
print()

print("=" * 80)
print("Raw outputs for verification:")
print("-" * 80)
print(f"P1: {output1}")
print(f"P2: {output2}")
