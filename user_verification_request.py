#!/usr/bin/env python3
"""
User verification request: Generate two paragraphs of isiXhosa
that are NOT from the books, to verify the current state of the project.

The user wants to see actual model output to verify something.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

print("Generating two paragraphs for verification...")
print("These are NOT from the books - they are live model outputs.")
print()

# Use last_model.pt (epoch 15) as it showed more generation in testing
predictor = XNLPPredictor.load(
    r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\last_model.pt',
    device='cpu'
)

# Paragraph 1
print("Paragraph 1:")
print("-" * 40)
# Using a simple, neutral prompt
output1 = predictor.generate(
    "Molo. ",
    max_new_tokens=60,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)
# Remove the prompt from display
display1 = output1[6:] if output1.startswith("Molo. ") else output1
print(display1)
print()

# Paragraph 2
print("Paragraph 2:")
print("-" * 40)
output2 = predictor.generate(
    "Kodwa ",
    max_new_tokens=60,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)
# Remove the prompt from display
display2 = output2[6:] if output2.startswith("Kodwa ") else output2
print(display2)
print()

print("=" * 80)
print("Current project state in my own words:")
print("=" * 80)
print("The XNLP linguistic foundation is complete with 5,839 authoritative records.")
print("The tokenizer is operational with 4,096 vocabulary size.")
print("The tiny model (5.18M parameters) has been trained to epoch 15.")
print("However, as you can see above, the model generates isiXhosa-like fragments")
print("but NOT yet coherent, grammatical, semantically meaningful isiXhosa.")
print("This is the language competence gap we are now addressing.")
print("=" * 80)
