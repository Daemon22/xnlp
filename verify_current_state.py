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
predictor = XNLPPredictor.load(r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt', device='cpu')
print()

# Set generation parameters for coherent output
gen_params = {
    'max_new_tokens': 100,
    'temperature': 0.7,
    'top_k': 40,
    'top_p': 0.9,
    'repetition_penalty': 1.2,
    'do_sample': True
}

print("Generating Paragraph 1:")
print("-" * 40)
# Prompt in isiXhosa: "Tell me about the current state of the XNLP project"
prompt1 = "Ngitshele malunga ngesimo esibhekene kuso sesiXhosa saseXNLP namhlanje."
output1 = predictor.generate(prompt1, **gen_params)
print(output1)
print()

print("Generating Paragraph 2:")
print("-" * 40)
# Prompt in isiXhosa: "What is happening with the isiXhosa language model today?"
prompt2 = "Kuyini oku korho kwi isiXhosa yemodeli yeXNLP namhlanje?"
output2 = predictor.generate(prompt2, **gen_params)
print(output2)
print()

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()
print("Model metadata:")
print(f"  Vocab size: {predictor.info()['vocab_size']}")
print(f"  Parameters: {predictor.info()['parameters']:,}")
print(f"  Epoch: {predictor.metadata.get('epoch', 'N/A')}")
print(f"  Best val loss: {predictor.metadata.get('best_val_loss', 'N/A')}")
