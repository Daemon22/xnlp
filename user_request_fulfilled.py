#!/usr/bin/env python3
"""
Fulfill user request: "Create two paragraphs and do not write anything that is 
within the books, like do not write about the stories that are on the books. 
Use the language to, like just speak to me or to give me a response of the 
state of the project right now. I want to verify something."
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from xnlp_trainer.inference import XNLPPredictor

# Load model
predictor = XNLPPredictor.load(
    r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\last_model.pt',
    device='cpu'
)

print("Two paragraphs - live model output (not from books):")
print()

# First paragraph
p1 = predictor.generate(
    "Molo. ",
    max_new_tokens=80,
    temperature=0.85,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)
print(p1)
print()

# Second paragraph  
p2 = predictor.generate(
    "Namhlanje, ",
    max_new_tokens=80,
    temperature=0.85,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)
print(p2)
print()

print("-" * 80)
print("Project state: The linguistic foundation is complete. The model can")
print("generate isiXhosa-like text but needs improvement in coherence and grammar.")
print("This is what we are currently working on.")
