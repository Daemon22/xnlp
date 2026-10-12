#!/usr/bin/env python3
"""
Test if using the clean tokenizer (without digit tokens) improves generation.
This is a quick check before full retraining.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

print("=" * 80)
print("TEST: Clean Tokenizer with Existing Model")
print("=" * 80)
print()

import torch
import json
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from xnlp_trainer.data import tokenizer_from_state_dict

# Load the clean tokenizer
with open(r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer\tokenizer_state.json', 'r', encoding='utf-8') as f:
    clean_tokenizer_state = json.load(f)

clean_tokenizer = tokenizer_from_state_dict(clean_tokenizer_state)
print(f"Clean tokenizer: vocab={clean_tokenizer.vocab_size}, 0 digit tokens")
print()

# Load the existing best model
ckpt = torch.load(
    r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt',
    map_location='cpu',
    weights_only=False
)

# Get model config from checkpoint
model_config_dict = ckpt['config']
model_config = XNLPConfig(
    vocab_size=clean_tokenizer.vocab_size,  # Use clean tokenizer vocab size
    hidden_size=model_config_dict['hidden_size'],
    intermediate_size=model_config_dict['intermediate_size'],
    num_hidden_layers=model_config_dict['num_hidden_layers'],
    num_attention_heads=model_config_dict['num_attention_heads'],
    num_key_value_heads=model_config_dict['num_key_value_heads'],
    max_position_embeddings=model_config_dict['max_position_embeddings'],
)

# Create model
model = XNLPCoreLLM(model_config)
model.load_state_dict(ckpt['model_state_dict'])
model.to('cpu')
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Model expects vocab size: {model_config.vocab_size}")
print(f"Clean tokenizer vocab size: {clean_tokenizer.vocab_size}")
print()

# Test generation
test_prompts = [
    "Molo. ",
    "Namhlanje, ",
    "Ndiyakwazi ukuthetha isiXhosa. ",
]

print("Testing generation with clean tokenizer + existing model weights:")
print("-" * 80)

for prompt in test_prompts:
    # Encode
    ids = clean_tokenizer.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device='cpu')
    
    # Generate
    model.eval()
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=60,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
        )
    
    # Decode
    output = clean_tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print()

print("=" * 80)
print("CONCLUSION:")
print("If generation improves, the issue was the tokenizer.")
print("If generation is still bad, we need to retrain the model.")
print("=" * 80)
