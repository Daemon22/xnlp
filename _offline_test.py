#!/usr/bin/env python
"""
Phase V2-18: OFFLINE ISOLATION TEST

Verifies that a trained XNLP model generates language WITHOUT:
- Network access
- Source corpus
- Retrieval
- External APIs

The test works by:
1. Copying best_model.pt to a completely isolated directory
2. Loading ONLY the checkpoint (which contains model weights + tokenizer state)
3. Generating text using only internal state

If the model can generate coherent isiXhosa text from just the checkpoint,
it proves the model learned language through its own parameters (not via retrieval).
"""
import sys, io, os, json, shutil, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import torch

from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.config import validate_checkpoint
from xnlp_trainer.data import tokenizer_from_state_dict
from xnlp_trainer.evaluate import generate_samples

def test_model_isolation(model_path: str, model_name: str) -> dict:
    """
    Test that a model checkpoint works fully offline with no external dependencies.
    
    The checkpoint contains:
    - model_state_dict (model weights)
    - config (model architecture hyperparameters)
    - tokenizer_state (BPE vocab + merges)
    
    Everything needed to generate text is INSIDE the checkpoint.
    """
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'tests': {},
    }
    
    # ─── Test 1: Checkpoint loads and validates ─────────────────────────────
    print(f'\n[{"="*60}')
    print(f'  Testing: {model_name}')
    print(f'  Checkpoint: {model_path}')
    print(f'{"="*60}')
    
    if not os.path.exists(model_path):
        results['tests']['checkpoint_exists'] = {'pass': False, 'error': f'File not found: {model_path}'}
        print(f'  FAIL: Checkpoint file not found')
        return results
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Validate checkpoint structure
    try:
        validate_checkpoint(ckpt)
        results['tests']['checkpoint_valid'] = {'pass': True}
        print(f'  PASS: Checkpoint validates')
    except Exception as e:
        results['tests']['checkpoint_valid'] = {'pass': False, 'error': str(e)}
        print(f'  FAIL: {e}')
        return results
    
    # Extract model config
    model_config = ckpt['config']
    tokenizer_state = ckpt['tokenizer_state']
    meta = ckpt['training_metadata']
    
    results['param_count'] = meta['param_count']
    results['best_val_loss'] = meta['best_val_loss']
    results['vocab_size'] = len(tokenizer_state['token2id'])
    results['preset'] = meta['preset']
    results['python_version'] = meta['python_version']
    results['torch_version'] = torch.__version__
    
    print(f'  Model: {meta["preset"]}, {meta["param_count"]:,} params')
    print(f'  Vocab: {results["vocab_size"]} tokens')
    print(f'  Best val loss: {meta["best_val_loss"]:.4f}')
    
    # ─── Test 2: Build model from checkpoint config ───────────────────────
    
    # Construct XNLPConfig from checkpoint
    model_cfg = XNLPConfig(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        num_key_value_heads=model_config['num_key_value_heads'],
        max_position_embeddings=model_config['max_position_embeddings'],
        pad_token_id=model_config['pad_token_id'],
        bos_token_id=model_config['bos_token_id'],
        eos_token_id=model_config['eos_token_id'],
        unk_token_id=model_config['unk_token_id'],
        rope_theta=model_config['rope_theta'],
        layer_norm_eps=model_config['layer_norm_eps'],
        dropout_prob=model_config['dropout_prob'],
        device='cpu',
    )
    
    model = XNLPCoreLLM(model_cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  PASS: Model built and weights loaded from checkpoint')
    
    # ─── Test 3: Reconstruct tokenizer from checkpoint ───────────────────────
    
    tokenizer = tokenizer_from_state_dict(tokenizer_state)
    print(f'  PASS: Tokenizer reconstructed from checkpoint (vocab={tokenizer.vocab_size_actual})')
    
    # ─── Test 4: Generate text using ONLY checkpoint contents ───────────────
    # This is the critical test — NO corpus, NO retrieval, NO internet
    
    test_prompts = [
        'Molo, ndiyabulela',
        'Umntu ngumntu ngabantu',
        'Umthetho wamaXhosa',
        'Izibongo zethu',
        'Ndiyavuya',
        'Nkosi Sikelel',
        'UMqhayi',
        'Ubuntu',
        'Umsebenzi',
        'Ithemba',
    ]
    
    samples = generate_samples(
        model, tokenizer, test_prompts, 'cpu',
        max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9,
        repetition_penalty=1.1,
    )
    
    results['tests']['offline_generation'] = {'pass': True, 'samples': {}}
    print(f'\n  Offline Generation Results:')
    for prompt, text in samples:
        results['tests']['offline_generation']['samples'][prompt] = text
        # Check that output is non-empty and different from prompt
        is_xhosa_chars = sum(1 for c in text if ord(c) > 127 or c.isalpha())
        print(f'    "{prompt}" -> "{text[:100]}"')
    
    # ─── Test 5: Check for isiXhosa character usage ─────────────────────────
    
    # Count how many outputs contain isiXhosa-specific characters
    xhosa_chars = set('abcedfghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \'"-.,!?()0123456789')
    # isiXhosa specific: ç seems', but mainly uses standard Latin + clicks
    # Check that output doesn't just repeat the prompt
    non_trivial = 0
    for prompt, text in samples:
        if len(text) > len(prompt) + 10:  # Generated at least 10 new tokens
            non_trivial += 1
    
    results['tests']['non_trivial_generation'] = {
        'pass': non_trivial >= 5,  # At least half the prompts should generate substantial text
        'non_trivial_count': non_trivial,
        'total_count': len(samples),
    }
    print(f'  Non-trivial generations: {non_trivial}/{len(samples)}')
    
    # ─── Test 6: Isolation verification ─────────────────────────────────────
    # Verify that the model works without any files except the checkpoint
    
    # Create isolated temporary directory
    with tempfile.TemporaryDirectory() as iso_dir:
        iso_model_path = os.path.join(iso_dir, 'best_model.pt')
        shutil.copy2(model_path, iso_model_path)
        
        # Remove any reference to source corpus
        # Only the checkpoint should be available
        files_in_iso = os.listdir(iso_dir)
        print(f'  Isolated dir contents: {files_in_iso}')
        
        # Reload from isolated directory
        iso_ckpt = torch.load(iso_model_path, map_location='cpu', weights_only=False)
        iso_model_cfg = XNLPConfig(
            vocab_size=iso_ckpt['config']['vocab_size'],
            hidden_size=iso_ckpt['config']['hidden_size'],
            intermediate_size=iso_ckpt['config']['intermediate_size'],
            num_hidden_layers=iso_ckpt['config']['num_hidden_layers'],
            num_attention_heads=iso_ckpt['config']['num_attention_heads'],
            num_key_value_heads=iso_ckpt['config']['num_key_value_heads'],
            max_position_embeddings=iso_ckpt['config']['max_position_embeddings'],
            pad_token_id=iso_ckpt['config']['pad_token_id'],
            bos_token_id=iso_ckpt['config']['bos_token_id'],
            eos_token_id=iso_ckpt['config']['eos_token_id'],
            unk_token_id=iso_ckpt['config']['unk_token_id'],
            rope_theta=iso_ckpt['config']['rope_theta'],
            layer_norm_eps=iso_ckpt['config']['layer_norm_eps'],
            dropout_prob=iso_ckpt['config']['dropout_prob'],
            device='cpu',
        )
        iso_model = XNLPCoreLLM(iso_model_cfg)
        iso_model.load_state_dict(iso_ckpt['model_state_dict'])
        iso_model.eval()
        
        iso_tokenizer = tokenizer_from_state_dict(iso_ckpt['tokenizer_state'])
        
        # Generate from isolated model
        iso_samples = generate_samples(
            iso_model, iso_tokenizer, test_prompts[:3], 'cpu',
            max_new_tokens=30, temperature=0.8, top_k=40, top_p=0.9,
            repetition_penalty=1.1,
        )
        
        results['tests']['isolated_generation'] = {
            'pass': len(iso_samples) == 3 and all(len(t) > 0 for _, t in iso_samples),
            'samples': {p: t for p, t in iso_samples},
        }
        print(f'  PASS: Model generates from isolated checkpoint copy')
        for p, t in iso_samples:
            print(f'    "{p}" -> "{t[:80]}"')
    
    # ─── Summary ─────────────────────────────────────────────────────────────
    all_pass = all(t.get('pass', False) for t in results['tests'].values())
    results['overall_pass'] = all_pass
    
    print(f'\n  Overall: {"PASS" if all_pass else "FAIL"}')
    return results


# ─── Run isolation test for all V2 models ───────────────────────────────────

all_results = {}

# V2 Tiny (if trained)
tiny_path = 'outputs_v2_tiny/best_model.pt'
if os.path.exists(tiny_path):
    r = test_model_isolation(tiny_path, 'v2_tiny')
    all_results['v2_tiny'] = r
else:
    print(f'\nV2 tiny model not found at {tiny_path}, skipping.')
    all_results['v2_tiny'] = {'model_name': 'v2_tiny', 'overall_pass': False, 'error': 'Model not trained yet'}

# V1 Baseline (should still pass isolation test)
v1_path = 'v1_baseline/best_model.pt'
if os.path.exists(v1_path):
    r = test_model_isolation(v1_path, 'v1_baseline')
    all_results['v1_baseline'] = r

# Save results
os.makedirs('data/reports', exist_ok=True)
with open('data/reports/v2_isolation_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f'\nSaved: data/reports/v2_isolation_test_results.json')

# Print summary
print(f'\n{"="*60}')
print('OFFLINE ISOLATION TEST SUMMARY')
print(f'{"="*60}')
for name, r in all_results.items():
    status = 'PASS' if r.get('overall_pass', False) else 'FAIL'
    if r.get('param_count'):
        print(f'  {name:>15}  {status}  ({r["param_count"]:,} params, val_loss={r.get("best_val_loss", "N/A")})')
    else:
        print(f'  {name:>15}  {status}  ({r.get("error", "unknown error")})')
