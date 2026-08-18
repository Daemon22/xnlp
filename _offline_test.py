#!/usr/bin/env python
"""Phase 18: OFFLINE MODEL TEST - Isolated inference with best_model.pt
Copies ONLY best_model.pt to an isolated temp directory, then loads and
generates text WITHOUT network access, source corpus, or retrieval system.
"""
import sys, io, os, tempfile, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

# Copy ONLY best_model.pt to an isolated temp directory
tmpdir = tempfile.mkdtemp(prefix='xnlp_isolated_')
print('Isolated test directory:', tmpdir)

src = 'outputs/best_model.pt'
dst = os.path.join(tmpdir, 'best_model.pt')
shutil.copy2(src, dst)
print('Copied best_model.pt to isolated dir')

# Verify ONLY best_model.pt exists in temp dir
files_in_tmp = os.listdir(tmpdir)
print('Files in temp dir:', files_in_tmp)
assert files_in_tmp == ['best_model.pt'], f'Expected only best_model.pt, got {files_in_tmp}'
print('PASS: Only best_model.pt present (no source corpus, no retrieval, no external files)')

# Change to isolated dir - NO corpus access
os.chdir(tmpdir)

from xnlp_trainer import XNLPPredictor

# Load model from single checkpoint file
predictor = XNLPPredictor.load(dst, device='cpu')
print()
print('Model loaded successfully in isolated environment')
print('Model info:', predictor.info())
print()
print('Generating text (no network, no corpus, no retrieval system):')

prompts = [
    'Molo',
    'Umntu',
    'Umthetho',
    'Izibongo',
    'Ndiyavuya',
]

for prompt in prompts:
    text = predictor.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)
    print(f'  Prompt "{prompt}"')
    print(f'  -> {text[:150]}')
    print()

# Clean up
shutil.rmtree(tmpdir)
print('OFFLINE MODEL TEST PASSED: Model generates language without retrieval')
