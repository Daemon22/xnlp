#!/usr/bin/env python
"""Phase V2-1: PRESERVE THE CURRENT BASELINE as XNLP Baseline V1

Archives:
  - best_model.pt (the trained model)
  - training corpus + provenance
  - tokenizer state
  - training metrics (loss, perplexity, epochs, time)
  - evaluation results
  - generation examples
"""
import sys, io, os, shutil, json, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

# ─── V1 Baseline Archive ──────────────────────────────────────────────────

archive_dir = 'v1_baseline'
os.makedirs(archive_dir, exist_ok=True)

# 1. Archive best_model.pt
src_ckpt = 'outputs/best_model.pt'
dst_ckpt = os.path.join(archive_dir, 'best_model.pt')
if os.path.exists(src_ckpt):
    shutil.copy2(src_ckpt, dst_ckpt)
    ckpt = torch.load(dst_ckpt, map_location='cpu', weights_only=False)
    meta = ckpt['training_metadata']
    print(f'[1] best_model.pt archived: {os.path.getsize(dst_ckpt)/1024/1024:.1f}MB')
else:
    print('[1] best_model.pt NOT FOUND')
    sys.exit(1)

# 2. Archive training corpus + provenance
for fname in ['training_corpus.txt', 'training_corpus_provenance.jsonl']:
    src = os.path.join('data/processed', fname)
    dst = os.path.join(archive_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'[2] {fname} archived')

# 3. Archive evaluation report
eval_src = 'data/reports/evaluation_report.json'
eval_dst = os.path.join(archive_dir, 'evaluation_report.json')
if os.path.exists(eval_src):
    shutil.copy2(eval_src, eval_dst)
    print(f'[3] evaluation_report.json archived')

# 4. Archive dataset reports
for report in ['dataset_report_phase1.json', 'dataset_report_phase12.json']:
    src = os.path.join('data/reports', report)
    if os.path.exists(src):
        dst = os.path.join(archive_dir, report)
        shutil.copy2(src, dst)
        print(f'[4] {report} archived')

# 5. Record baseline metrics
with open('data/processed/training_corpus_provenance.jsonl', 'r', encoding='utf-8') as f:
    prov = [json.loads(line) for line in f]

# Count tokens
tok = ckpt['tokenizer_state']
total_tokens = 0
total_words = 0
for rec in prov:
    text = rec.get('text', '')
    tokens = tok['token2id']
    # Simple token count
    words = text.split()
    total_words += len(words)

# Load training history
history_path = 'outputs/training_history.json'
history = {}
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)

baseline_metrics = {
    'name': 'XNLP Baseline V1',
    'description': 'First fully trained XNLP model from authoritative corpus',
    'model': {
        'preset': meta.get('preset'),
        'param_count': meta.get('param_count'),
        'vocab_size': meta.get('vocab_size'),
        'val_loss': meta.get('best_val_loss'),
        'best_epoch': meta.get('epoch'),
        'global_step': meta.get('global_step'),
        'python_version': meta.get('python_version'),
        'torch_version': meta.get('torch_version'),
        'device': meta.get('device'),
        'training_start_time': meta.get('training_start_time'),
        'training_end_time': meta.get('training_end_time'),
    },
    'dataset': {
        'total_records': len(prov),
        'mqhayi_records': sum(1 for r in prov if r.get('source') == 'mqhayi'),
        'masikhanyise_records': sum(1 for r in prov if r.get('source') == 'masikhanyise'),
        'generated_records': sum(1 for r in prov if r.get('source') == 'generated'),
        'authoritative_records': sum(1 for r in prov if r.get('validation_status') == 'authoritative'),
        'validated_records': sum(1 for r in prov if r.get('validation_status') == 'validated'),
        'total_words': total_words,
        'unique_words': len(set(w.lower() for r in prov for w in r.get('text', '').split())),
    },
    'training_history': {
        'train_losses': history.get('train_loss', []),
        'val_losses': history.get('val_loss', []),
        'learning_rates': history.get('lr', []),
        'epochs': history.get('epoch', []),
    },
    'evaluation': json.load(open(eval_dst, 'r', encoding='utf-8')) if os.path.exists(eval_dst) else {},
    'generation_examples': [],
}

# 6. Generate and archive sample outputs
from xnlp_trainer.inference import XNLPPredictor
predictor = XNLPPredictor.load(dst_ckpt, device='cpu')
prompts = ['Molo', 'Umntu', 'Umthetho', 'Izibongo', 'Inkosi', 'Ndiyavuya', 'Ubuntu', 'Imbongi']
for prompt in prompts:
    gen = predictor.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)
    baseline_metrics['generation_examples'].append({'prompt': prompt, 'output': gen})

# 7. Compute perplexity
import math
perplexity = math.exp(meta['best_val_loss'])
baseline_metrics['model']['perplexity'] = perplexity

# 8. Save baseline manifest
manifest_path = os.path.join(archive_dir, 'baseline_manifest.json')
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)

print(f'\n[5] Baseline manifest saved to {manifest_path}')
print(json.dumps(baseline_metrics, indent=2, ensure_ascii=False)[:2000])
print(f'\n=== V1 Baseline Preserved ===')
print(f'Model: {meta["preset"]} preset, {meta["param_count"]:,} params')
print(f'Val loss: {meta["best_val_loss"]:.4f}, Perplexity: {perplexity:.1f}')
print(f'Dataset: {len(prov)} records ({baseline_metrics["dataset"]["mqhayi_records"]} Mqhayi + {baseline_metrics["dataset"]["masikhanyise_records"]} Masikhanyise)')
print(f'Vocab: {meta["vocab_size"]} tokens')
