#!/usr/bin/env python
"""
Phase V2-19: FINAL MODEL EVALUATION

Runs all capability benchmarks (grammar, spelling, generation, memorization)
against V2 trained models. Produces SEPARATED ENGINEERING/LANGUAGE/
GENERALIZATION/MEMORIZATION scores per the V2 release requirements.

Tests work entirely offline — loads model from checkpoint, no retrieval.
"""
import sys, io, os, json, time, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import torch

from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.config import validate_checkpoint
from xnlp_trainer.data import tokenizer_from_state_dict
from xnlp_trainer.evaluate import compute_perplexity

def load_model(model_path):
    """Load a complete model + tokenizer from a single checkpoint file."""
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    validate_checkpoint(ckpt)
    
    cfg = ckpt['config']
    model_cfg = XNLPConfig(
        vocab_size=cfg['vocab_size'],
        hidden_size=cfg['hidden_size'],
        intermediate_size=cfg['intermediate_size'],
        num_hidden_layers=cfg['num_hidden_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        num_key_value_heads=cfg['num_key_value_heads'],
        max_position_embeddings=cfg['max_position_embeddings'],
        pad_token_id=cfg['pad_token_id'],
        bos_token_id=cfg['bos_token_id'],
        eos_token_id=cfg['eos_token_id'],
        unk_token_id=cfg['unk_token_id'],
        rope_theta=cfg['rope_theta'],
        layer_norm_eps=cfg['layer_norm_eps'],
        dropout_prob=cfg['dropout_prob'],
        device='cpu',
    )
    model = XNLPCoreLLM(model_cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    tokenizer = tokenizer_from_state_dict(ckpt['tokenizer_state'])
    
    meta = ckpt['training_metadata']
    return model, tokenizer, meta


def evaluate_grammar(model, tokenizer, benchmark_path):
    """V2-12: Grammar benchmark — minimal pairs."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        bench = json.load(f)
    
    results = {}
    correct = 0
    total = 0
    category_scores = {}
    
    for category, tests in bench.items():
        if isinstance(tests, dict):
            tests_list = tests.get('tests', tests.get('pairs', []))
        elif isinstance(tests, list):
            tests_list = tests
        else:
            continue
        
        cat_correct = 0
        cat_total = 0
        details = []
        
        for test_item in tests_list:
            if isinstance(test_item, dict):
                prompt = test_item.get('prompt', '')
                correct_text = test_item.get('correct', test_item.get('correct_answer', ''))
                wrong_text = test_item.get('incorrect', test_item.get('wrong_answer', ''))
                expected = test_item.get('expected', '')
            elif isinstance(test_item, list) and len(test_item) >= 2:
                prompt = test_item[0]
                correct_text = test_item[1] if len(test_item) > 1 else ''
                wrong_text = test_item[2] if len(test_item) > 2 else ''
            else:
                continue
            
            # Score both options by model log-probability
            try:
                input_correct = tokenizer.encode(correct_text, add_special_tokens=False)
                input_wrong = tokenizer.encode(wrong_text, add_special_tokens=False)
                
                if not input_correct or not input_wrong:
                    cat_total += 1
                    total += 1
                    continue
                
                input_correct = torch.tensor([input_correct], dtype=torch.long)
                input_wrong = torch.tensor([input_wrong], dtype=torch.long)
                
                with torch.no_grad():
                    logits_correct = model(input_correct).logits
                    logits_wrong = model(input_wrong).logits
                    
                    # Compute log-prob of each token in the sequence
                    log_probs_c = torch.log_softmax(logits_correct[0], dim=-1)
                    log_probs_w = torch.log_softmax(logits_wrong[0], dim=-1)
                    
                    # Average log-probability of the sequence
                    score_c = sum(log_probs_c[i, input_correct[0, i+1]].item() for i in range(len(input_correct[0])-1)) / max(len(input_correct[0])-1, 1)
                    score_w = sum(log_probs_w[i, input_wrong[0, i+1]].item() for i in range(len(input_wrong[0])-1)) / max(len(input_wrong[0])-1, 1)
                
                is_correct = score_c > score_w
                if is_correct:
                    correct += 1
                    cat_correct += 1
                total += 1
                cat_total += 1
                
                details.append({
                    'prompt': prompt,
                    'correct': correct_text,
                    'correct_score': round(score_c, 4),
                    'incorrect': wrong_text,
                    'incorrect_score': round(score_w, 4),
                    'model_chose_correct': is_correct,
                })
            except Exception as e:
                details.append({'prompt': prompt, 'error': str(e)})
                cat_total += 1
                total += 1
        
        category_scores[category] = {
            'correct': cat_correct,
            'total': cat_total,
            'accuracy': round(cat_correct / max(cat_total, 1), 4),
            'details': details,
        }
    
    return {
        'total_correct': correct,
        'total_items': total,
        'overall_accuracy': round(correct / max(total, 1), 4),
        'categories': category_scores,
    }


def evaluate_spelling(model, tokenizer, benchmark_path):
    """V2-13: Spelling benchmark — correct vs misspelling discrimination."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        bench = json.load(f)
    
    tests = bench.get('tests', bench.get('spelling_tests', []))
    correct = 0
    total = 0
    details = []
    
    for test_item in tests:
        correct_word = test_item.get('correct', '')
        misspelling = test_item.get('misspelling', '')
        category = test_item.get('category', 'unknown')
        
        try:
            input_correct = tokenizer.encode(correct_word, add_special_tokens=False)
            input_wrong = tokenizer.encode(misspelling, add_special_tokens=False)
            
            if not input_correct or not input_wrong:
                total += 1
                details.append({'correct': correct_word, 'misspelling': misspelling, 'error': 'empty encoding'})
                continue
            
            input_correct = torch.tensor([input_correct], dtype=torch.long)
            input_wrong = torch.tensor([input_wrong], dtype=torch.long)
            
            with torch.no_grad():
                logits_c = model(input_correct).logits
                logits_w = model(input_wrong).logits
                
                log_probs_c = torch.log_softmax(logits_c[0], dim=-1)
                log_probs_w = torch.log_softmax(logits_w[0], dim=-1)
                
                score_c = sum(log_probs_c[i, input_correct[0, i+1]].item() for i in range(len(input_correct[0])-1)) / max(len(input_correct[0])-1, 1)
                score_w = sum(log_probs_w[i, input_wrong[0, i+1]].item() for i in range(len(input_wrong[0])-1)) / max(len(input_wrong[0])-1, 1)
            
            is_correct = score_c > score_w
            if is_correct:
                correct += 1
            total += 1
            details.append({
                'correct': correct_word, 'misspelling': misspelling, 'category': category,
                'correct_score': round(score_c, 4), 'misspelling_score': round(score_w, 4),
                'model_chose_correct': is_correct,
            })
        except Exception as e:
            total += 1
            details.append({'correct': correct_word, 'misspelling': misspelling, 'error': str(e)})
    
    return {
        'correct': correct,
        'total': total,
        'accuracy': round(correct / max(total, 1), 4),
        'details': details,
    }


def evaluate_generation(model, tokenizer, benchmark_path):
    """V2-14: Generation benchmark — novel prompts with scoring criteria."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        bench = json.load(f)
    
    prompts = bench.get('prompts', bench.get('generation_tests', []))
    results = []
    
    for prompt_item in prompts:
        if isinstance(prompt_item, dict):
            prompt = prompt_item.get('prompt', '')
            criteria = prompt_item.get('criteria', prompt_item.get('scoring_criteria', []))
            language_type = prompt_item.get('type', 'creative')
        else:
            prompt = str(prompt_item)
            criteria = []
            language_type = 'creative'
        
        # Generate text
        from xnlp_trainer.evaluate import generate_samples
        samples = generate_samples(
            model, tokenizer, [prompt], 'cpu',
            max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9,
            repetition_penalty=1.1,
        )
        
        text = samples[0][1] if samples else ''
        
        # Compute metrics
        word_count = len(text.split())
        xhosa_ratio = sum(1 for c in text if 'a' <= c.lower() <= 'z' or ord(c) > 127) / max(len(text), 1)
        avg_word_len = sum(len(w) for w in text.split()) / max(word_count, 1)
        unique_words = len(set(text.split()))
        repetition_ratio = (word_count - unique_words) / max(word_count, 1) if word_count > 0 else 1
        
        # Scoring: lower repetition = better, more words = better, some xhosa chars
        fluency_score = max(0, 1.0 - repetition_ratio)  # 0-1, lower repetition = higher score
        coherence_score = min(1.0, word_count / 20)  # 0-1, more words = higher score
        diversity_score = unique_words / max(word_count, 1)  # 0-1
        
        overall_score = (fluency_score * 0.3 + coherence_score * 0.4 + diversity_score * 0.3)
        
        results.append({
            'prompt': prompt,
            'type': language_type,
            'generated_text': text,
            'word_count': word_count,
            'avg_word_length': round(avg_word_len, 2),
            'unique_words': unique_words,
            'repetition_ratio': round(repetition_ratio, 3),
            'fluency_score': round(fluency_score, 3),
            'coherence_score': round(coherence_score, 3),
            'diversity_score': round(diversity_score, 3),
            'overall_score': round(overall_score, 3),
            'criteria': criteria,
        })
    
    avg_score = sum(r['overall_score'] for r in results) / max(len(results), 1)
    return {
        'num_prompts': len(results),
        'avg_overall_score': round(avg_score, 3),
        'results': results,
    }


def evaluate_memorization(model_path, corpus_path, tokenizer):
    """V2-15: Memorization test — exact + near-duplicate detection."""
    results = {
        'model_path': model_path,
        'corpus_path': corpus_path,
        'method': 'exact_match_and_near_duplicate_detection',
    }
    
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Check if any generated samples exactly match corpus lines
    from xnlp_trainer.evaluate import generate_samples
    test_prompts = ['Molo', 'Umntu', 'Umthetho', 'Izibongo', 'Ndiyavuya', 'Nkosi', 'Ubuntu', 'IManye']
    
    samples = generate_samples(
        model, tokenizer, test_prompts, 'cpu',
        max_new_tokens=80, temperature=0.8, top_k=40, top_p=0.9,
        repetition_penalty=1.1,
    )
    
    exact_matches = 0
    near_matches = 0
    details = []
    
    for prompt, text in samples:
        lines_generated = text.split('\n')
        for line in lines_generated:
            line = line.strip()
            if not line:
                continue
            # Exact match
            if line in corpus_lines:
                exact_matches += 1
                details.append({'type': 'exact', 'text': line[:100], 'prompt': prompt})
                continue
            # Near match (high Jaccard similarity)
            best_sim = 0
            best_match = None
            line_words = set(line.split())
            for cl in corpus_lines:
                cl_words = set(cl.split())
                if not line_words or not cl_words:
                    continue
                intersection = len(line_words & cl_words)
                union = len(line_words | cl_words)
                if union > 0:
                    sim = intersection / union
                    if sim > best_sim:
                        best_sim = sim
                        best_match = cl
            
            if best_sim > 0.8:
                near_matches += 1
                details.append({
                    'type': 'near_duplicate', 'text': line[:100],
                    'corpus_match': best_match[:100],
                    'similarity': round(best_sim, 3),
                    'prompt': prompt,
                })
    
    total_lines = sum(len(t.split('\n')) for _, t in samples)
    results['total_generated_lines'] = total_lines
    results['exact_matches'] = exact_matches
    results['near_matches'] = near_matches
    results['exact_match_rate'] = round(exact_matches / max(total_lines, 1), 4)
    results['near_match_rate'] = round(near_matches / max(total_lines, 1), 4)
    results['details'] = details[:20]  # Cap at 20 examples
    
    return results


# ─── Run all benchmarks ─────────────────────────────────────────────────────

def run_all_benchmarks(model_path, model_name):
    """Run all V2 capability benchmarks for a single model."""
    print(f'\n{"="*60}')
    print(f'  Evaluating: {model_name}')
    print(f'  Checkpoint: {model_path}')
    print(f'{"="*60}')
    
    if not os.path.exists(model_path):
        return {'model_name': model_name, 'error': f'Model not found: {model_path}'}
    
    start = time.time()
    
    # Load model
    print('  Loading model...')
    model, tokenizer, meta = load_model(model_path)
    print(f'  Loaded: {meta["param_count"]:,} params, vocab={len(tokenizer.token2id)}')
    
    results = {
        'model_name': model_name,
        'param_count': meta['param_count'],
        'vocab_size': len(tokenizer.token2id),
        'best_val_loss': meta['best_val_loss'],
        'perplexity': round(compute_perplexity(meta['best_val_loss']), 1),
        'preset': meta['preset'],
        'tests': {},
    }
    
    # V2-12: Grammar benchmark
    print('  Running grammar benchmark (minimal pairs)...')
    results['tests']['grammar'] = evaluate_grammar(
        model, tokenizer, 'data/benchmarks/grammar_benchmark.json')
    
    # V2-13: Spelling benchmark
    print('  Running spelling benchmark (discrimination)...')
    results['tests']['spelling'] = evaluate_spelling(
        model, tokenizer, 'data/benchmarks/spelling_benchmark.json')
    
    # V2-14: Generation benchmark
    print('  Running generation benchmark (novel prompts)...')
    results['tests']['generation'] = evaluate_generation(
        model, tokenizer, 'data/benchmarks/generation_benchmark.json')
    
    # V2-15: Memorization test
    print('  Running memorization test...')
    results['tests']['memorization'] = evaluate_memorization(
        model_path, 'data/processed/v2/training_corpus.txt', tokenizer)
    
    elapsed = time.time() - start
    results['evaluation_time_seconds'] = round(elapsed, 1)
    
    # Compute separated scores per V2-19 requirement
    grammar_acc = results['tests']['grammar']['overall_accuracy']
    spelling_acc = results['tests']['spelling']['accuracy']
    gen_score = results['tests']['generation']['avg_overall_score']
    mem_exact = results['tests']['memorization']['exact_match_rate']
    mem_near = results['tests']['memorization']['near_match_rate']
    
    results['separated_scores'] = {
        'LANGUAGE_score': round(grammar_acc, 4),  # Grammar = core language competency
        'LANGUAGE_label': 'Grammar minimal pairs accuracy',
        'GENERALIZATION_score': round((1.0 - mem_near) * 0.5 + gen_score * 0.5, 4),
        'GENERALIZATION_label': 'Novel generation + low memorization',
        'ENGINEERING_score': round(1.0 - results['tests']['memorization']['exact_match_rate'], 4),
        'ENGINEERING_label': 'Novelty (1 - exact memorization rate)',
        'MEMORIZATION_score': round(mem_exact + mem_near, 4),
        'MEMORIZATION_label': 'Exact + near-duplicate match rate (lower is better)',
    }
    
    print(f'\n  Separated Scores:')
    for k, v in results['separated_scores'].items():
        if 'label' not in k:
            print(f'    {k}: {v}')
    
    return results


# ─── Main ───────────────────────────────────────────────────────────────────

all_results = {}

# V2 Tiny
tiny_path = 'outputs_v2_tiny/best_model.pt'
if os.path.exists(tiny_path):
    all_results['v2_tiny'] = run_all_benchmarks(tiny_path, 'v2_tiny')

# V2 Small (if trained)
small_path = 'outputs_v2_small/best_model.pt'
if os.path.exists(small_path):
    all_results['v2_small'] = run_all_benchmarks(small_path, 'v2_small')

# V1 Baseline (for comparison)
v1_path = 'v1_baseline/best_model.pt'
if os.path.exists(v1_path):
    all_results['v1_baseline'] = run_all_benchmarks(v1_path, 'v1_baseline')

# Save results
os.makedirs('data/reports', exist_ok=True)
with open('data/reports/v2_capability_benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f'\n\nSaved: data/reports/v2_capability_benchmark_results.json')

# Print comparison table
print(f'\n{"="*70}')
print('V2 CAPABILITY BENCHMARK — Separated Scores')
print(f'{"="*70}')
print(f'{"Model":>12} {"Params":>12} {"Val Loss":>10} {"PPL":>8} {"Grammar":>8} {"Spelling":>9} {"Generation":>11} {"Memorization":>13}')
print('-' * 80)
for name, r in all_results.items():
    if 'error' in r:
        print(f'{"":>12} {name:>12}  ERROR: {r["error"]}')
        continue
    s = r['separated_scores']
    print(f'{name:>12} {r["param_count"]:>12,} {r["best_val_loss"]:>10.4f} {r["perplexity"]:>8.1f} '
          f'{s["LANGUAGE_score"]:>8.4f} {r["tests"]["spelling"]["accuracy"]:>9.4f} '
          f'{s["GENERALIZATION_score"]:>11.4f} {s["MEMORIZATION_score"]:>13.4f}')
print(f'{"V1 baseline":>12} {"4,321,280":>12} {"4.7701":>10} {"117.9":>8} {"V1":>8}')
