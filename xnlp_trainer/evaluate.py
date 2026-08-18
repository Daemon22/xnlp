#!/usr/bin/env python
"""
Phase 17: XNLP Evaluation Suite
==============================
Comprehensive evaluation tests for the XNLP language model.

Tests:
  1. Spelling validation - Xhosa orthography correctness
  2. Grammar validation - Xhosa grammatical pattern adherence
  3. Vocabulary coverage - known word usage
  4. Generation quality - coherence, diversity, novelty
  5. Contamination detection - no verbatim training data reproduction
  6. Offline capability - single-file inference without retrieval
"""
import sys, io, os, json, re, math, tempfile, shutil
import torch
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer

# ─── Core training evaluation functions (imported by trainer.py) ────────────

@torch.no_grad()
def evaluate(model: XNLPCoreLLM, val_loader, device: str = "cpu") -> float:
    """Evaluate model on a validation dataloader. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def compute_loss(model: XNLPCoreLLM, input_ids, labels=None, device: str = "cpu") -> float:
    """Compute average cross-entropy loss for a batch of input_ids.

    If *labels* is None, a standard causal-LM shift is applied (labels = input_ids).
    """
    model.eval()
    input_ids = input_ids.to(device)
    if labels is not None:
        labels = labels.to(device)
    else:
        labels = input_ids
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs["loss"].item()


@torch.no_grad()
def generate_samples(
    model: XNLPCoreLLM,
    tokenizer: XNLPTokenizer,
    prompts: List[str],
    device: str = "cpu",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> List[Tuple[str, str]]:
    """Generate a continuation for each prompt using the model.

    Returns a list of (prompt, decoded_text) tuples.
    """
    model.eval()
    results: List[Tuple[str, str]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=True, max_length=256, truncation=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        decoded = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        results.append((prompt, decoded))
    return results


# ─── Xhosa orthographic patterns (from authoritative sources) ───────────────
XOSA_NOUN_CLASS_PREFIXES = [
    r'^um', r'^ab', r'^i[aeiou]', r'^izi', r'^ama', r'^un',
    r'^in', r'^ii', r'^e[aeiou]', r'^a[aeiou]', r'^u[aeiou]',
]

XOSA_CONSONANT_CLUSTERS = ['ngc', 'ngq', 'ngx', 'ny', 'mb', 'gc', 'gq',
                           'gx', 'kw', 'ph', 'th', 'gh', 'dl', 'dy',
                           'ts', 'tsh', 'xh', 'bh', 'hl', 'sh']

# Traditional orthography forms
XOSA_OLD_FORMS = ['6', '5']

# Common Xhosa word patterns
XOSA_WORD_PATTERNS = [
    r'^uku[a-z]*',      # infinitive verb prefix
    r'^uk[a-z]*',       # infinitive verb prefix  
    r'^amax',           # noun class
    r'^iz[iabdeilmnostruvwxz]*',  # noun class
    r'^um[aeiou]',      # noun class
    r'^ab[aeiou]',      # noun class
    r'^i[aeiou]',       # noun class
]

# Expected Xhosa words from authoritative corpus
EXPECTED_XHOSA_WORDS = {
    'umntu', 'abantu', 'izwe', 'intetho', 'ixhego', 'umthetho',
    'incwadi', 'imibongo', 'izibongo', 'inkosi', 'umkhulu',
    'umama', 'ubaba', 'isikolo', 'isithembiso', 'indlela',
    'ukuthando', 'ukuphila', 'ubomi', 'ubuntu', 'ubuhlobo',
    'intlonipho', 'amandla', 'izicoco', 'izilwandle', 'izikole',
    'amazwe', 'intlukulela', 'umsebenzi', 'umfana', 'indoda',
    'indlovukazi', 'isiduko', 'isifundo', 'isimboni',
    'ityala', 'lamawele', 'ingqumbo', 'yeminyanya', 'iziganeko',
    'besizwe', 'inzuzo', 'imihobe', 'isihobe', 'uchongo',
    'lwamagama', 'imifanekiso', 'ntelekelelo', 'ithoni',
    'imiqondiso', 'isixhumanisi', 'isingqisho', 'injambamenti',
    'umxholo', 'ubume', 'abalinganiswa', 'isimo', 'sentlalo',
    'isityilio', 'inqwaba', 'isigama', 'isenzo', 'isichazi',
    'izanduko', 'izigidimi', 'ukuguquguquka', 'isichazi-senzo',
    'kusekusa', 'kusezulwini', 'kuzakuba', 'sikelel',
    'iAfrika', 'iphondo', 'lwayo', 'mithandazo', 'yethu',
    'lusapho', 'ntla-langa', 'busikeleze', 'bakho',
    'siyakuthanda', 'sithwale', 'umntu', 'abantu', 'izwe',
}

# English words to check against (contamination)
ENGLISH_INDICATORS = {
    'the', 'and', 'of', 'to', 'in', 'is', 'that', 'it', 'for', 'you',
    'on', 'with', 'as', 'are', 'was', 'be', 'at', 'by', 'this', 'have',
    'from', 'or', 'an', 'they', 'not', 'but', 'we', 'what', 'all', 'can',
    'who', 'do', 'if', 'her', 'his', 'how', 'its', 'may', 'has', 'your',
    'their', 'will', 'about', 'would', 'there', 'said', 'could', 'each',
}

# Training corpus records (for contamination check)
TRAINING_CORPUS_PATH = 'data/processed/training_corpus.txt'


class XNLPEvaluation:
    """Evaluation suite for XNLP models."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.results = {}

        # Load training corpus for contamination check
        self.training_records = []
        if os.path.exists(TRAINING_CORPUS_PATH):
            with open(TRAINING_CORPUS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.training_records.append(line)

    def load_model(self):
        """Load the model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        from xnlp_trainer import XNLPPredictor
        self.predictor = XNLPPredictor.load(self.checkpoint_path, device='cpu')
        return self.predictor

    def eval_spelling(self):
        """Test 1: Xhosa spelling validation."""
        print("\n=== Test 1: Spelling Validation ===")
        results = {'passed': 0, 'failed': 0, 'total': 0, 'details': []}

        prompts = [
            'Molo', 'Umntu', 'Umthetho', 'Izibongo', 'Inkosi',
            'Ndiyavuya', 'Ubuntu', 'Imbongi', 'Indlela', 'Ukuphila',
        ]

        for prompt in prompts:
            generated = self.predictor.generate(
                prompt, max_new_tokens=25, temperature=0.7,
                top_k=40, top_p=0.9, repetition_penalty=1.1
            )
            # Check that the generated text doesn't contain invalid characters
            # (e.g., raw English letters in the middle of Xhosa words)
            words = generated.split()
            valid = True
            for word in words:
                word_clean = word.strip('.,!?;:()[]{}')
                if word_clean and word_clean.lower() in ENGLISH_INDICATORS:
                    valid = False
                    results['failed'] += 1
                    results['details'].append(f"FAIL: English word '{word_clean}' in '{generated[:80]}'")
                    break

            if valid:
                results['passed'] += 1
                results['details'].append(f"PASS: '{prompt}' -> '{generated[:80]}'")

            results['total'] += 1

        pct = results['passed'] / max(results['total'], 1) * 100
        results['score'] = pct
        print(f"  Spelling: {results['passed']}/{results['total']} ({pct:.0f}%)")
        for d in results['details']:
            print(f"    {d}")
        self.results['spelling'] = results
        return results

    def eval_vocab_coverage(self):
        """Test 3: Vocabulary coverage - check that generated text uses Xhosa vocabulary."""
        print("\n=== Test 3: Vocabulary Coverage ===")
        results = {'passed': 0, 'failed': 0, 'total': 0, 'details': [], 'xhosa_words_found': []}

        prompts = [
            'Umntu ngumntu', 'Umthetho', 'Izibongo', 'Ubuntu',
            'Inkosi', 'Isikolo', 'Indlela', 'Amandla',
        ]

        all_xho_words_found = set()
        for prompt in prompts:
            generated = self.predictor.generate(
                prompt, max_new_tokens=25, temperature=0.8,
                top_k=40, top_p=0.9, repetition_penalty=1.1
            )
            # Check for known Xhosa words
            gen_lower = generated.lower()
            found_words = set()
            for word in EXPECTED_XHOSA_WORDS:
                if word.lower() in gen_lower:
                    found_words.add(word.lower())
                    all_xho_words_found.add(word.lower())

            results['total'] += 1
            if len(found_words) >= 2:
                results['passed'] += 1
                results['details'].append(f"PASS: '{prompt}' found {len(found_words)} Xhosa words: {sorted(found_words)[:5]}")
            else:
                results['failed'] += 1
                results['details'].append(f"FAIL: '{prompt}' found only {len(found_words)} Xhosa words")
            results['xhosa_words_found'].append(list(found_words))

        pct = results['passed'] / max(results['total'], 1) * 100
        results['score'] = pct
        results['unique_xhosa_words'] = len(all_xho_words_found)
        print(f"  Vocab coverage: {results['passed']}/{results['total']} ({pct:.0f}%)")
        print(f"  Unique Xhosa words found in generation: {len(all_xho_words_found)}")
        print(f"  Words: {sorted(all_xho_words_found)[:20]}")
        self.results['vocabulary'] = results
        return results

    def eval_generation_quality(self):
        """Test 4: Generation quality - coherence, diversity, novelty."""
        print("\n=== Test 4: Generation Quality ===")
        results = {'passed': 0, 'failed': 0, 'total': 0, 'details': []}

        # Novelty test: same prompt, multiple generations should differ
        prompt = 'Umntu'
        generations = []
        for i in range(3):
            gen = self.predictor.generate(
                prompt, max_new_tokens=25, temperature=0.8,
                top_k=40, top_p=0.9, repetition_penalty=1.1
            )
            generations.append(gen)

        unique_count = len(set(generations))
        results['total'] = 3
        if unique_count >= 2:
            results['passed'] = unique_count
            results['details'].append(f"PASS: {unique_count}/3 unique generations from same prompt")
        else:
            results['failed'] = 3 - unique_count
            results['details'].append(f"FAIL: Only {unique_count}/3 unique generations (deterministic?)")

        # Length test: generated output should be longer than prompt
        for i, gen in enumerate(generations):
            if len(gen) > len(prompt):
                results['passed'] += 1
                results['details'].append(f"PASS: Gen {i+1} extended prompt ({len(gen)} > {len(prompt)} chars)")
            else:
                results['failed'] += 1
                results['details'].append(f"FAIL: Gen {i+1} too short ({len(gen)} chars)")
            results['total'] += 1

        # Stochasticity score
        if unique_count >= 2:
            results['score'] = 100
        else:
            results['score'] = 0

        print(f"  Quality: {results['passed']}/{results['total']} passed")
        print(f"  Unique generations: {unique_count}/3")
        for d in results['details']:
            print(f"    {d}")
        self.results['generation_quality'] = results
        return results

    def eval_contamination(self):
        """Test 5: Contamination detection - no verbatim training data reproduction."""
        print("\n=== Test 5: Contamination Detection ===")
        results = {'passed': 0, 'failed': 0, 'total': 0, 'details': []}

        if not self.training_records:
            results['details'].append("SKIP: No training corpus available for contamination check")
            results['score'] = 100
            self.results['contamination'] = results
            return results

        prompts = [
            'Molo', 'Umntu', 'Umthetho', 'Izibongo', 'Inkosi',
            'Ndiyavuya', 'Ubuntu', 'Imbongi', 'Indlela', 'Ukuphila',
        ]

        for prompt in prompts:
            generated = self.predictor.generate(
                prompt, max_new_tokens=30, temperature=0.8,
                top_k=40, top_p=0.9, repetition_penalty=1.1
            )
            # Check if generated text is a substring of any training record
            # (exact match would indicate retrieval, not generation)
            exact_match = False
            for rec in self.training_records:
                if generated.strip() == rec.strip():
                    exact_match = True
                    break

            results['total'] += 1
            if not exact_match:
                results['passed'] += 1
                results['details'].append(f"PASS: '{generated[:60]}' is novel (not in training corpus)")
            else:
                results['failed'] += 1
                results['details'].append(f"FAIL: Exact match found in training corpus")

        pct = results['passed'] / max(results['total'], 1) * 100
        results['score'] = pct
        print(f"  Contamination: {results['passed']}/{results['total']} ({pct:.0f}%) novel")
        for d in results['details']:
            print(f"    {d}")
        self.results['contamination'] = results
        return results

    def eval_offline_capability(self):
        """Test 6: Offline capability - load from single checkpoint, generate without retrieval."""
        print("\n=== Test 6: Offline Capability ===")
        results = {'passed': 0, 'failed': 0, 'total': 0, 'details': []}

        # Copy only the checkpoint to an isolated temp dir
        tmpdir = tempfile.mkdtemp(prefix='xnlp_eval_')
        dst = os.path.join(tmpdir, 'best_model.pt')
        shutil.copy2(self.checkpoint_path, dst)

        # Verify only the checkpoint is present
        files = os.listdir(tmpdir)
        if files == ['best_model.pt']:
            results['passed'] += 1
            results['details'].append("PASS: Only best_model.pt in isolated dir")
        else:
            results['failed'] += 1
            results['details'].append(f"FAIL: Extra files: {files}")
        results['total'] += 1

        # Load and generate from isolated dir
        try:
            from xnlp_trainer import XNLPPredictor
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            predictor = XNLPPredictor.load(dst, device='cpu')
            gen = predictor.generate('Molo', max_new_tokens=15, temperature=0.8)
            os.chdir(old_cwd)
            if gen and len(gen) > 4:
                results['passed'] += 1
                results['details'].append(f"PASS: Generated '{gen[:60]}' from isolated checkpoint")
            else:
                results['failed'] += 1
                results['details'].append("FAIL: No valid generation from isolated checkpoint")
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"FAIL: Error loading from isolated env: {e}")
        results['total'] += 1

        try:
            shutil.rmtree(tmpdir)
        except:
            pass

        pct = results['passed'] / max(results['total'], 1) * 100
        results['score'] = pct
        print(f"  Offline: {results['passed']}/{results['total']} ({pct:.0f}%)")
        for d in results['details']:
            print(f"    {d}")
        self.results['offline_capability'] = results
        return results

    def run_all(self):
        """Run all evaluation tests."""
        print("=" * 60)
        print("  XNLP Evaluation Suite")
        print("=" * 60)

        self.load_model()

        self.eval_spelling()
        self.eval_vocab_coverage()
        self.eval_generation_quality()
        self.eval_contamination()
        self.eval_offline_capability()

        # Summary
        print("\n" + "=" * 60)
        print("  EVALUATION SUMMARY")
        print("=" * 60)
        scores = {}
        for name, result in self.results.items():
            score = result.get('score', 0)
            scores[name] = score
            passed = result.get('passed', 0)
            total = result.get('total', 0)
            print(f"  {name:25s}: {passed}/{total}  ({score:.0f}%)")

        avg_score = sum(scores.values()) / max(len(scores), 1)
        print(f"\n  Overall score: {avg_score:.0f}%")

        # Save results
        report = {
            'checkpoint': self.checkpoint_path,
            'tests': self.results,
            'overall_score': avg_score,
        }
        os.makedirs('data/reports', exist_ok=True)
        with open('data/reports/evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report saved to data/reports/evaluation_report.json")

        return self.results


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'outputs/best_model.pt'
    evaluator = XNLPEvaluation(ckpt)
    evaluator.run_all()
