# XNLP isiXhosa Linguistic Foundation

This layer is separate from raw evidence, neural-model code, and model outputs. `generated/` is reproducibly created from the reconciled authoritative Mqhayi and Masikhanyise corpus:

```powershell
python xnlp_language/build_foundation.py
python xnlp_language/validation/validate_foundation.py
```

The generator preserves source spelling, provides record-level provenance, and does not infer grammar from orthography. It consequently reports the first release gate as incomplete until reviewed corpus annotation supplies evidence for morphology, agreement, verbs, syntax, and related domains.

`generated/annotation/linguistic_review_queue.jsonl` is the grammar-review interface. The dedicated `semantic_review_queue.jsonl` and `discourse_review_queue.jsonl` worklists cover each eligible source record while leaving all analyses empty. Their formats are defined by `schemas/semantic_frame_schema.json` and `schemas/discourse_annotation_schema.json`. The schemas support evidence-linked predicate/argument and discourse representations, machine suggestions explicitly separated from reviewer decisions, and unresolved ambiguity.

The bundle builder also materializes `generated/validation/linguistic_claims.jsonl`, `generated/annotation/lexical_analysis_queue.jsonl`, a work-isolated `generated/evaluation/gold_candidate_pool.jsonl`, split-integrity audits, and machine-/human-readable validation reports. The claim schema requires corpus record IDs; validation and establishment require explicit review history. Evidence state is independent of confidence. `ESTABLISHED` is never assigned by the generator. Lexical lemma, normalized form, part of speech, and meaning remain unassigned in the lexical queue until review.

Human claim decisions are applied with `xnlp_language.validation.linguistic_layer.apply_review_decision`: every decision records a reviewer and protocol, `ESTABLISHED` requires a prior `VALIDATED` decision plus a written strong-evidence basis, and cross-source support requires evidence across multiple works and source collections. Claims without resolved source records cannot enter the validated foundation.

The lexical queue supports bounded adjudication batches. Import adjudications with `python -m xnlp_language.annotation.adjudicate_lexical --limit 25 --adjudications decisions.jsonl`, where each JSONL row names a task ID, status, adjudicator, and note. Machine-only decisions are refused unless `--machine-assisted` is explicitly passed. To conservatively flag the current front batch for specialist review without assigning linguistic content, use `python -m xnlp_language.annotation.adjudicate_lexical --limit 25 --default-disputed --machine-assisted`; this records machine-only disputed tasks and writes an aggregate progress report.

The evaluation records are only an unannotated candidate pool, not gold labels. The work-isolated training report reserves the candidate's source work from that experiment's training partition; records have not passed human validation, and prior models trained with the historically overlapping split must not be evaluated against this work as though it were unseen.

Experimental model training is unfrozen. A run does not establish grammar or certify generated text as professional quality: the semantic and discourse queues are annotation work, not validated analyses, and professional-generation release remains gated on linguistic review and evaluation.

Build a single machine-readable index of all generated evidence, lexical and grammar artifacts, observation/review status, source provenance, model reports, checksums, and known split limitations:

```powershell
python -m xnlp_language.build_bundle
```

This writes `generated/foundation_bundle.json` and deterministically rebuilds the validation artifacts above. It embeds the compact structured grammar and observations while referencing larger corpus, lexicon, claim, and review files by path, record count, and SHA-256 rather than duplicating them. The bundle audits cited corpus record IDs and reports unresolved references, evidence states, pending annotation counts, and split leakage. It preserves existing confidence and review labels; it does not promote observed or supported entries to established grammar. Its purpose is to be a portable, auditable entry point to the complete foundation, not a claim that the language has been fully learned.

The evidence-bounded formal grammar audit can be regenerated separately after the language derivation and foundation bundle exist:

```powershell
python -m xnlp_language.derive_grammar
```

It writes `generated/grammar_derivation/grammar_specification.json` and linked disposition, surface-form scan, unresolved-question, coverage, independence, provenance, and human-report artifacts. It reuses `DERIVATION_EVIDENCE_SET_V1`, verifies its input hashes, and does not change corpus rows or source annotations. Legacy `SUPPORTED` labels remain visible as source metadata; they are not promoted to supported grammar rules. Surface-form scans are not grammatical counterexample judgments, and unresolved analyses remain subject to human adjudication. Rebuild the bundle after generation to catalog and hash these outputs.

## JSON/HTTP API

The deterministic analyzer can be exposed through a versioned JSON API. It uses only the Python standard library; clients in any language can call it over HTTP. It preserves original text, spelling, punctuation, whitespace, and character offsets. `UNKNOWN` means the available structured evidence did not support an analysis; it is not a spelling judgment.

Install the lightweight analyzer and start the local service:

```sh
python -m pip install .
xnlp-serve
```

The service listens on `127.0.0.1:8765` by default. Set `--host 0.0.0.0` only when you intentionally need access from other devices on a trusted network; this initial service does not provide authentication or TLS.

Check service status:

```sh
curl http://127.0.0.1:8765/health
```

Analyze text:

```sh
curl -X POST http://127.0.0.1:8765/v1/analyze \
	-H "Content-Type: application/json" \
	-d '{"text":"Molo, bantu bam!"}'
```

The response is UTF-8 JSON containing `api_version`, `language`, `processor`, and `result`. `result.text` is the exact submitted string; `result.tokens` contains word analyses and untouched separator spans with zero-based, end-exclusive character offsets.

Any HTTP-capable client can use the same contract. For example, JavaScript running in Node.js:

```js
const response = await fetch("http://127.0.0.1:8765/v1/analyze", {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({ text: "Molo, bantu bam!" }),
});
const result = await response.json();
console.log(result.result);
```

The HTTP boundary is a transport contract, not a claim that all client runtimes are natively supported. The analyzer is deterministic and evidence-bound; neural generation remains a separate, optional model layer. Install `.[model]` only when using PyTorch model components.
# Structured understanding and comprehension

The runtime exposes three evidence-bound operations in `xnlp_language`:

```python
from xnlp_language import comprehend, understand, validate_generation

structure = understand("Umntu uyafunda isiXhosa.")
answer = comprehend("UMqhayi ngumbongi.", "Ngubani uMqhayi?")
quality = validate_generation("Umntu", "Umntu ngumntu ngabantu.")
```

`understand` returns token analyses, clause frames, predicate/argument slots,
and explicit unknown boundaries. `comprehend` is extractive and cites the
supporting sentence; it returns `ABSTAIN` when the context does not support an
answer. `validate_generation` measures extension, analyzer coverage, and
repetition but intentionally does not claim native-speaker grammaticality.

The HTTP API provides `/v1/analyze`, `/v1/understand`, and
`/v1/comprehend`. Neural checkpoint loading and generation are implemented by
`xnlp_trainer.inference.XNLPPredictor`; the checkpoint must contain
`tokenizer_state` so inference is self-contained and offline.
