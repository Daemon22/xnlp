# xNLP audit and current milestone

## Baseline

The original checkpoint is a 4,656,384-parameter tiny decoder-only model with a
2,048-token BPE vocabulary. It can load offline and sample continuations, but
its baseline task-prompt evaluation produced no valid structured output and no
target containment on the held-out sample.

## Implemented milestone

The structured curriculum contains 51,115 corpus-derived examples:

| Task | Examples |
|---|---:|
| STRUCTURE | 13,120 |
| GRAMMAR | 13,112 |
| QUESTION_ANSWER | 13,112 |
| CONTINUATION | 11,771 |

The updated checkpoint was trained in two bounded passes: 20 steps followed by
200 steps. The second pass reported mean loss 3.6425 and final loss 2.5426.

## Held-out comparison

The reproducible comparison is in
`artifacts/xnlp_structured/evaluation_baseline_vs_updated.json` and uses
`data/processed/v2/splits/test.txt`.

On the 20-example comparison sample:

- Baseline extension rate: 0% for all four tasks.
- Updated extension rate: 100% for all four tasks.
- Baseline valid structured JSON: 0%.
- Updated valid structured JSON: 0%.
- Baseline and updated target containment: 0%.

This is measurable progress in task-conditioned continuation, not proof of
language understanding. The model is still producing malformed JSON-like text
and weak answers.

## Reliable interfaces

The deterministic layer provides validated structure and grounded extraction
through `predictor.understand(text)` and `predictor.answer(context, question)`.
It abstains when evidence is insufficient. Neural output is not silently
treated as correct when it disagrees with the deterministic layer.

## Remaining gaps

The project still needs expert-reviewed morphological and syntax annotations,
unanswerable and contradiction QA sets, document/discourse examples, balanced
held-out source-family splits, and native-speaker evaluation of generated
isiXhosa. The current model is too small and too lightly trained to claim full
language competence.

## Context-length retry

The v4 retry compacted duplicated structure diagnostics and expanded the
RoPE context window to 1024 positions without changing learned weights.
`artifacts/xnlp_structured/model/structured_model_v4.report.json` records
500 steps, loss `0.8787` to `0.3824`, `39,346` retained examples, and
`11,769` dropped long targets. The non-structure tasks account for `37,995`
examples, so approximately `1,351` structure examples were retained. This
is an improvement in task coverage, but long-document structure analysis
still requires segmentation or a larger-context design.

## Next-milestone correction record (2026-10-19)

The original zero-valid-JSON result had two separate causes. The evaluator
used a fixed 24-token generation budget for every task, although structure
targets are substantially longer than that. The training dataset also
silently truncated overlong prompt-target sequences. The dataset now drops
such examples with an explicit count, defaults to the model's 512-position
context, and fails loudly if no trainable examples remain.

The evaluator now uses task-specific budgets and metrics. Containment is
retained only for extractive question answering; continuation is measured as
non-empty extension, while structure additionally requires canonical schema
validity. The bounded corrected comparison is recorded in
`artifacts/xnlp_structured/evaluation_baseline_vs_updated_v2.json` (one
held-out structure example because CPU generation is very slow): baseline
extension 0%, updated extension 100%, JSON validity 0% for both, and schema
validity 0% for both. This confirms the model has not yet learned reliable
structured generation.

The full retry is recorded in
`artifacts/xnlp_structured/evaluation_baseline_vs_updated_v2_full.json`.
It evaluates 20 held-out examples (5 per task) with task-specific budgets.
Baseline extension was 0% for every task; updated extension and non-empty
rate were 100% for every task. Updated valid JSON and canonical schema
validity were both 0% for every task, and extractive QA target containment
was 0%. The result is therefore improved task-conditioned continuation, not
language understanding or reliable answer generation.

The strongest retained candidate, v6, was re-evaluated in
`artifacts/xnlp_structured/evaluation_baseline_vs_v6.json` on the same
20-example held-out set. It produced non-empty extensions on 100% of all
tasks, but valid JSON, canonical schema validity, and extractive QA target
containment remained 0%. Additional loss reduction has not yet produced
reliable structured or answer generation.

The isolated four-example fixture is recorded in
`artifacts/xnlp_structured/fixture/overfit_model.report.json`: 20 steps,
initial loss 7.7145, final loss 5.1322, with two long targets deliberately
dropped rather than truncated. This demonstrates a functioning supervised
loss signal on the retained examples, but not linguistic competence.

Canonical analysis output is version 1 and is checked by
`xnlp_language/validation/analysis_schema.py`, with the corresponding JSON
Schema in `xnlp_language/schemas/analysis_schema.json`. Curriculum rows now
carry dataset version, source-linked provenance, and an explicit
`NOT_EXPERT_REVIEWED` annotation status. Grounded QA exposes exact evidence
spans, explicit-vs-unsupported claim type, and abstention.
