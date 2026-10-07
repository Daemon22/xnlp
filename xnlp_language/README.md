# XNLP isiXhosa Linguistic Foundation

This layer is separate from raw evidence, neural-model code, and model outputs. `generated/` is reproducibly created from the reconciled authoritative Mqhayi and Masikhanyise corpus:

```powershell
python xnlp_language/build_foundation.py
python xnlp_language/validation/validate_foundation.py
```

The generator preserves source spelling and record-level provenance. It does not infer grammar from orthography. It also materializes a partial semantic layer from glosses already present in the human-reviewed noun-class foundation.

Semantic outputs:

- `generated/semantics/lexical_senses.jsonl`: verified noun glosses linked to target-language source records containing the exact form.
- `generated/semantics/noun_class_profiles.json`: reviewed class-level semantic tendencies, explicitly marked as tendencies rather than per-word meaning rules.
- `generated/reports/semantic_coverage.json`: counts, source coverage, rejected examples, and gaps.
- `generated/reports/structural_coverage.json`: semantics is reported as partial when reviewed senses are available.

The semantic layer does not invent glosses from context. It does not yet encode verified synonymy, semantic roles, compositional sentence meaning, or discourse interpretation. The corpus-linked structural review queue remains available at `generated/annotation/linguistic_review_queue.jsonl`.


## Model-output quality gate

The optional PyTorch inference package calls a deterministic foundation gate
on every generated continuation by default. Build the language artifacts
before generating:

```sh
python xnlp_language/build_foundation.py
```

The gate blocks a continuation if it contains a word form absent from the
exact-form lexicon extracted from records classified `TARGET_LANGUAGE`, or
a letter outside the foundation's observed orthographic inventory. A blocked
candidate raises `GeneratedTextRejected` and exposes a structured
`validation_report`. A valid isiXhosa form not present in the corpus can be
rejected, so this gate favors evidence over coverage.

A pass means only that the continuation is lexically observed and uses
supported letters. The current foundation does not encode complete grammar,
semantics, or discourse rules; the gate cannot certify that a sentence is
grammatical or sensible. It does not rewrite or normalize generated text.


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
