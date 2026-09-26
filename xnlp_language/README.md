# XNLP isiXhosa Linguistic Foundation

This layer is separate from raw evidence, neural-model code, and model outputs. `generated/` is reproducibly created from the reconciled authoritative Mqhayi and Masikhanyise corpus:

```powershell
python xnlp_language/build_foundation.py
python xnlp_language/validation/validate_foundation.py
```

The generator preserves source spelling, provides record-level provenance, and does not infer grammar from orthography. It consequently reports the first release gate as incomplete until reviewed corpus annotation supplies evidence for morphology, agreement, verbs, syntax, and related domains.

`generated/annotation/linguistic_review_queue.jsonl` is the next analysis interface. Its tasks are deliberately unanalysed and corpus-linked: a reviewer must supply evidence before a grammatical assertion can be promoted into the foundation.

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
