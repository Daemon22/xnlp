# XNLP isiXhosa Linguistic Foundation

This layer is separate from raw evidence, neural-model code, and model outputs. `generated/` is reproducibly created from the reconciled authoritative Mqhayi and Masikhanyise corpus:

```powershell
python xnlp_language/build_foundation.py
python xnlp_language/validation/validate_foundation.py
```

The generator preserves source spelling, provides record-level provenance, and does not infer grammar from orthography. It consequently reports the first release gate as incomplete until reviewed corpus annotation supplies evidence for morphology, agreement, verbs, syntax, and related domains.

`generated/annotation/linguistic_review_queue.jsonl` is the next analysis interface. Its tasks are deliberately unanalysed and corpus-linked: a reviewer must supply evidence before a grammatical assertion can be promoted into the foundation.
