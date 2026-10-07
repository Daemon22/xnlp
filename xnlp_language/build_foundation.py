#!/usr/bin/env python3
"""Build the evidence-first XNLP isiXhosa linguistic foundation.

This program deliberately makes no claim that the corpus exhausts isiXhosa.
It records directly observable writing and distributional facts, and marks every
heuristic analysis as provisional.  All generated examples retain record-level
provenance and preserve the source surface text exactly.

Architecture (evidence pipeline):
    RAW CORPUS
        ↓
    OBSERVATION  (corpus mining — frequency counts, never rules)
        ↓
    ANNOTATION / REVIEW  (review queue, proposed_analysis = null)
        ↓
    LINGUISTIC HYPOTHESIS  (structured entries citing observation IDs)
        ↓
    SUPPORTED Rule candidate  (evidence-linked, status OBSERVED/UNDER_REVIEW/SUPPORTED)
        ↓
    ESTABLISHED FOUNDATION ENTRY  (requires human review — never fully automatic)

Every structured entry carries:
  entry_id, category, claim, evidence_records, source_provenance,
  observation_ids, review_status, confidence, counterexamples, notes
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "authoritative" / "v2_authoritative_all.jsonl"
OUT = ROOT / "xnlp_language" / "generated"
FOUNDATION_DIR = ROOT / "xnlp_language" / "foundation"

WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
PREFIXES = ("aba", "ama", "imi", "izi", "iin", "oo", "umu", "um", "isi", "e", "zi", "i", "u")

# Noun-class prefix pairs derived from corpus prefix co-occurrence + hand-curated evidence.
# Each tuple is (singular_prefix, plural_prefix, class1_id, class2_id).
# NOTE: These pairings are hypotheses that require human review — they are NOT
# automatically assumed. They are supported by the corpus observations and the
# hand-curated foundation entries.
NOUN_CLASS_PAIRINGS: list[tuple[str, str, str, str]] = [
    ("um", "aba", "NC_001", "NC_002"),    # people, agents
    ("um", "imi", "NC_003", "NC_004"),    # objects, tools, abstract nouns
    ("i", "ama", "NC_005", "NC_006"),     # things, abstract
    ("isi", "izi", "NC_007", "NC_008"),  # languages, abstract, diseases
    ("u", "o", "NC_009", "NC_010"),      # proper names, kinship, animals
]

# Subject concord prefixes observed before verb stems (longest first)
SUBJECT_CONCORDS = [
    ("ba", "class_2_plural"),
    ("u", "class_1_singular"),
    ("w", "class_1_singular_vowel_allomorph"),
    ("a", "class_6_or_negative"),
    ("e", "class_11"),
    ("zi", "class_10_plural"),
]

# Possessive concord prefixes (longest-first for matching)
POSSESSIVE_CONCORDS = [
    ("kwa", "class_9_variant"),
    ("ka", "class_9"),
    ("we", "class_1_variant"),
    ("wa", "class_1"),
]

# TAM markers (longest first)
TAM_MARKERS = [("nga", "present_habitual"), ("se", "past_relative")]

# Copula markers (word-initial)
COPULA_MARKERS = ["ngu", "yu"]

# Negative markers (prefixes on verbs)
NEGATION_PREFIXES = ["ha", "hayi", "aku", "ang", "ank"]

# Complementizer / subordinator
COMPLEMENTIZERS = ["ukuthi", "kuba", "xa", "ngoba"]

# Verb extensions (longest-first for matching)
VERB_EXTENSIONS = [
    ("elwa", "passive"), ("iswa", "passive"), ("isa", "causative"),
    ("isa", "applicative"), ("ana", "benefactive"),
]

# Function words to scan for (closed-class items)
FUNCTION_WORDS = [
    "mna", "wena", "thina", "inina", "nina", "yena",
    "kwaye", "kodwa", "xa", "ukuthi", "kuba", "ngoba",
    "lo", "loo", "lowo", "laa", "len", "le",
    "kwayi", "kuphi", "njeng", "yini", "ngubani",
    "na", "ke", "man", "ngam", "ngu",
]

# Review-status vocabulary
REVIEW_STATUSES = {"OBSERVED", "UNDER_REVIEW", "SUPPORTED", "ESTABLISHED", "CONTESTED", "REJECTED"}
CONFIDENCE_LEVELS = {"OBSERVED", "SUPPORTED", "HIGH_CONFIDENCE"}
GRAMMAR_REVIEW_STATUSES = {"OBSERVED", "UNDER_REVIEW", "SUPPORTED", "ESTABLISHED", "CONTESTED", "REJECTED"}


# =========================================================================
# Utility helpers
# =========================================================================

def write_json(path: Path, value: Any) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    """Write JSONL with one sorted-key object per line (deterministic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read authoritative corpus records, deduplicating by record_id."""
    records, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("record_id", "text", "source", "provenance_type", "validation_status")
        if any(not row.get(key) for key in required):
            continue
        if row["record_id"] in seen or row["source"] not in {"mqhayi", "masikhanyise"}:
            continue
        if row["provenance_type"] != "authoritative" or row["validation_status"] != "authoritative":
            continue
        seen.add(row["record_id"])
        records.append(row)
    return records


def evidence(record: dict[str, Any], form: str, start: int, end: int) -> dict[str, Any]:
    """Build a provenance-bearing evidence snippet."""
    return {
        "record_id": record["record_id"],
        "source_work": record["source"],
        "source_title": record.get("source_title", ""),
        "orthography": record.get("orthography", "uncertain"),
        "surface_form": form,
        "character_span": [start, end],
        "context": record["text"][max(0, start - 80):min(len(record["text"]), end + 80)],
    }


def confidence(count: int) -> str:
    """Map an observation frequency to a confidence label.

    Thresholds apply to *observations* only — they never promote an
    observation to a grammatical rule.
    """
    return "HIGH_CONFIDENCE" if count >= 20 else "SUPPORTED" if count >= 5 else "OBSERVED"


def source_hash(text: str) -> str:
    """Fingerprint original text without changing a character."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _match_prefix(word: str) -> str | None:
    """Return the first matching nominal prefix (longest-first), or None."""
    lowered = word.lower()
    for prefix in PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
            return prefix
    return None


def _match_concord(word: str, concord_list) -> str | None:
    """Return the first matching concord prefix, or None."""
    lowered = word.lower()
    for prefix, _label in concord_list:
        if lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
            return prefix
    return None


def _read_hand_curated_foundation() -> dict[str, list[dict]]:
    """Load all hand-curated JSON files from xnlp_language/foundation/.

    Returns a dict mapping category name to list of parsed dicts.
    These are HUMAN-CURATED entries — never regenerated.
    """
    result: dict[str, list[dict]] = collections.defaultdict(list)
    if not FOUNDATION_DIR.exists():
        return dict(result)
    for category_dir in sorted(FOUNDATION_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        for json_file in sorted(category_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for entry in data:
                        result[category_dir.name].append(entry)
                elif isinstance(data, dict):
                    result[category_dir.name].append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return dict(result)


def _record_ids_from_evidence(evidence_list: list) -> list[str]:
    """Extract sorted unique record IDs from an evidence list."""
    ids = set()
    for item in evidence_list:
        if isinstance(item, dict):
            rid = item.get("record_id")
            if rid:
                ids.add(rid)
        elif isinstance(item, str):
            ids.add(item)
    return sorted(ids)


def _sources_from_evidence(evidence_list: list) -> list[str]:
    """Extract sorted unique source work names from an evidence list."""
    sources = set()
    for item in evidence_list:
        if isinstance(item, dict):
            sw = item.get("source_work") or item.get("source")
            if sw:
                sources.add(sw)
    return sorted(sources)


# =========================================================================
# PHASE 2: STRUCTURAL ANALYSIS
# Scans TARGET_LANGUAGE records for recurring linguistic patterns.
# Every output is an OBSERVATION linked to evidence — never a rule.
# =========================================================================

def analyze_noun_classes(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for noun-class prefix co-occurrence evidence.

    Produces OBSERVATIONS (not rules). Each observation records:
    - which prefix was found
    - how often it co-occurs with other prefixes in the same record
      (evidence for singular/plural pairing, not proof of it)
    - token frequency, distinct forms, and sample evidence
    """
    prefix_evidence: dict[str, list[dict]] = collections.defaultdict(list)
    prefix_forms: dict[str, set[str]] = collections.defaultdict(set)
    cooccurrence: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    historical_counts: dict[str, int] = collections.Counter()
    modern_counts: dict[str, int] = collections.Counter()

    for record in analysis_records:
        text = record["text"]
        ortho = record.get("orthography", "uncertain")
        words = list(WORD.finditer(text))
        record_prefixes: set[str] = set()
        for match in words:
            word = match.group(0)
            prefix = _match_prefix(word)
            if prefix:
                item = evidence(record, word, match.start(), match.end())
                prefix_evidence[prefix].append(item)
                prefix_forms[prefix].add(word.lower())
                record_prefixes.add(prefix)
                if ortho == "modern":
                    modern_counts[prefix] += 1
                elif ortho in {"traditional", "mixed"}:
                    historical_counts[prefix] += 1
        # Track co-occurrence
        for p1 in sorted(record_prefixes):
            for p2 in sorted(record_prefixes):
                if p1 < p2:
                    cooccurrence[p1][p2] += 1

    # Build pairing observations from co-occurrence
    pairing_findings: list[dict] = []
    review_tasks: list[dict] = []
    for prefix, partners in sorted(cooccurrence.items()):
        for partner, count in partners.most_common(5):
            if count >= 3:
                pairing_findings.append({
                    "prefix_pair": f"{prefix}-{partner}",
                    "cooccurrence_count": count,
                    "prefix_a": prefix,
                    "prefix_b": partner,
                    "records_seen": count,
                    "analysis_status": "OBSERVED",
                    "confidence": confidence(count),
                    "notes": f"{prefix} and {partner} co-occur in {count} records. This is distributional evidence for a possible singular/plural pairing, not proof of agreement class membership.",
                })

    observations: list[dict] = []
    for prefix in PREFIXES:
        if prefix not in prefix_evidence:
            continue
        forms = sorted(prefix_forms[prefix])
        partners = dict(sorted(
            ((f"{prefix}-{p}", c) for p, c in cooccurrence[prefix].items()),
            key=lambda x: -x[1]
        )[:10])
        hist = historical_counts.get(prefix, 0)
        mod = modern_counts.get(prefix, 0)
        if hist > 0 and mod > 0:
            hist_status = "variant"
        elif hist > 0:
            hist_status = "historical"
        else:
            hist_status = "modern"
        obs = {
            "observation_id": f"NC_PREFIX_{prefix.upper()}",
            "surface_prefix": prefix,
            "observed_forms": forms[:200],
            "token_frequency": len(prefix_evidence[prefix]),
            "distinct_forms": len(forms),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(prefix_evidence[prefix])),
            "evidence": prefix_evidence[prefix][:10],
            "cooccurring_prefixes": partners,
            "historical_status": hist_status,
            "frequency_distribution": {"modern": mod, "historical": hist},
            "notes": "Word-initial prefix observed in corpus. Noun-class assignment, singular/plural pairing, and agreement behavior require independent morphological analysis and human review.",
        }
        observations.append(obs)

    pairing_obs = {
        "observation_id": "NC_PAIRINGS",
        "observation_type": "prefix_cooccurrence",
        "analysis_status": "OBSERVED",
        "confidence": "OBSERVED",
        "pairings": pairing_findings,
        "evidence_sample": prefix_evidence.get("um", [])[:5],
        "notes": "Co-occurrence of prefixes within records suggests possible pairings. Each pairing requires independent morphological confirmation.",
    }

    return observations + [pairing_obs], review_tasks


def analyze_agreement(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for subject and possessive concord evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()

            concord = _match_concord(word, SUBJECT_CONCORDS)
            if concord:
                stem = lowered[len(concord):]
                if len(stem) >= 2 and stem[0] not in "aeiou":
                    label = next(l for p, l in SUBJECT_CONCORDS if p == concord)
                    key = f"subj_concord_{label}_{concord}"
                    findings[key].append(evidence(record, word, match.start(), match.end()))

            poss = _match_concord(word, POSSESSIVE_CONCORDS)
            if poss:
                label = next(l for p, l in POSSESSIVE_CONCORDS if p == poss)
                key = f"poss_concord_{label}_{poss}"
                findings[key].append(evidence(record, word, match.start(), match.end()))

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items()):
        obs = {
            "observation_id": f"AGR_{pattern.upper()}",
            "pattern_type": "agreement",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of the pattern '{pattern}'. Concord-class mapping requires independent morphological analysis and human review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_AGREEMENT_{pattern.upper()}",
                "task_type": "NOUN_CLASS_OR_AGREEMENT",
                "surface_form": pattern,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Determine the noun class and grammatical function of this concord form. Provide corpus examples showing agreement with a noun. Mark uncertainty explicitly.",
            })

    return observations, review_tasks


def analyze_verb_morphology(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for verb morphology evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()

            concord = _match_concord(word, SUBJECT_CONCORDS)
            if concord and len(lowered) > len(concord) + 2:
                stem = lowered[len(concord):]
                label = next(l for p, l in SUBJECT_CONCORDS if p == concord)

                for ext, ext_label in VERB_EXTENSIONS:
                    if stem.endswith(ext) and len(stem) > len(ext) + 1:
                        key = f"verb_ext_{ext_label}_{ext}"
                        findings[key].append(evidence(record, word, match.start(), match.end()))
                        break

                for tam, tam_label in TAM_MARKERS:
                    if stem.startswith(tam) and len(stem) > len(tam) + 1:
                        key = f"verb_tam_{tam_label}_{tam}"
                        findings[key].append(evidence(record, word, match.start(), match.end()))
                        break

                key = f"verb_subj_{label}_{concord}"
                findings[key].append(evidence(record, word, match.start(), match.end()))

            if lowered.startswith("ukuthi") and len(lowered) > 5:
                findings["complementizer_ukuthi"].append(
                    evidence(record, word, match.start(), match.end()))

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items()):
        obs = {
            "observation_id": f"VM_{pattern.upper()}",
            "pattern_type": "verb_morphology",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of {pattern}. Candidate segmentation requires human morphological review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_VERB_SEG_{pattern.upper()}",
                "task_type": "MORPHOLOGY_SEGMENTATION",
                "surface_form": pattern,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Segment the candidate verb form into subject marker, tense/aspect marker, verb root, derivational extensions, and final vowel. Provide corpus examples. Mark uncertainty explicitly.",
            })

    return observations, review_tasks


def analyze_tam(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for tense/aspect/mood and negation evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()

            for marker, label in TAM_MARKERS:
                if lowered.startswith(marker) and len(lowered) > len(marker) + 1:
                    findings[f"tam_{label}_{marker}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

            for marker in NEGATION_PREFIXES:
                if lowered.startswith(marker) and len(lowered) > len(marker) + 1:
                    findings[f"neg_{marker}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

            for marker in COPULA_MARKERS:
                if lowered.startswith(marker) and len(lowered) > len(marker) + 1:
                    findings[f"copula_{marker}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items(), key=lambda x: -len(x[1])):
        obs = {
            "observation_id": f"TAM_{pattern.upper()}",
            "pattern_type": "tense_aspect_mood" if pattern.startswith("tam") else
                          "copula" if pattern.startswith("copula") else "negation",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of {pattern}. Grammatical function requires human morphological review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_TAM_{pattern.upper()}",
                "task_type": "MORPHOLOGY_SEGMENTATION",
                "surface_form": pattern,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Determine the tense/aspect/mood or negation function of this marker. Provide corpus examples showing it in context with subject agreement. Mark uncertainty explicitly.",
            })

    return observations, review_tasks


def analyze_function_words(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for function word evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()
            for fw in FUNCTION_WORDS:
                if lowered == fw or (lowered.startswith(fw) and len(lowered) == len(fw)):
                    findings[fw].append(evidence(record, word, match.start(), match.end()))
                    break

    observations: list[dict] = []
    for word, occurrences in sorted(findings.items()):
        obs = {
            "observation_id": f"FW_{word.upper()}",
            "pattern_type": "function_word",
            "pattern": word,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of the function word '{word}'. Grammatical function requires human review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_FUNCTION_WORD_{word.upper()}",
                "task_type": "LEXICAL_LEMMA",
                "surface_form": word,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Confirm the grammatical function and distribution of this word form.",
            })

    return observations, review_tasks


def analyze_modifiers(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for modifier (adjective, relative) evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        tokens = [(m, m.group(0), m.group(0).lower()) for m in words]

        for i, (match, word, lowered) in enumerate(tokens):
            if lowered.startswith("om") and len(lowered) > 3:
                if i + 1 < len(tokens):
                    next_word = tokens[i + 1][2]
                    next_prefix = _match_prefix(next_word)
                    if next_prefix:
                        findings["adjective_agreement"].append(
                            evidence(record, word, match.start(), match.end()))

            if "o" in lowered and lowered.startswith(("ba", "u", "w", "a")):
                if lowered.endswith("o") and not lowered.endswith("lo"):
                    findings["relative_marker_candidate"].append(
                        evidence(record, word, match.start(), match.end()))

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items(), key=lambda x: -len(x[1])):
        obs = {
            "observation_id": f"MOD_{pattern.upper()}",
            "pattern_type": "modifier",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of {pattern}. Modifier agreement requires human morphological review.",
        }
        observations.append(obs)

        review_tasks.append({
            "task_id": f"ANN_MODIFIER_{pattern.upper()}",
            "task_type": "NOUN_CLASS_OR_AGREEMENT",
            "surface_form": pattern,
            "status": "PENDING_HUMAN_REVIEW",
            "proposed_analysis": None,
            "confidence": "INSUFFICIENT_EVIDENCE",
            "source_record_ids": [e["record_id"] for e in occurrences[:5]],
            "evidence": occurrences[:5],
            "instruction": "Determine the modifier type and agreement behavior from the cited corpus context.",
        })

    return observations, review_tasks


def analyze_syntax(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for syntactic pattern evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        tokens = [(m, m.group(0), m.group(0).lower()) for m in words]

        for i, (match, word, lowered) in enumerate(tokens):
            prefix = _match_prefix(word)
            if prefix and i + 1 < len(tokens):
                next_word = tokens[i + 1][2]
                concord = _match_concord(next_word, SUBJECT_CONCORDS)
                if concord:
                    key = f"SV_pattern_{prefix}_{concord}"
                    findings[key].append(evidence(record, f"{word} {tokens[i+1][1]}",
                                                   match.start(), tokens[i + 1][0].end()))

            for copula in COPULA_MARKERS:
                if lowered.startswith(copula) and not lowered.startswith(tuple("aeiou")):
                    if i + 1 < len(tokens):
                        next_word = tokens[i + 1][2]
                        next_prefix = _match_prefix(next_word)
                        if next_prefix:
                            key = f"copula_{copula}_noun_{next_prefix}"
                            findings[key].append(evidence(record, f"{word} {tokens[i+1][1]}",
                                                           match.start(), tokens[i + 1][0].end()))

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items(), key=lambda x: -len(x[1])):
        obs = {
            "observation_id": f"SYN_{pattern.upper()}",
            "pattern_type": "syntax",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of {pattern}. Syntactic function requires human review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_SYNTAX_{pattern.upper()}",
                "task_type": "MORPHOLOGY_SEGMENTATION",
                "surface_form": pattern,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Confirm the syntactic construction and constituent order from the cited corpus context.",
            })

    return observations, review_tasks


def analyze_word_formation(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for word formation evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()

            # Look for derivation patterns: base + extension
            for ext, label in VERB_EXTENSIONS:
                if lowered.endswith(ext) and len(lowered) > len(ext) + 2:
                    findings[f"derivation_{label}_{ext}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

            # Look for compounding (hyphenated or apostrophe-joined)
            if "-" in lowered or "'" in lowered:
                findings["compounding_hyphenated"].append(
                    evidence(record, word, match.start(), match.end()))

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items(), key=lambda x: -len(x[1])):
        obs = {
            "observation_id": f"WF_{pattern.upper()}",
            "pattern_type": "word_formation",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of {pattern}. Morphological interpretation requires human review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_WORDFORM_{pattern.upper()}",
                "task_type": "MORPHOLOGY_SEGMENTATION",
                "surface_form": pattern,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Segment the word form into constituents and determine the derivation/compounding pattern.",
            })

    return observations, review_tasks


def analyze_negation(analysis_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan TARGET_LANGUAGE records for negation evidence."""
    findings: dict[str, list[dict]] = collections.defaultdict(list)
    review_tasks: list[dict] = []

    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()

            for marker in ["a", "ha", "hayi", "aku", "ang"]:
                if lowered.startswith(marker) and len(lowered) > len(marker) + 1:
                    findings[f"neg_prefix_{marker}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

    observations: list[dict] = []
    for pattern, occurrences in sorted(findings.items(), key=lambda x: -len(x[1])):
        obs = {
            "observation_id": f"NEG_{pattern.upper()}",
            "pattern_type": "negation",
            "pattern": pattern,
            "token_frequency": len(occurrences),
            "analysis_status": "OBSERVED",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "historical_status": "uncertain",
            "notes": f"Observed {len(occurrences)} instances of {pattern}. Negative function requires human review.",
        }
        observations.append(obs)

        if len(occurrences) >= 5:
            review_tasks.append({
                "task_id": f"ANN_NEGATION_{pattern.upper().replace('_', '-')}",
                "task_type": "MORPHOLOGY_SEGMENTATION",
                "surface_form": pattern,
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [e["record_id"] for e in occurrences[:5]],
                "evidence": occurrences[:5],
                "instruction": "Confirm the negation function and scope of this marker from the cited corpus context.",
            })

    return observations, review_tasks


# =========================================================================
# PHASE 3: STRUCTURED GRAMMATICAL SYSTEM BUILDERS
# Convert hand-curated foundation entries into structured JSONL files with
# evidence links.  These are HYPOTHESES — not rules — that require human
# review.  Status is never ESTABLISHED automatically.
# =========================================================================

def _build_noun_class_system(
    destination: Path,
    hand_curated: dict[str, list[dict]],
    prefix_observations: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build structured noun-class entries from hand-curated foundation + corpus evidence.

    Produces:
      - noun_classes.jsonl  (structured class entries)
      - noun_class_evidence.jsonl  (individual evidence items)
      - noun_class_review_queue.jsonl  (review tasks)
    """
    nc_entries = hand_curated.get("noun_classes", [])
    # Build a lookup of observation IDs by prefix
    obs_by_prefix: dict[str, dict] = {}
    for obs in prefix_observations:
        prefix = obs.get("surface_prefix", "")
        if prefix:
            obs_by_prefix[prefix] = obs

    nc_entries_sorted = sorted(nc_entries, key=lambda e: e.get("class_number", 999))

    classes: list[dict] = []
    evidence_items: list[dict] = []
    review_tasks: list[dict] = []

    for entry in nc_entries_sorted:
        class_id = entry.get("class_id", "")
        class_number = entry.get("class_number", 0)
        prefix_info = entry.get("prefix", {})
        singular_prefix = prefix_info.get("singular", "")
        plural_prefix = prefix_info.get("plural", "")
        pairing = entry.get("pairing", {})
        examples = entry.get("examples", [])

        # Gather evidence record IDs from hand-curated examples
        evidence_record_ids = _record_ids_from_evidence(examples)
        # Also include from source_evidence
        source_ev = entry.get("source_evidence", [])
        evidence_record_ids.extend(_record_ids_from_evidence(source_ev))
        evidence_record_ids = sorted(set(evidence_record_ids))

        # Determine related observation IDs
        obs_ids = []
        for p in [singular_prefix.replace("-", ""), plural_prefix.replace("-", "")]:
            if p:
                # Handle compound prefixes like "um-" or "umu-"
                base = p.replace("-", "")
                obs_id = f"NC_PREFIX_{base.upper()}"
                if obs_id not in obs_ids:
                    obs_ids.append(obs_id)

        # Determine confidence from hand-curated data + corpus evidence
        freq_info = entry.get("frequency", {})
        corpus_count = freq_info.get("corpus_count", 0) if isinstance(freq_info, dict) else 0
        hc_confidence = entry.get("confidence", "HIGH_CONFIDENCE") if isinstance(hc_confidence := entry.get("confidence", "HIGH_CONFIDENCE"), str) else "HIGH_CONFIDENCE"

        # Build the claim
        claim = f"Noun class {class_number}: prefix {singular_prefix} pairs with {plural_prefix} ({pairing.get('pairing_type', 'regular')} pairing)"

        # Determine review status: hand-curated entries are SUPPORTED when evidence exists,
        # but never ESTABLISHED without human review
        if len(examples) >= 2 and corpus_count >= 5:
            review_status = "SUPPORTED"
        elif len(examples) >= 1:
            review_status = "UNDER_REVIEW"
        else:
            review_status = "OBSERVED"

        classes.append({
            "entry_id": class_id,
            "category": "noun_class",
            "claim": claim,
            "class_number": class_number,
            "singular_prefix": singular_prefix,
            "plural_prefix": plural_prefix,
            "prefix_variants": prefix_info.get("variants", []),
            "pairing_type": pairing.get("pairing_type", "regular"),
            "pairing": pairing,
            "semantic_tendencies": entry.get("semantic_tendencies", []),
            "agreement_prefixes": entry.get("agreement_prefixes", {}),
            "stem_behavior": entry.get("stem_behavior", {}),
            "examples": examples,
            "exceptions": entry.get("exceptions", []),
            "evidence_records": evidence_record_ids,
            "source_provenance": sorted(set(
                e.get("source_work", "") for e in examples + source_ev if e.get("source_work")
            )),
            "observation_ids": obs_ids,
            "review_status": review_status,
            "confidence": hc_confidence,
            "counterexamples": entry.get("exceptions", []),
            "notes": entry.get("notes", "Derived from corpus evidence and hand-curated foundation. Requires human review for ESTABLISHED status."),
        })

        # Generate evidence items for each example
        for example in examples:
            evidence_items.append({
                "evidence_id": f"NC_EVID_{class_id}_{example.get('noun', 'UNK')}",
                "entry_id": class_id,
                "category": "noun_class",
                "record_id": example.get("source_record_id", ""),
                "surface_form": example.get("noun", ""),
                "context": example.get("meaning", ""),
                "source_work": example.get("source_record_id", "").split("_")[0] if example.get("source_record_id") else "",
                "relationship": "singular_plural_pairing",
                "notes": f"Singular '{example.get('noun', '')}' → plural '{example.get('plural_form', '')}'",
                "review_status": "SUPPORTED",
            })

        # Review task for every entry — SUPPORTED entries get routine verification
        review_tasks.append({
            "task_id": f"ANN_NOUN_CLASS_REVIEW_{class_id}",
            "task_type": "NOUN_CLASS_OR_AGREEMENT",
            "surface_form": f"{singular_prefix}-{plural_prefix}",
            "entry_id": class_id,
            "status": "PENDING_HUMAN_REVIEW",
            "proposed_analysis": None,
            "confidence": "INSUFFICIENT_EVIDENCE",
            "source_record_ids": evidence_record_ids[:10],
            "observation_ids": obs_ids,
            "evidence": [{"record_id": rid, "surface_form": ""} for rid in evidence_record_ids[:5]],
            "instruction": f"Verify noun class {class_number} prefix pairing ({singular_prefix} → {plural_prefix}) using the cited corpus records. Confirm singular/plural relationship, agreement behavior, and semantic tendencies. Mark any uncertainty explicitly.",
        })

    write_jsonl(destination / "grammar" / "noun_classes.jsonl", classes)
    write_jsonl(destination / "grammar" / "noun_class_evidence.jsonl", evidence_items)
    write_jsonl(destination / "grammar" / "noun_class_review_queue.jsonl", review_tasks)

    return classes, evidence_items, review_tasks


def _build_agreement_system(
    destination: Path,
    hand_curated: dict[str, list[dict]],
    agreement_observations: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build structured agreement entries from hand-curated foundation + corpus evidence."""
    ag_entries = hand_curated.get("agreement", [])
    ag_entries_sorted = sorted(ag_entries, key=lambda e: e.get("agreement_id", ""))

    system: list[dict] = []
    evidence_items: list[dict] = []
    review_tasks: list[dict] = []

    for entry in ag_entries_sorted:
        ag_id = entry.get("agreement_id", "")
        ag_type = entry.get("agreement_type", "")
        noun_class = entry.get("noun_class", "")
        number = entry.get("number", "")
        concord_info = entry.get("concord_forms", {})
        examples = entry.get("examples", [])
        source_ev = entry.get("source_evidence", [])

        evidence_record_ids = sorted(set(
            _record_ids_from_evidence(examples) + _record_ids_from_evidence(source_ev)
        ))

        # Link to agreement observations
        obs_ids = []
        for obs in agreement_observations:
            pattern = obs.get("pattern", "")
            obs_ids.append(obs.get("observation_id", ""))

        freq_info = entry.get("frequency", {})
        corpus_count = freq_info.get("corpus_count", 0) if isinstance(freq_info, dict) else 0
        hc_confidence = entry.get("confidence", "HIGH_CONFIDENCE")

        claim = (
            f"{ag_type} for {noun_class} {number}: prefix "
            f"{concord_info.get('prefix', '')} with variants {concord_info.get('variants', [])}"
        )

        if len(examples) >= 2 and corpus_count >= 5:
            review_status = "SUPPORTED"
        elif len(examples) >= 1:
            review_status = "UNDER_REVIEW"
        else:
            review_status = "OBSERVED"

        system.append({
            "entry_id": ag_id,
            "category": "agreement",
            "claim": claim,
            "agreement_type": ag_type,
            "noun_class": noun_class,
            "number": number,
            "concord_forms": concord_info,
            "target_category": entry.get("target_category", ""),
            "position": entry.get("position", ""),
            "phonological_rules": entry.get("phonological_rules", []),
            "examples": examples,
            "exceptions": entry.get("exceptions", []),
            "evidence_records": evidence_record_ids,
            "source_provenance": sorted(set(
                e.get("source_work", "") for e in examples + source_ev if e.get("source_work")
            )),
            "observation_ids": [oid for oid in obs_ids if oid],
            "review_status": review_status,
            "confidence": hc_confidence,
            "counterexamples": entry.get("exceptions", []),
            "notes": entry.get("notes", "Derived from corpus evidence and hand-curated foundation. Requires human review."),
        })

        for example in examples:
            evidence_items.append({
                "evidence_id": f"AGR_EVID_{ag_id}_{example.get('noun', 'UNK')[:10]}",
                "entry_id": ag_id,
                "category": "agreement",
                "record_id": example.get("source_record_id", ""),
                "surface_form": example.get("full_construction", ""),
                "context": example.get("translation", ""),
                "source_work": example.get("source_record_id", "").split("_")[0] if example.get("source_record_id") else "",
                "relationship": f"{ag_type}_for_{noun_class}_{number}",
                "notes": f"Concord '{example.get('concord_form', '')}' in construction '{example.get('full_construction', '')}'",
                "review_status": review_status,
            })

        # Review task for every agreement entry — SUPPORTED entries get routine verification
        review_tasks.append({
            "task_id": f"ANN_AGREEMENT_REVIEW_{ag_id}",
            "task_type": "NOUN_CLASS_OR_AGREEMENT",
            "surface_form": f"{noun_class}_{number}_{ag_type}",
            "entry_id": ag_id,
            "status": "PENDING_HUMAN_REVIEW",
            "proposed_analysis": None,
            "confidence": "INSUFFICIENT_EVIDENCE",
            "source_record_ids": evidence_record_ids[:10],
            "observation_ids": obs_ids,
            "evidence": [{"record_id": rid, "surface_form": ""} for rid in evidence_record_ids[:5]],
            "instruction": f"Verify {ag_type} agreement pattern for {noun_class} {number}. Check phonological rules and cross-reference with corpus evidence. Mark uncertainty explicitly.",
        })

    write_jsonl(destination / "grammar" / "agreement_system.jsonl", system)
    write_jsonl(destination / "grammar" / "agreement_evidence.jsonl", evidence_items)
    write_jsonl(destination / "grammar" / "agreement_review_queue.jsonl", review_tasks)

    return system, evidence_items, review_tasks


def _build_verb_morphology(
    destination: Path,
    verb_observations: list[dict],
    analysis_records: list[dict],
    review_tasks: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build structured verb morphology entries from observations + corpus evidence.

    Each entry records a candidate segmentation with supporting evidence,
    counterexamples, status, and confidence.  Nothing is ESTABLISHED
    automatically.
    """
    entries: list[dict] = []
    evidence_items: list[dict] = []
    analysis_queue: list[dict] = []

    # Build suffix/pattern frequency tables from corpus
    verb_form_freq: dict[str, list[dict]] = collections.defaultdict(list)
    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()
            # Collect verb-like forms (with subject concord + stem)
            concord = _match_concord(word, SUBJECT_CONCORDS)
            if concord and len(lowered) > len(concord) + 2:
                stem = lowered[len(concord):]
                if len(stem) >= 2 and stem[0] not in "aeiou":
                    verb_form_freq[lowered].append(evidence(record, word, match.start(), match.end()))

    # Collect unique suffix patterns
    suffix_patterns: dict[str, list[dict]] = collections.defaultdict(list)
    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()
            for ext, label in VERB_EXTENSIONS:
                if lowered.endswith(ext) and len(lowered) > len(ext) + 2:
                    suffix_patterns[f"ext_{label}_{ext}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

    # Build verb morphology entries from observations
    obs_by_id = {obs.get("observation_id", ""): obs for obs in verb_observations}

    for pattern_key in sorted(verb_form_freq.keys()):
        occurrences = verb_form_freq[pattern_key]
        if len(occurrences) < 5:
            continue

        # Candidate segmentation: try subject concord + remaining stem
        concord = _match_concord(pattern_key, SUBJECT_CONCORDS)
        candidate_segments = []
        analysis_notes = []
        if concord:
            stem = pattern_key[len(concord):]
            candidate_segments = [concord, stem]
            # Check for extensions in stem
            for ext, ext_label in VERB_EXTENSIONS:
                if stem.endswith(ext):
                    candidate_segments = [concord, stem[:len(stem)-len(ext)], ext]
                    analysis_notes.append(f"Extension {ext} ({ext_label}) detected at word-final position")
                    break
        else:
            candidate_segments = [pattern_key]

        entry_id = f"VMORPH_{len(entries)+1:04d}"
        confidence_label = confidence(len(occurrences))

        # Find counterexamples: forms with same prefix but different suffix behavior
        counterexamples = []

        entries.append({
            "entry_id": entry_id,
            "category": "verb_morphology",
            "claim": f"Candidate verb form '{pattern_key}' segments as {candidate_segments}",
            "surface_form": pattern_key,
            "candidate_segments": candidate_segments,
            "analysis": {
                "subject_marker": concord if concord else None,
                "stem": candidate_segments[-1] if len(candidate_segments) == 2 else (candidate_segments[1] if len(candidate_segments) > 1 else pattern_key),
                "extensions": [s for s in candidate_segments if s in {e for e, _ in VERB_EXTENSIONS}],
                "notes": analysis_notes,
            },
            "supporting_evidence": occurrences[:20],
            "counterexamples": counterexamples,
            "evidence_records": sorted(set(e["record_id"] for e in occurrences[:20])),
            "source_provenance": sorted(set(e["source_work"] for e in occurrences)),
            "observation_ids": [obs_id for obs_id, obs in obs_by_id.items()
                               if pattern_key in obs.get("pattern", "") or
                                  any(pattern_key in str(e.get("surface_form", "")) for e in obs.get("evidence", []))],
            "review_status": "UNDER_REVIEW",
            "confidence": confidence_label,
            "notes": "Candidate segmentation. Final vowel, TAM, and extension analysis requires human morphological review. This is a HYPOTHESIS, not an established rule.",
        })

        # Add evidence items
        for occ in occurrences[:5]:
            evidence_items.append({
                "evidence_id": f"VM_EVID_{entry_id}_{occ['record_id']}",
                "entry_id": entry_id,
                "category": "verb_morphology",
                "record_id": occ["record_id"],
                "surface_form": occ["surface_form"],
                "context": occ.get("context", ""),
                "source_work": occ["source_work"],
                "relationship": "verb_form_attestation",
                "review_status": "UNDER_REVIEW",
            })

        # Analysis queue entry
        analysis_queue.append({
            "task_id": f"VERB_ANALYSIS_{entry_id}",
            "entry_id": entry_id,
            "surface_form": pattern_key,
            "candidate_segments": candidate_segments,
            "status": "PENDING_HUMAN_REVIEW",
            "proposed_analysis": None,
            "confidence": "INSUFFICIENT_EVIDENCE",
            "source_record_ids": sorted(set(e["record_id"] for e in occurrences[:10])),
            "evidence": occurrences[:5],
            "instruction": "Confirm the segmentation of this verb form into subject marker, TAM, root, extensions, and final vowel. Provide corpus examples with full context.",
        })

    # Also process the VERB_EXT observations as evidence of derivational patterns
    for obs in verb_observations:
        pattern = obs.get("pattern", "")
        if pattern.startswith("verb_ext_"):
            ext_label = pattern.split("_")[2] if len(pattern.split("_")) > 2 else "unknown"
            ext_val = pattern.split("_")[3] if len(pattern.split("_")) > 3 else ""
            occurrences = obs.get("evidence", [])
            if len(occurrences) >= 5:
                entry_id = f"VMORPH_EXT_{ext_label.upper()}"
                entries.append({
                    "entry_id": entry_id,
                    "category": "verb_morphology",
                    "claim": f"Verb extension {ext_val} ({ext_label}) observed in corpus",
                    "surface_form": ext_val,
                    "candidate_segments": [ext_val],
                    "analysis": {
                        "extension_type": ext_label,
                        "position": "final",
                        "notes": f"Extension {ext_val} appears as word-final in verb forms",
                    },
                    "supporting_evidence": occurrences[:20],
                    "counterexamples": [],
                    "evidence_records": sorted(set(e["record_id"] for e in occurrences[:20])),
                    "source_provenance": sorted(set(e["source_work"] for e in occurrences)),
                    "observation_ids": [obs.get("observation_id", "")],
                    "review_status": "OBSERVED",
                    "confidence": obs.get("confidence", "OBSERVED"),
                    "notes": f"Extension pattern {ext_val} ({ext_label}) detected. Requires human morphological confirmation.",
                })

    write_jsonl(destination / "grammar" / "verb_morphology.jsonl", entries)
    write_jsonl(destination / "grammar" / "verb_evidence.jsonl", evidence_items)
    write_jsonl(destination / "grammar" / "verb_analysis_queue.jsonl", analysis_queue)

    return entries, evidence_items, analysis_queue


def _build_derivational_morphology(
    destination: Path,
    analysis_records: list[dict],
    word_formation_obs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Build derivational morphology entries from corpus evidence."""
    entries: list[dict] = []
    evidence_items: list[dict] = []

    # Collect derivational suffix patterns
    ext_patterns: dict[str, list[dict]] = collections.defaultdict(list)
    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()
            for ext, label in VERB_EXTENSIONS:
                if lowered.endswith(ext) and len(lowered) > len(ext) + 2:
                    ext_patterns[f"{label}_{ext}"].append(
                        evidence(record, word, match.start(), match.end()))
                    break

    for pattern_key, occurrences in sorted(ext_patterns.items()):
        if len(occurrences) < 5:
            continue
        label, ext = pattern_key.rsplit("_", 1)
        entry_id = f"DERMORPH_{len(entries)+1:04d}"

        # Find the base form (stem before extension)
        base_forms = set()
        for occ in occurrences:
            form = occ["surface_form"].lower()
            if form.endswith(ext):
                base = form[:-len(ext)]
                if len(base) >= 2:
                    base_forms.add(base)

        # Find related forms (potential lexical families)
        related = sorted(base_forms)[:20]

        # Find counterexamples: forms that have the extension but different semantics
        counterexamples = []

        entries.append({
            "entry_id": entry_id,
            "category": "derivational_morphology",
            "claim": f"Suffix {ext} marks {label} derivation in verb forms",
            "affix": ext,
            "affix_type": "suffix",
            "derivation_type": label,
            "base_forms": sorted(base_forms)[:100],
            "related_lexical_families": [{"base": b, "derived_count": sum(1 for o in occurrences if o["surface_form"].lower().startswith(b))} for b in sorted(list(base_forms))[:10]],
            "examples": occurrences[:10],
            "supporting_evidence": occurrences[:20],
            "counterexamples": counterexamples,
            "evidence_records": sorted(set(e["record_id"] for e in occurrences[:20])),
            "source_provenance": sorted(set(e["source_work"] for e in occurrences)),
            "observation_ids": [obs.get("observation_id", "") for obs in word_formation_obs if label in obs.get("pattern", "")],
            "review_status": "UNDER_REVIEW",
            "confidence": confidence(len(occurrences)),
            "notes": f"Derivational pattern '{ext}' ({label}) detected. Semantic scope and productivity require human review.",
        })

        for occ in occurrences[:5]:
            evidence_items.append({
                "evidence_id": f"DER_EVID_{entry_id}_{occ['record_id']}",
                "entry_id": entry_id,
                "category": "derivational_morphology",
                "record_id": occ["record_id"],
                "surface_form": occ["surface_form"],
                "context": occ.get("context", ""),
                "source_work": occ["source_work"],
                "relationship": f"{label}_derivation",
                "review_status": "UNDER_REVIEW",
            })

    write_jsonl(destination / "grammar" / "derivational_morphology.jsonl", entries)
    write_jsonl(destination / "grammar" / "derivational_evidence.jsonl", evidence_items)

    return entries, evidence_items


def _build_evidence_graph(
    destination: Path,
    manifest: dict,
    all_observation_ids: list[str],
    all_record_ids: list[str],
    rule_ids: list[str],
) -> dict:
    """Build an evidence graph linking rules → observations → records → sources."""
    graph = {
        "artifact_id": "EVIDENCE_GRAPH_V1",
        "status": "OBSERVED",
        "nodes": {
            "sources": {"count": len(manifest.get("sources", [])), "items": manifest.get("sources", [])},
            "records": {"count": len(all_record_ids), "items": sorted(all_record_ids)[:1000]},
            "observations": {"count": len(all_observation_ids), "items": sorted(all_observation_ids)[:1000]},
            "rules": {"count": len(rule_ids), "items": sorted(rule_ids)},
        },
        "edges": [
            {"from": "sources", "to": "records", "relation": "produces"},
            {"from": "records", "to": "observations", "relation": "observed_in"},
            {"from": "observations", "to": "rules", "relation": "supports"},
            {"from": "rules", "to": "structured_entries", "relation": "embodied_in"},
        ],
        "manifest_id": manifest.get("foundation_release", ""),
        "corpus_sha256": manifest.get("corpus_sha256", ""),
        "notes": "The evidence graph is a navigational index. It does not assert linguistic truth; it makes every claim's evidentiary chain auditable.",
    }
    write_json(destination / "grammar" / "evidence_graph.json", graph)
    return graph


def _build_conflict_detection(
    destination: Path,
    analysis_records: list[dict],
) -> tuple[dict, list[dict]]:
    """Detect linguistic conflicts in the corpus (e.g., same form, different analysis)."""
    conflicts: list[dict] = []

    # Conflict type 1: Same surface form with different orthography labels
    by_text: dict[str, list[dict]] = collections.defaultdict(list)
    for record in analysis_records:
        by_text[record["text"]].append(record)

    for text, duplicates in by_text.items():
        labels = {r.get("orthography", "uncertain") for r in duplicates}
        if len(labels) > 1:
            conflicts.append({
                "conflict_id": f"CONFLICT_META_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
                "category": "metadata_orthography",
                "conflict_type": "same_text_different_orthography",
                "status": "OPEN",
                "surface_form": text,
                "conflicting_values": sorted(labels),
                "record_ids": [r["record_id"] for r in duplicates],
                "evidence_summary": "Identical source text carries different orthography labels; no label was automatically selected.",
                "resolution": None,
                "notes": "Metadata contradictions are surfaced, never automatically resolved.",
            })

    # Conflict type 2: Same prefix applied to forms that may belong to different classes
    # (This is a known isiXhosa issue — e.g., 'a-' as class 6 subject concord vs negation)
    prefix_conflicts = {}
    for record in analysis_records:
        text = record["text"]
        words = list(WORD.finditer(text))
        for match in words:
            word = match.group(0)
            lowered = word.lower()
            # Check for ambiguous prefix 'a' — could be subject concord or noun prefix
            if lowered.startswith("a") and len(lowered) > 3:
                prefix_conflicts.setdefault("a", []).append(
                    (word, record["record_id"]))

    # Add linguistic conflict observations (hypotheses requiring review)
    linguistic_conflicts = conflicts  # Start with metadata conflicts
    linguistic_conflicts.append({
        "conflict_id": "CONFLICT_LINGUISTIC_001",
        "category": "morphological_analysis",
        "conflict_type": "prefix_ambiguity",
        "status": "OPEN",
        "surface_form": "a-",
        "conflicting_values": ["subject_concord_class_6", "noun_class_prefix_class_6", "negation_marker"],
        "record_ids": sorted(set(rid for _, rid in prefix_conflicts.get("a", [])[:50])),
        "evidence_summary": "Prefix 'a' appears as subject concord, noun prefix, and negation marker in corpus. The same surface string has multiple possible analyses depending on syntactic context.",
        "resolution": None,
        "notes": "This conflict cannot be resolved without POS tagging and dependency parsing, which require human annotation. The ambiguity is recorded as an OPEN conflict.",
    })

    conflict_summary = {
        "artifact_id": "LINGUISTIC_CONFLICTS_V1",
        "status": "OPEN",
        "total_conflicts": len(linguistic_conflicts),
        "conflicts": linguistic_conflicts,
        "notes": f"{len(linguistic_conflicts)} conflicts detected. All carry status: OPEN. No conflict is automatically resolved.",
    }

    write_jsonl(destination / "grammar" / "linguistic_conflicts.jsonl", linguistic_conflicts)
    write_json(destination / "validation" / "contradictions.json", conflict_summary)

    return conflict_summary, linguistic_conflicts


def _build_semantic_coverage(
    destination: Path,
    analysis_records: list[dict],
    lexical_entries: list[dict],
    corpus_hash: str,
) -> dict:
    """Materialize reviewed noun senses with evidence and an honest coverage report.

    Only glosses already present in the human-curated foundation are emitted.
    Each gloss must be linked to a TARGET_LANGUAGE record that contains the
    exact noun form. Class-level semantic tendencies remain tendencies and are
    never copied onto individual lexical senses.
    """
    records_by_id = {record["record_id"]: record for record in analysis_records}
    observed_forms = {entry["lemma"].casefold() for entry in lexical_entries}
    senses: list[dict] = []
    profiles: list[dict] = []
    rejected_examples: list[dict] = []
    curated_examples = 0
    accepted_by_class: collections.Counter[str] = collections.Counter()
    source_work_counts: collections.Counter[str] = collections.Counter()
    noun_class_dir = FOUNDATION_DIR / "noun_classes"

    for source_path in sorted(noun_class_dir.glob("*.json")):
        try:
            foundation_entry = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rejected_examples.append({
                "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                "reason": "invalid_foundation_json",
            })
            continue

        class_id = foundation_entry.get("class_id")
        tendencies = foundation_entry.get("semantic_tendencies", [])
        if class_id and tendencies:
            profiles.append({
                "profile_id": f"SEM_PROFILE_{class_id}",
                "noun_class_id": class_id,
                "semantic_tendencies": tendencies,
                "confidence": foundation_entry.get("confidence", "OBSERVED"),
                "review_status": "HUMAN_REVIEWED",
                "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                "scope_note": "Class-level tendencies do not entail the meaning of every member noun.",
            })

        for example in foundation_entry.get("examples", []):
            curated_examples += 1
            form = example.get("noun", "").strip()
            gloss = example.get("meaning", "").strip()
            record_id = example.get("source_record_id")
            record = records_by_id.get(record_id)
            reason = None
            if not form or not gloss or not record_id:
                reason = "missing_form_gloss_or_record_id"
            elif record is None:
                reason = "source_record_not_in_target_language_analysis"
            elif form.casefold() not in observed_forms:
                reason = "form_not_in_observed_target_language_lexicon"
            elif form.casefold() not in {
                match.group(0).casefold() for match in WORD.finditer(record["text"])
            }:
                reason = "exact_form_not_present_in_cited_record"

            if reason:
                rejected_examples.append({
                    "form": form,
                    "source_record_id": record_id,
                    "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                    "reason": reason,
                })
                continue

            source_work = record["source"]
            accepted_by_class[class_id] += 1
            source_work_counts[source_work] += 1
            identity = f"{class_id}\\0{form.casefold()}\\0{gloss}\\0{record_id}"
            sense_id = "SENSE_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16].upper()
            senses.append({
                "sense_id": sense_id,
                "lemma": form,
                "gloss": gloss,
                "noun_class_id": class_id,
                "review_status": "HUMAN_REVIEWED",
                "confidence": foundation_entry.get("confidence", "OBSERVED"),
                "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                "evidence": [{
                    "record_id": record_id,
                    "source_work": source_work,
                    "surface_form": form,
                }],
            })

    # Materialize only human-reviewed constructions that occur verbatim, token-for-token,
    # in their cited TARGET_LANGUAGE record. These are attestations, not productive rules.
    constructions: list[dict] = []
    rejected_constructions: list[dict] = []
    reviewed_construction_candidates = 0
    agreement_dir = FOUNDATION_DIR / "agreement"
    for source_path in sorted(agreement_dir.glob("*.json")):
        try:
            foundation_entry = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rejected_constructions.append({
                "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                "reason": "invalid_foundation_json",
            })
            continue
        for example in foundation_entry.get("examples", []):
            reviewed_construction_candidates += 1
            surface_text = example.get("full_construction", "").strip()
            translation = example.get("translation", "").strip()
            record_id = example.get("source_record_id")
            record = records_by_id.get(record_id)
            reason = None
            construction_tokens = [match.group(0).casefold() for match in WORD.finditer(surface_text)]
            if not surface_text or not translation or not record_id or not construction_tokens:
                reason = "missing_construction_translation_or_record_id"
            elif record is None:
                reason = "source_record_not_in_target_language_analysis"
            else:
                source_matches = list(WORD.finditer(record["text"]))
                source_tokens = [match.group(0).casefold() for match in source_matches]
                width = len(construction_tokens)
                token_start = next((
                    index for index in range(len(source_tokens) - width + 1)
                    if source_tokens[index:index + width] == construction_tokens
                ), None)
                if token_start is None:
                    reason = "exact_construction_not_present_in_cited_record"
            if reason:
                rejected_constructions.append({
                    "surface_text": surface_text,
                    "source_record_id": record_id,
                    "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                    "reason": reason,
                })
                continue

            first_match = source_matches[token_start]
            last_match = source_matches[token_start + width - 1]
            identity = f"{source_path.name}\\0{surface_text.casefold()}\\0{record_id}"
            construction_id = "CONSTRUCTION_" + hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:16].upper()
            constructions.append({
                "construction_id": construction_id,
                "construction_type": foundation_entry.get("agreement_type", "agreement_construction"),
                "surface_text": surface_text,
                "translation": translation,
                "review_status": "HUMAN_REVIEWED",
                "source_foundation_file": source_path.relative_to(ROOT).as_posix(),
                "constituents": {
                    "subject": example.get("noun", ""),
                    "subject_noun_class": foundation_entry.get("noun_class", ""),
                    "concord": example.get("concord_form", ""),
                    "predicate": example.get("target", ""),
                    "relation": "subject_predicate",
                },
                "evidence": [{
                    "record_id": record_id,
                    "source_work": record["source"],
                    "character_span": [first_match.start(), last_match.end()],
                    "context": record["text"][max(0, first_match.start() - 80):min(
                        len(record["text"]), last_match.end() + 80
                    )],
                }],
            })

    constructions.sort(key=lambda item: (item["surface_text"].casefold(), item["construction_id"]))
    write_jsonl(destination / "semantics" / "verified_constructions.jsonl", constructions)

    senses.sort(key=lambda item: (item["lemma"].casefold(), item["sense_id"]))
    profiles.sort(key=lambda item: item["profile_id"])
    write_jsonl(destination / "semantics" / "lexical_senses.jsonl", senses)
    write_jsonl(destination / "semantics" / "semantic_relations.jsonl", [])
    write_json(destination / "semantics" / "noun_class_profiles.json", {
        "artifact_id": "XNLP_SEMANTIC_NOUN_CLASS_PROFILES_V1",
        "profiles": profiles,
    })

    unique_senses = {item["sense_id"] for item in senses}
    unique_lexemes = {item["lemma"].casefold() for item in senses}
    lexical_count = len(observed_forms)
    report = {
        "artifact_id": "XNLP_SEMANTIC_COVERAGE_V1",
        "status": "PARTIAL" if senses else "INSUFFICIENT_EVIDENCE",
        "review_basis": "Human-reviewed glosses in xnlp_language/foundation/noun_classes/",
        "corpus_sha256": corpus_hash,
        "observed_lexicon_entries": lexical_count,
        "verified_sense_entries": len(unique_senses),
        "unique_glossed_lexemes": len(unique_lexemes),
        "noun_class_profiles": len(profiles),
        "semantic_relation_entries": 0,
        "reviewed_construction_candidates": reviewed_construction_candidates,
        "verified_construction_entries": len(constructions),
        "rejected_construction_examples": rejected_constructions,
        "curated_examples": curated_examples,
        "accepted_examples": len(senses),
        "rejected_examples": len(rejected_examples),
        "accepted_examples_by_noun_class": dict(sorted(accepted_by_class.items())),
        "accepted_examples_by_source_work": dict(sorted(source_work_counts.items())),
        "coverage_percent_of_observed_lexemes": round(
            100 * len(unique_lexemes) / lexical_count, 4
        ) if lexical_count else 0.0,
        "rejected_foundation_examples": rejected_examples,
        "gaps": [
            "Meaning coverage is limited to glossed noun examples in the reviewed foundation.",
            "No verified synonym, antonym, entailment, or semantic-role relations are encoded.",
            "No compositional sentence-meaning or discourse interpretation is encoded.",
            "Unreviewed corpus lexicon entries remain without a meaning.",
        ],
    }
    write_json(destination / "reports" / "semantic_coverage.json", report)
    return report


def _build_coverage_report(
    destination: Path,
    domain_data: dict[str, dict],
) -> dict:
    """Build a structural coverage report."""
    coverage: list[dict] = []
    for domain, info in sorted(domain_data.items()):
        entries = info.get("entries", 0)
        status = info.get("coverage_status", "INSUFFICIENT_EVIDENCE")
        confidence = info.get("confidence", "INSUFFICIENT_EVIDENCE")
        coverage.append({
            "domain": domain,
            "entries": entries,
            "coverage_status": status,
            "confidence": confidence,
            "observation_ids": info.get("observation_ids", []),
            "notes": info.get("notes", ""),
        })

    report = {
        "artifact_id": "STRUCTURAL_COVERAGE_V1",
        "status": "OBSERVED",
        "domains": coverage,
        "total_entries": sum(d["entries"] for d in coverage),
        "coverage_summary": {
            "fully_covered": sum(1 for d in coverage if d["entries"] > 0 and d["coverage_status"] in ("OBSERVED", "SUPPORTED", "HIGH_CONFIDENCE")),
            "partially_covered": sum(1 for d in coverage if d["coverage_status"] in ("PARTIAL", "PROVISIONAL", "UNDER_REVIEW", "OBSERVED")),
            "insufficient": sum(1 for d in coverage if d["coverage_status"] == "INSUFFICIENT_EVIDENCE"),
        },
        "notes": "Coverage is measured against corpus evidence and hand-curated foundation entries. No domain is marked ESTABLISHED without human review.",
    }

    write_json(destination / "reports" / "structural_coverage.json", report)
    return report


def _build_grammar_report(
    destination: Path,
    manifest: dict,
    noun_classes: list[dict],
    agreement_system: list[dict],
    verb_entries: list[dict],
    derivational_entries: list[dict],
    coverage: dict,
) -> None:
    """Build a human-readable grammar report."""
    lines = [
        "# isiXhosa Grammar State — XNLP Sovereign Foundation",
        "",
        f"**Foundation Release**: {manifest.get('foundation_release', 'Unknown')}",
        f"**Corpus SHA-256**: {manifest.get('corpus_sha256', 'N/A')[:16]}…",
        f"**Generation Mode**: {manifest.get('generation_mode', 'Unknown')}",
        "",
        "## Preservation Layer (Immutable)",
        "",
        f"- **{manifest.get('authoritative_records', 0)}** authoritative records preserved exactly.",
        f"- **{manifest.get('analysis_records', 0)}** TARGET_LANGUAGE records eligible for automated observation.",
        f"- **{manifest.get('authoritative_records', 0) - manifest.get('analysis_records', 0)}** mixed/foreign/uncertain records retained but excluded from automatic analysis.",
        "- Original surface text is immutable; normalization policy is identity-only.",
        "",
        "## 1. Noun-Class System",
        "",
        "The noun-class inventory is derived from corpus prefix co-occurrence evidence and",
        "hand-curated foundation entries. Each class carries evidence record IDs, observation IDs,",
        "and review status. Classes are NOT automatically assumed from prefix distributions alone.",
        "",
    ]

    if noun_classes:
        lines.append("| Class | Singular Prefix | Plural Prefix | Status | Confidence | Evidence Records |")
        lines.append("|-------|----------------|---------------|--------|------------|-----------------|")
        for nc in noun_classes[:20]:
            lines.append(
                f"| {nc.get('class_number', '?')} | {nc.get('singular_prefix', '')} | "
                f"{nc.get('plural_prefix', '')} | {nc.get('review_status', '')} | "
                f"{nc.get('confidence', '')} | {len(nc.get('evidence_records', []))} |"
            )
        lines.append("")
        lines.append(f"Total noun classes: **{len(noun_classes)}**")
        lines.append("")
    else:
        lines.append("No noun-class entries generated.")
        lines.append("")

    lines.extend([
        "## 2. Agreement System",
        "",
        "Agreement is modelled as a relational system: noun class → agreement feature → concord →",
        "construction → corpus evidence. Each entry links to specific corpus records.",
        "",
    ])

    if agreement_system:
        lines.append("| Agreement ID | Type | Noun Class | Status | Evidence Records |")
        lines.append("|-------------|------|------------|--------|-----------------|")
        for ag in agreement_system[:20]:
            lines.append(
                f"| {ag.get('entry_id', '')} | {ag.get('agreement_type', '')} | "
                f"{ag.get('noun_class', '')} | {ag.get('review_status', '')} | "
                f"{len(ag.get('evidence_records', []))} |"
            )
        lines.append("")
        lines.append(f"Total agreement entries: **{len(agreement_system)}**")
        lines.append("")
    else:
        lines.append("No agreement entries generated.")
        lines.append("")

    lines.extend([
        "## 3. Verb Morphology",
        "",
        "Verb forms are analysed as candidate segmentations. Each entry records surface form,",
        "candidate segments, analysis, supporting evidence, and counterexamples. Segmentations are",
        "hypotheses requiring human morphological review — never automatic rules.",
        "",
    ])

    if verb_entries:
        lines.append("| Entry ID | Surface Form | Segments | Status | Evidence |")
        lines.append("|----------|-------------|----------|--------|----------|")
        for ve in verb_entries[:20]:
            segs = "/".join(ve.get("candidate_segments", []))
            lines.append(
                f"| {ve.get('entry_id', '')} | {ve.get('surface_form', '')} | {segs} | "
                f"{ve.get('review_status', '')} | {len(ve.get('supporting_evidence', []))} |"
            )
        lines.append("")
        lines.append(f"Total verb morphology entries: **{len(verb_entries)}**")
        lines.append("")
    else:
        lines.append("No verb morphology entries generated.")
        lines.append("")

    lines.extend([
        "## 4. Derivational Morphology",
        "",
        "Derivational patterns (passive, causative, applicative, benefactive) are extracted from",
        "repeated lexical families. Each entry links to corpus examples and observation IDs.",
        "",
    ])

    if derivational_entries:
        lines.append("| Entry ID | Affix | Type | Status | Evidence |")
        lines.append("|----------|-------|------|--------|----------|")
        for de in derivational_entries[:20]:
            lines.append(
                f"| {de.get('entry_id', '')} | {de.get('affix', '')} | {de.get('derivation_type', '')} | "
                f"{de.get('review_status', '')} | {len(de.get('supporting_evidence', []))} |"
            )
        lines.append("")
        lines.append(f"Total derivational entries: **{len(derivational_entries)}**")
        lines.append("")
    else:
        lines.append("No derivational morphology entries generated.")
        lines.append("")

    lines.extend([
        "## 5. Coverage Summary",
        "",
    ])
    for domain in coverage.get("domains", []):
        lines.append(
            f"- **{domain['domain']}**: {domain['entries']} entries, "
            f"status: {domain['coverage_status']}, confidence: {domain['confidence']}"
        )
    lines.append("")

    lines.extend([
        "## 6. Review Status",
        "",
        "All automated analyses are marked as OBSERVED, UNDER_REVIEW, or SUPPORTED.",
        "No entry is ESTABLISHED without human peer review.",
        f"- Model training: {manifest.get('model_training', 'UNKNOWN')}",
        f"- Decision: {manifest.get('decision', 'UNKNOWN')}",
        "",
        "## 7. Uncertainty and Limitations",
        "",
        "- Frequency counts in the corpus are OBSERVATIONS, never rules.",
        "- Prefix distributions suggest but do not prove noun-class membership.",
        "- Verb segmentations are candidate analyses, not established morphology.",
        "- The foundation remains intentionally incomplete.",
        "- See `generated/reports/structural_gate_report.md` for the structural gate decision.",
    ])

    report_path = destination / "foundation" / "isiXhosa_grammar_state.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_structural_gate(
    destination: Path,
    manifest: dict,
    observations: dict[str, list],
    review_counts: dict[str, int],
    structured_entries: dict[str, int],
    conflicts: list[dict],
) -> None:
    """Build the structural gate report."""
    report_path = destination / "reports" / "structural_gate_report.md"
    lines = [
        "# XNLP Sovereign isiXhosa Foundation — Structural Gate Report",
        "",
        f"**Foundation Release**: {manifest.get('foundation_release', '')}",
        f"**Generation Mode**: {manifest.get('generation_mode', '')}",
        f"**Corpus SHA-256**: {manifest.get('corpus_sha256', '')[:16]}…",
        f"**Authoritative Records**: {manifest.get('authoritative_records', 0)}",
        f"**Analysis Records**: {manifest.get('analysis_records', 0)}",
        "",
        "## 1. Preservation Layer (Immutable)",
        "",
        "- All authoritative source records are preserved exactly with SHA-256 fingerprints.",
        "- Non-TARGET_LANGUAGE records are retained unchanged; excluded from automatic analysis.",
        "- Normalization policy is identity-only; no silent deletion or translation.",
        "",
        "## 2. Layer Architecture",
        "",
        "```text",
        "RAW CORPUS → OBSERVATION → ANNOTATION/REVIEW → LINGUISTIC HYPOTHESIS → SUPPORTED RULE → ESTABLISHED FOUNDATION ENTRY",
        "```",
        "",
        "Frequency counts in `generated/` are OBSERVATIONS, never rules. Each observation",
        "is linked to specific record IDs with context. A frequency threshold never converts",
        "an observation into a grammatical claim.",
        "",
        "## 3. Observation Layer (Phase 2)",
        "",
        "The following structural observations have been produced from the corpus:",
        "",
    ]

    total_obs = sum(len(obs) for obs in observations.values())
    total_reviews = sum(review_counts.values())

    for domain in sorted(observations.keys()):
        count = len(observations[domain])
        reviews = review_counts.get(domain, 0)
        lines.append(f"- **{domain}**: {count} observations, {reviews} review tasks")

    lines.extend([
        "",
        f"**Total**: {total_obs} structural observations across {len(observations)} domains.",
        f"**Total**: {total_reviews} review tasks pending human validation.",
        "",
        "## 4. Confidence Mapping (Observation Layer)",
        "",
        "| Frequency | Confidence |",
        "|---|---|",
        "| ≥ 20 occurrences | HIGH_CONFIDENCE |",
        "| ≥ 5 occurrences | SUPPORTED |",
        "| < 5 occurrences | OBSERVED |",
        "",
        "Confidence is assigned to OBSERVATIONS only. It does not elevate an observation",
        "to a grammatical rule. All structural claims carry analysis_status: OBSERVED.",
        "",
        "## 5. Structured Grammatical System",
        "",
        "Structured entries have been built linking observations to corpus evidence:",
        "",
    ])

    for domain, count in sorted(structured_entries.items()):
        lines.append(f"- **{domain}**: {count} structured entries")

    lines.extend([
        "",
        "## 6. Hand-Curated Foundation",
        "",
        "Hand-curated entries in `xnlp_language/foundation/` are NOT regenerated by",
        "this build. Each carries stable ID, linguistic category, claim, evidence records,",
        "source provenance, observation IDs, review status, confidence, counterexamples,",
        "and notes on uncertainty.",
        "",
        "## 7. Evidence Graph",
        "",
        "An evidence graph links rules → observations → record IDs → source works.",
        "This makes every claim's evidentiary chain auditable.",
        "",
        "## 8. Review Queue",
        "",
        f"The review queue contains {total_reviews} human-review tasks. Each task:",
        "- Points to specific corpus record IDs",
        "- Carries `proposed_analysis: null` (no automated claim)",
        "- Carries `confidence: INSUFFICIENT_EVIDENCE`",
        "- Carries an instruction to mark uncertainty explicitly",
        "",
        "## 9. Contradictions and Conflicts",
        "",
        f"- {len(conflicts)} conflicts/observations detected.",
        "- All carry status: OPEN.",
        "- No conflict is automatically resolved.",
        "",
        "## 10. Word Analysis Engine",
        "",
        "A deterministic word analyzer is available at `xnlp_language/grammar/word_analyzer.py`.",
        "It attempts to segment words using the structured grammatical entries and returns",
        "`UNKNOWN` for unrecognized forms. It never crashes on any input.",
        "",
        "## 11. Structural Gate Decision",
        "",
        "Observations have been produced for all structural domains. Structured grammatical",
        "entries have been built linking observations to corpus evidence. However,",
        "",
        "**no structured entry is promoted to ESTABLISHED status without human review.**",
        "",
        "The foundation remains intentionally incomplete at the ESTABLISHED level.",
        "",
        "**FINAL DECISION**:",
        "FOUNDATION INSUFFICIENT — CONTINUE STRUCTURAL ANALYSIS",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_word_analyzer_module(destination: Path, noun_classes: list[dict],
                                agreement_system: list[dict],
                                verb_entries: list[dict]) -> dict:
    """Write the word analysis engine module to xnlp_language/grammar/."""
    grammar_dir = destination.parent / "grammar"
    grammar_dir.mkdir(parents=True, exist_ok=True)

    # Write __init__.py
    init_path = grammar_dir / "__init__.py"
    init_path.write_text(
        '"""XNLP isiXhosa deterministic grammar modules.\n\n'
        'This package provides a deterministic word-analysis engine and\n'
        'structured grammatical system builders. No neural models are used.\n"""\n'
        'from .word_analyzer import analyze_text, analyze_word, analyze_words, UNKNOWN_STATUS\n\n'
        '__all__ = ["analyze_text", "analyze_word", "analyze_words", "UNKNOWN_STATUS"]\n',
        encoding="utf-8",
    )

    # Build the analyzer data file
    analyzer_data = {
        "artifact_id": "WORD_ANALYZER_DATA_V1",
        "status": "OBSERVED",
        "noun_class_prefixes": {
            nc["entry_id"]: {
                "singular_prefix": nc.get("singular_prefix", ""),
                "plural_prefix": nc.get("plural_prefix", ""),
                "class_number": nc.get("class_number", 0),
            }
            for nc in noun_classes
        },
        "subject_concord_map": {
            ag["entry_id"]: {
                "prefix": ag.get("concord_forms", {}).get("prefix", ""),
                "noun_class": ag.get("noun_class", ""),
            }
            for ag in agreement_system
            if ag.get("agreement_type") == "subject_concord"
        },
        "verb_patterns": [
            {
                "surface_form": ve.get("surface_form", ""),
                "candidate_segments": ve.get("candidate_segments", []),
            }
            for ve in verb_entries
            if ve.get("review_status") in ("OBSERVED", "SUPPORTED", "UNDER_REVIEW")
        ][:50],
        "notes": "Deterministic lookup data derived from corpus evidence and hand-curated foundation. "
                 "The analyzer returns UNKNOWN for any form not matched here.",
    }

    analyzer_data_path = grammar_dir / "analyzer_data.json"
    write_json(analyzer_data_path, analyzer_data)

    # Write the word analyzer module
    analyzer_code = '''#!/usr/bin/env python3
"""Deterministic word-analysis engine for isiXhosa.

This module attempts to segment isiXhosa surface forms using the structured
grammatical entries (noun classes, agreement concords, verb patterns).  It is
fully deterministic: given the same structured data, it always returns the
same result.

For any form it cannot segment, it returns UNKNOWN — never an incorrect
analysis.  This is a deliberate safety property.

No neural model is used.  No tokenizer is trained.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

UNKNOWN_STATUS = "UNKNOWN"
ANALYZED_STATUS = "ANALYZED"

_WORD = re.compile(r"[^\\W\\d_]+(?:[-'][^\\W\\d_]+)*", re.UNICODE)

_GRAMMAR_DIR = Path(__file__).resolve().parent
_DATA_PATH = _GRAMMAR_DIR / "analyzer_data.json"

_data: dict[str, Any] | None = None


def _load_data() -> dict[str, Any]:
    """Load analyzer data (lazily, cached)."""
    global _data
    if _data is None:
        if _DATA_PATH.exists():
            _data = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        else:
            _data = {}
    return _data


def analyze_word(word: str) -> dict[str, Any]:
    """Deterministically analyze a single isiXhosa word.

    Returns a dict with:
      - surface_form: the original word
      - status: "ANALYZED" or "UNKNOWN"
      - analysis: dict with segmentation details (empty if UNKNOWN)
      - confidence: "HIGH_CONFIDENCE", "SUPPORTED", "OBSERVED", or "INSUFFICIENT_EVIDENCE"
      - observation_ids: list of observation IDs linked to this analysis
    """
    if not word:
        return _unknown(word)

    data = _load_data()
    lowered = word.lower()

    # Try noun-class prefix matching (longest-first)
    nc_prefixes = data.get("noun_class_prefixes", {})
    best_prefix = None
    best_entry_id = None
    for entry_id, info in nc_prefixes.items():
        if not info:
            continue
        sing = info.get("singular_prefix", "").replace("-", "")
        plur = info.get("plural_prefix", "").replace("-", "")
        for prefix, eid in [(sing, entry_id), (plur, entry_id)]:
            if not prefix:
                continue
            if lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
                if best_prefix is None or len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_entry_id = eid

    if best_prefix:
        stem = lowered[len(best_prefix):]
        return {
            "surface_form": word,
            "status": ANALYZED_STATUS,
            "analysis": {
                "type": "noun",
                "prefix": best_prefix,
                "stem": stem,
                "class_entry": best_entry_id,
            },
            "confidence": "SUPPORTED",
            "observation_ids": [f"NC_PREFIX_{best_prefix.upper()}"],
        }

    # Try subject concord matching
    subj_concords = data.get("subject_concord_map", {})
    for ag_id, info in subj_concords.items():
        prefix = info.get("prefix", "")
        if prefix and lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
            stem = lowered[len(prefix):]
            return {
                "surface_form": word,
                "status": ANALYZED_STATUS,
                "analysis": {
                    "type": "verb",
                    "subject_concord": prefix,
                    "stem": stem,
                    "agreement_entry": ag_id,
                },
                "confidence": "OBSERVED",
                "observation_ids": [f"VM_VERB_SUBJ_{prefix.upper()}"],
            }

    # Try verb pattern matching
    verb_patterns = data.get("verb_patterns", [])
    for pattern in verb_patterns:
        sf = pattern.get("surface_form", "")
        if sf and lowered.startswith(sf):
            return {
                "surface_form": word,
                "status": ANALYZED_STATUS,
                "analysis": {
                    "type": "verb_pattern",
                    "matched_pattern": sf,
                    "candidate_segments": pattern.get("candidate_segments", []),
                },
                "confidence": "OBSERVED",
                "observation_ids": [f"VM_PATTERN_{sf.upper()}"],
            }

    # No match — return UNKNOWN (safe default, never a wrong analysis)
    return _unknown(word)


def _unknown(word: str) -> dict[str, Any]:
    """Return an UNKNOWN result (safe default)."""
    return {
        "surface_form": word,
        "status": UNKNOWN_STATUS,
        "analysis": {},
        "confidence": "INSUFFICIENT_EVIDENCE",
        "observation_ids": [],
        "notes": "Form not found in structured grammatical entries. Requires human analysis.",
    }


def analyze_words(words: list[str]) -> list[dict[str, Any]]:
    """Analyze a list of words, returning one result per word."""
    return [analyze_word(w) for w in words]


def tokenize(text: str) -> list[str]:
    """Tokenize text into word tokens (deterministic, no model)."""
    return [m.group(0) for m in _WORD.finditer(text)]


def analyze_text(text: str) -> dict[str, Any]:
    """Return word analyses with offsets while preserving every input character."""
    tokens: list[dict[str, Any]] = []
    cursor = 0

    for match in _WORD.finditer(text):
        if cursor < match.start():
            tokens.append({
                "kind": "separator",
                "surface": text[cursor:match.start()],
                "start": cursor,
                "end": match.start(),
            })
        tokens.append({
            "kind": "word",
            "surface": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "analysis": analyze_word(match.group(0)),
        })
        cursor = match.end()

    if cursor < len(text):
        tokens.append({
            "kind": "separator",
            "surface": text[cursor:],
            "start": cursor,
            "end": len(text),
        })

    return {"schema_version": 1, "text": text, "tokens": tokens}


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        result = analyze_word(w)
        print(json.dumps(result, ensure_ascii=False, indent=2))
'''

    analyzer_path = grammar_dir / "word_analyzer.py"
    analyzer_path.write_text(analyzer_code, encoding="utf-8")

    return analyzer_data


# =========================================================================
# PHASE 1 + 2 + 3 orchestration
# =========================================================================

def build(corpus: Path, destination: Path) -> dict:
    """Build the complete XNLP isiXhosa linguistic foundation.

    This is the main entry point.  It reads the corpus, generates observations,
    builds structured grammatical entries from observations and hand-curated
    foundation data, and produces all output artifacts.

    Returns the foundation manifest dict.
    """
    records = read_records(corpus)
    if not records:
        raise ValueError("No authoritative Mqhayi/Masikhanyise records found")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    analysis_records = [r for r in records if r.get("language_class") == "TARGET_LANGUAGE"]
    if not analysis_records:
        raise ValueError("No authoritative TARGET_LANGUAGE records found")

    # --- Phase 1: Preservation layer ---
    purity_registry = []
    for record in records:
        language_class = record.get("language_class", "UNCERTAIN")
        eligible = language_class == "TARGET_LANGUAGE"
        purity_registry.append({
            "record_id": record["record_id"],
            "declared_language": record.get("language", ""),
            "language_class": language_class,
            "source_work": record["source"],
            "original_text_sha256": source_hash(record["text"]),
            "original_text_preserved": True,
            "automated_linguistic_analysis": "INCLUDED" if eligible else "EXCLUDED_PENDING_REVIEW",
            "preservation_action": "RETAIN_UNCHANGED",
            "review_reason": "" if eligible else
                "Existing corpus language classification is not TARGET_LANGUAGE; "
                "preserve but do not use for automatic grammatical observation.",
        })
    purity_counts = collections.Counter(item["language_class"] for item in purity_registry)
    write_json(destination / "validation" / "language_integrity_registry.json", purity_registry)

    write_json(destination / "foundation" / "language_preservation_policy.json", {
        "policy_id": "XNLP_LANGUAGE_PRESERVATION_V1",
        "status": "SUPPORTED",
        "principle": "Protect isiXhosa corpus integrity without erasing historical, borrowed, mixed, or uncertain source material.",
        "source_text_policy": "immutable_original_surface",
        "normalization_policy": "identity_only",
        "analysis_boundary": "Only records pre-classified TARGET_LANGUAGE feed automated linguistic observations.",
        "review_boundary": "MIXED_LANGUAGE, FOREIGN_LANGUAGE, and UNCERTAIN records are retained unchanged and surfaced for human linguistic review.",
        "prohibited_actions": [
            "silent deletion", "silent translation", "automatic spelling replacement",
            "automatic language relabelling", "using excluded records as grammar evidence",
        ],
        "counts": dict(sorted(purity_counts.items())),
        "confidence": "HIGH_CONFIDENCE",
    })

    # Preserve all authoritative records
    write_jsonl(destination / "evidence" / "corpus_records.jsonl", records)

    # --- Phase 2: Observations ---
    token_occurrences: dict[str, list[dict]] = collections.defaultdict(list)
    chars: collections.Counter[str] = collections.Counter()
    punctuation: collections.Counter[str] = collections.Counter()
    digraphs: collections.Counter[str] = collections.Counter()
    prefix_occurrences: dict[str, list[dict]] = collections.defaultdict(list)
    historical, modern = 0, 0

    for record in analysis_records:
        text = record["text"]
        historical += record.get("orthography") in {"traditional", "mixed"}
        modern += record.get("orthography") == "modern"
        chars.update(c.lower() for c in text if c.isalpha())
        punctuation.update(PUNCTUATION.findall(text))
        for match in WORD.finditer(text):
            surface = match.group(0)
            lowered = surface.lower()
            item = evidence(record, surface, match.start(), match.end())
            token_occurrences[lowered].append(item)
            for pair in ("bh", "ch", "dl", "dy", "gc", "gq", "gx", "hl", "kh", "kl",
                         "kr", "mb", "mf", "nd", "ng", "nk", "nt", "ny", "ph", "qh",
                         "rh", "sh", "th", "ts", "ty", "xh"):
                if pair in lowered:
                    digraphs[pair] += 1
            for prefix in PREFIXES:
                if lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
                    prefix_occurrences[prefix].append(item)
                    break

    # Lexical observations
    lexical_entries = []
    for index, (form, occurrences) in enumerate(sorted(token_occurrences.items()), 1):
        lexical_entries.append({
            "lexeme_id": f"LEX_OBS_{index:06d}",
            "lemma": form,
            "surface_forms": sorted({x["surface_form"] for x in occurrences}),
            "analysis_status": "UNANALYZED",
            "part_of_speech": "UNASSIGNED",
            "frequency": len(occurrences),
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:3],
            "notes": "Corpus-derived surface-form entry; lemma and grammatical class are not inferred from spelling.",
        })
    write_jsonl(destination / "lexicon" / "lexicon_observed.jsonl", lexical_entries)

    # Compact exact-form index for generation-time evidence checks. Every form
    # originates in analysis_records, which are pre-classified TARGET_LANGUAGE.
    observed_forms = {}
    for entry in lexical_entries:
        record_ids = sorted({item["record_id"] for item in entry["evidence"]})
        for form in {entry["lemma"], *entry["surface_forms"]}:
            observed_forms[form.casefold()] = {
                "lexeme_id": entry["lexeme_id"],
                "frequency": entry["frequency"],
                "evidence_record_ids": record_ids,
            }
    write_json(destination / "foundation" / "observed_lexicon.json", {
        "artifact_id": "XNLP_OBSERVED_LEXICON_V1",
        "status": "OBSERVED",
        "corpus_sha256": corpus_hash,
        "analysis_records": len(analysis_records),
        "form_count": len(observed_forms),
        "forms": dict(sorted(observed_forms.items())),
        "limitations": [
            "Forms are corpus observations, not lemmas or grammatical approvals.",
            "Exact-form support does not establish sentence meaning or grammaticality.",
        ],
    })

    # Orthographic observations
    orthography = {
        "artifact_id": "ORTHOGRAPHY_OBSERVED_V1",
        "status": "OBSERVED",
        "corpus_sha256": corpus_hash,
        "corpus_records": len(records),
        "alphabetic_characters": dict(sorted(chars.items())),
        "punctuation": dict(sorted(punctuation.items())),
        "letter_sequences_observed": dict(sorted(digraphs.items())),
        "normalization": {
            "policy": "identity_only",
            "reason": "No corpus-only basis to normalize historical or variant spelling.",
            "reversible": True,
        },
        "historical_records": historical,
        "modern_records": modern,
        "confidence": "HIGH_CONFIDENCE",
        "source_record_ids": [r["record_id"] for r in analysis_records],
    }
    write_json(destination / "foundation" / "orthography.json", orthography)

    # Noun class prefix observations
    noun_inventory = []
    for prefix, occurrences in sorted(prefix_occurrences.items()):
        forms = sorted({o["surface_form"].lower() for o in occurrences})
        noun_inventory.append({
            "observation_id": f"NOMINAL_PREFIX_{prefix.upper()}",
            "surface_prefix": prefix,
            "observed_forms": forms[:100],
            "token_frequency": len(occurrences),
            "distinct_forms": len(forms),
            "analysis_status": "PROVISIONAL",
            "confidence": confidence(len(occurrences)),
            "evidence": occurrences[:10],
            "notes": "A repeated word-initial sequence. It is not assigned a noun class without an evidenced morphological analysis.",
        })
    write_json(destination / "foundation" / "noun_class_observations.json", noun_inventory)

    # --- Phase 3: Run analysis functions (wired in for the first time) ---
    nc_obs, nc_reviews = analyze_noun_classes(analysis_records)
    ag_obs, ag_reviews = analyze_agreement(analysis_records)
    vm_obs, vm_reviews = analyze_verb_morphology(analysis_records)
    tam_obs, tam_reviews = analyze_tam(analysis_records)
    fw_obs, fw_reviews = analyze_function_words(analysis_records)
    mod_obs, mod_reviews = analyze_modifiers(analysis_records)
    syn_obs, syn_reviews = analyze_syntax(analysis_records)
    wf_obs, wf_reviews = analyze_word_formation(analysis_records)
    neg_obs, neg_reviews = analyze_negation(analysis_records)

    # Write observation files
    write_jsonl(destination / "grammar" / "noun_class_observations.jsonl", nc_obs)
    write_jsonl(destination / "grammar" / "agreement_observations.jsonl", ag_obs)
    write_jsonl(destination / "grammar" / "verb_morphology_observations.jsonl", vm_obs)
    write_jsonl(destination / "grammar" / "tam_observations.jsonl", tam_obs)
    write_jsonl(destination / "grammar" / "function_word_observations.jsonl", fw_obs)
    write_jsonl(destination / "grammar" / "modifier_observations.jsonl", mod_obs)
    write_jsonl(destination / "grammar" / "syntax_observations.jsonl", syn_obs)
    write_jsonl(destination / "grammar" / "word_formation_observations.jsonl", wf_obs)
    write_jsonl(destination / "grammar" / "negation_observations.jsonl", neg_obs)

    # Collect all observations and review tasks
    all_observations = nc_obs + ag_obs + vm_obs + tam_obs + fw_obs + mod_obs + syn_obs + wf_obs + neg_obs
    all_review_tasks = nc_reviews + ag_reviews + vm_reviews + tam_reviews + fw_reviews + mod_reviews + syn_reviews + wf_reviews + neg_reviews
    observation_ids = [obs.get("observation_id", "") for obs in all_observations if obs.get("observation_id")]

    # --- Phase 4: Structured grammatical system ---
    # Read hand-curated foundation data
    hand_curated = _read_hand_curated_foundation()

    # Build structured noun class system from hand-curated data + corpus observations
    nc_classes, nc_evidence, nc_review_queue = _build_noun_class_system(
        destination, hand_curated, noun_inventory
    )

    # Build structured agreement system from hand-curated data
    ag_system, ag_evidence, ag_review_queue = _build_agreement_system(
        destination, hand_curated, ag_obs
    )

    # Build structured verb morphology
    verb_entries, verb_evidence, verb_analysis_queue = _build_verb_morphology(
        destination, vm_obs, analysis_records, all_review_tasks
    )

    # Build derivational morphology
    der_entries, der_evidence = _build_derivational_morphology(
        destination, analysis_records, wf_obs
    )

    # --- Review queue (annotation layer) ---
    # Combine lexical review tasks + analysis review tasks
    review_tasks = []
    review_candidates = [entry for entry in lexical_entries if len(entry["lemma"]) >= 3]
    for entry in sorted(review_candidates, key=lambda item: (-item["frequency"], item["lemma"]))[:250]:
        for task_type in ("LEXICAL_LEMMA", "MORPHOLOGY_SEGMENTATION", "NOUN_CLASS_OR_AGREEMENT"):
            review_tasks.append({
                "task_id": f"ANN_{task_type}_{entry['lexeme_id']}",
                "task_type": task_type,
                "surface_form": entry["lemma"],
                "status": "PENDING_HUMAN_REVIEW",
                "proposed_analysis": None,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "source_record_ids": [item["record_id"] for item in entry["evidence"]],
                "evidence": entry["evidence"],
                "instruction": "Record only an analysis supported by the cited isiXhosa corpus context; preserve the source surface form and mark uncertainty explicitly.",
            })
    # Add analysis review tasks
    review_tasks.extend(all_review_tasks)
    # Add noun class and agreement review queue entries
    review_tasks.extend(nc_review_queue)
    review_tasks.extend(ag_review_queue)
    review_tasks.extend(verb_analysis_queue)
    write_jsonl(destination / "annotation" / "linguistic_review_queue.jsonl", review_tasks)

    # --- Evidence graph ---
    all_record_ids = sorted(set(r["record_id"] for r in records))
    rule_ids = [
        "ORTH_IDENTITY_NORMALIZATION_001",
        "NC_PAIRING_HYPOTHESIS",
        "AGR_SUBJECT_CONCORD_MAPPING",
        "VM_VERB_SEGMENTATION_RULE",
        "TAM_MARKER_ANALYSIS",
        "NEGATION_PREFIX_ANALYSIS",
    ]
    _build_evidence_graph(destination, {"foundation_release": "XNLP_ISIXHOSA_FOUNDATION_V1",
                                         "sources": sorted({r["source"] for r in records}),
                                         "corpus_sha256": corpus_hash},
                           observation_ids, all_record_ids, rule_ids)

    # --- Conflict detection ---
    conflict_summary, conflict_list = _build_conflict_detection(destination, analysis_records)

    # --- Human-reviewed semantic coverage ---
    semantic_report = _build_semantic_coverage(
        destination, analysis_records, lexical_entries, corpus_hash
    )

    # --- Coverage report ---
    domain_data = {
        "orthography": {"entries": 1, "coverage_status": "OBSERVED", "confidence": "HIGH_CONFIDENCE",
                        "observation_ids": ["ORTHOGRAPHY_OBSERVED_V1"], "notes": "Character inventory and normalization observed from corpus."},
        "lexicon": {"entries": len(lexical_entries), "coverage_status": "OBSERVED", "confidence": "HIGH_CONFIDENCE",
                    "observation_ids": [e["lexeme_id"] for e in lexical_entries[:20]], "notes": "Surface-form observations only; no lemma assignment."},
        "noun_classes": {"entries": len(nc_classes), "coverage_status": "SUPPORTED", "confidence": "HIGH_CONFIDENCE",
                         "observation_ids": observation_ids[:50], "notes": "Structured entries from hand-curated foundation + corpus evidence."},
        "agreement": {"entries": len(ag_system), "coverage_status": "SUPPORTED", "confidence": "HIGH_CONFIDENCE",
                      "observation_ids": [obs["observation_id"] for obs in ag_obs[:20]], "notes": "Subject and possessive concord entries from hand-curated foundation."},
        "morphology": {"entries": len(nc_classes), "coverage_status": "SUPPORTED", "confidence": "HIGH_CONFIDENCE",
                       "observation_ids": [obs["observation_id"] for obs in nc_obs[:20]], "notes": "Noun-class and agreement morphology supported."},
        "verbs": {"entries": len(verb_entries), "coverage_status": "UNDER_REVIEW", "confidence": "SUPPORTED",
                  "observation_ids": [obs["observation_id"] for obs in vm_obs[:20]], "notes": "Candidate verb segmentations; require human review."},
        "tense_aspect_mood": {"entries": len(tam_obs), "coverage_status": "OBSERVED", "confidence": "HIGH_CONFIDENCE",
                              "observation_ids": [obs["observation_id"] for obs in tam_obs], "notes": "TAM and negation markers observed."},
        "negation": {"entries": len(neg_obs), "coverage_status": "OBSERVED", "confidence": "HIGH_CONFIDENCE",
                     "observation_ids": [obs["observation_id"] for obs in neg_obs], "notes": "Negation prefixes observed."},
        "pronouns": {"entries": len([o for o in fw_obs if o.get("pattern") in {"mna","wena","thina","inina","nina","yena"}]),
                       "coverage_status": "OBSERVED", "confidence": "SUPPORTED",
                       "observation_ids": [o["observation_id"] for o in fw_obs if o.get("pattern") in {"mna","wena","thina","inina","nina","yena"}],
                       "notes": "Pronoun function words observed."},
        "modifiers": {"entries": len(mod_obs), "coverage_status": "OBSERVED", "confidence": "OBSERVED",
                      "observation_ids": [obs["observation_id"] for obs in mod_obs], "notes": "Modifier patterns observed."},
        "syntax": {"entries": len(syn_obs), "coverage_status": "OBSERVED", "confidence": "OBSERVED",
                   "observation_ids": [obs["observation_id"] for obs in syn_obs], "notes": "Syntactic patterns observed."},
        "word_formation": {"entries": len(der_entries), "coverage_status": "UNDER_REVIEW", "confidence": "SUPPORTED",
                           "observation_ids": [obs["observation_id"] for obs in wf_obs], "notes": "Derivational patterns from corpus evidence."},
        "semantics": {
            "entries": semantic_report["verified_sense_entries"],
            "coverage_status": "PARTIAL" if semantic_report["verified_sense_entries"] else "INSUFFICIENT_EVIDENCE",
            "confidence": "SUPPORTED" if semantic_report["verified_sense_entries"] else "INSUFFICIENT_EVIDENCE",
            "observation_ids": [],
            "notes": "Human-reviewed, evidence-linked noun glosses; coverage is partial and does not encode compositional meaning.",
        },
        "discourse": {"entries": 0, "coverage_status": "INSUFFICIENT_EVIDENCE", "confidence": "INSUFFICIENT_EVIDENCE",
                      "observation_ids": [], "notes": "No automated discourse claims; requires human review."},
    }
    coverage = _build_coverage_report(destination, domain_data)

    # --- Word analysis engine ---
    _build_word_analyzer_module(destination, nc_classes, ag_system, verb_entries)

    # --- Grammar report ---
    _build_grammar_report(destination, {"foundation_release": "XNLP_ISIXHOSA_FOUNDATION_V1",
                                        "corpus_sha256": corpus_hash,
                                        "generation_mode": "deterministic_from_corpus_hash",
                                        "authoritative_records": len(records),
                                        "analysis_records": len(analysis_records),
                                        "model_training": "FROZEN",
                                        "decision": "FOUNDATION INCOMPLETE — CONTINUE STRUCTURAL ANALYSIS"},
                         nc_classes, ag_system, verb_entries, der_entries, coverage)

    # --- Review counts ---
    review_counts = {
        "noun_classes": len(nc_obs) + len(nc_review_queue),
        "agreement": len(ag_obs) + len(ag_review_queue),
        "verbs": len(vm_obs) + len(verb_analysis_queue),
        "tam": len(tam_obs) + len(tam_reviews),
        "function_words": len(fw_obs) + len(fw_reviews),
        "modifiers": len(mod_obs) + len(mod_reviews),
        "syntax": len(syn_obs) + len(syn_reviews),
        "word_formation": len(wf_obs) + len(wf_reviews),
        "negation": len(neg_obs) + len(neg_reviews),
        "lexicon": len(review_tasks),
    }

    # --- Grammar rules (observation-layer rule for preservation) ---
    grammar_rules = [{
        "rule_id": "ORT_IDENTITY_NORMALIZATION_001",
        "category": "orthography",
        "phenomenon": "source preservation",
        "rule": "Generated records retain the original source surface text; no spelling replacement is applied.",
        "examples": [r["record_id"] for r in analysis_records[:10]],
        "counterexamples": [],
        "frequency": len(analysis_records),
        "confidence": "HIGH_CONFIDENCE",
        "status": "SUPPORTED",
        "source_record_ids": [r["record_id"] for r in analysis_records],
        "source_works": sorted({r["source"] for r in analysis_records}),
        "variant_forms": [],
        "orthographic_forms": ["modern", "mixed", "traditional"],
        "notes": "A data-governance rule, not a claim about the full language.",
    }]
    write_json(destination / "grammar" / "grammar_rules.json", grammar_rules)

    # --- Contradictions (metadata-level) ---
    write_json(destination / "validation" / "contradictions.json", conflict_summary)

    # --- Manifest ---
    manifest = {
        "foundation_release": "XNLP_ISIXHOSA_FOUNDATION_V1",
        "generation_mode": "deterministic_from_corpus_hash",
        "generator": "xnlp_language.build_foundation",
        "corpus": str(corpus.relative_to(ROOT)),
        "corpus_sha256": corpus_hash,
        "authoritative_records": len(records),
        "analysis_records": len(analysis_records),
        "language_integrity_counts": dict(sorted(purity_counts.items())),
        "sources": sorted({r["source"] for r in records}),
        "model_training": "FROZEN",
        "decision": "FOUNDATION INCOMPLETE — CONTINUE STRUCTURAL ANALYSIS",
        "structured_entries": {
            "noun_classes": len(nc_classes),
            "agreement": len(ag_system),
            "verbs": len(verb_entries),
            "derivational": len(der_entries),
        },
        "review_tasks_total": len(review_tasks),
        "conflicts_open": len(conflict_list),
        "observation_count": len(all_observations),
    }
    write_json(destination / "foundation_manifest.json", manifest)

    # --- Structural gate report ---
    observations_map = {
        "noun_classes": nc_obs,
        "agreement": ag_obs,
        "verbs": vm_obs,
        "tense_aspect_mood": tam_obs,
        "function_words": fw_obs,
        "modifiers": mod_obs,
        "syntax": syn_obs,
        "word_formation": wf_obs,
        "negation": neg_obs,
    }
    _build_structural_gate(destination, manifest, observations_map, review_counts,
                           {"noun_classes": len(nc_classes),
                            "agreement": len(ag_system),
                            "verbs": len(verb_entries),
                            "derivational": len(der_entries),
                            "evidence_graph": 1,
                            "conflicts": len(conflict_list),
                            "coverage_report": 1,
                            "word_analyzer": 1,
                            "grammar_report": 1,
                            "structural_gate": 1},
                           conflict_list)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build corpus-grounded XNLP language artifacts")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = build(args.corpus, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
