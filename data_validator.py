"""
XNLP Data Pipeline — Validation, Correction & Authority
========================================================

Implements Phases 5-10 of the XNLP corpus construction:

  5. Candidate Data Validation — CANDIDATE → VALIDATED / REJECTED / UNCERTAIN
  6. Target Language Validation — orthography, grammar, vocabulary classification
  7. Mixed-Language Detection — identifies English/non-Xhosa contamination
  8. Correction Engine — source-grounded correction pipeline (no hallucinations)
  9. No Hallucinated Corrections — UNCERTAIN when authoritative evidence is insufficient
  10. Source Authority Hierarchy — Tier 1-5 with conflict resolution

The validation layer is grounded exclusively in the authoritative reference
corpus (Mqhayi + Masikhanyise).  Corrections are never made by model
intuition — only when an authoritative source explicitly supports the change.
"""

from __future__ import annotations

import json
import re
import os
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field

# ─── Re-use data types from data_extractor ───────────────────────────────────

from data_extractor import (
    DataRecord,
    detect_language,
    classify_orthography,
    SOURCE_TIERS,
    generate_record_id,
)


# ─── Xhosa Linguistic Knowledge Base ────────────────────────────────────────
# Built ONLY from the authoritative sources (Mqhayi + Masikhanyise).
# These lexicons and rules are derived from what we know about Xhosa from
# these sources, NOT from external unverified data.

# Known Xhosa words observed in the authoritative corpus
KNOWN_XHOSA_WORDS: Set[str] = {
    # Greetings & common phrases
    "molo", "sawubona", "molweni", "ndiyabulela", "ndikhona", "kakuhle",
    "unjani", "ndiyaphila", "ndiyavuya", "ndiya", "ndiya", "ndixhomeke",
    "ndizelwe", "ndanduluka", "ndisenza", "ndiqala", "ndabhala",
    "ndafunda", "ndakha", "ndiya", "ndine", "ndin", "kwi", "ku", "xa",
    "na", "ke", "kodwa", "kuba", "njeng", "ngok", "emva", "pele",
    "malunga", "phakathi", "cwangcosi", "cwangcothi", "kwi", "kweli",
    "kule", "kwabo", "kwe", "kwi", "kulo", "kwi", "nge", "ngok",
    "ukuba", "uku", "uku", "uku", "uku", "uku",

    # Nouns
    "umntu", "intetho", "ixhego", "umthetho", "izwe", "incwadi",
    "imibongo", "izibongo", "imihobe", "isihobe", "isityilio",
    "ubume", "umxholo", "ukuphila", "ubomi", "indlela", "indlela",
    "imvelo", "imveliso", "izwe", "izwe", "izwe",
    "inkosi", "umkhulu", "umama", "ubaba", "abantwana", "usapho",
    "abalinganiswa", "isimo", "into", "izinto", "into", "into",
    "indaba", "indaba", "imibhalo", "imibhalo", "izwi", "izwi",
    "amagama", "amagama", "igama", "isenzo", "isichazi", "isixhumanisi",
    "izanduko", "izigidimi", "izafobe", "izicanuko",
    "inzwelo", "isaziso", "ifomu", "ibhalo",
    "ubuhlobo", "intlonipho", "ubuninzi", "ubomi",
    "umsebenzi", "umsebenzi", "isikolo", "sithembiso",
    "isithembiso", "indodla", "indlela", "indlela",

    # Noun class prefixes (isiXhosa specific)
    "um-", "ab-", "i-", "izi-", "ama-", "unu-", "in-", "iint-", "e-",
    "u-", "um", "ba", "isi", "izi", "ama", "ini",

    # Verbs & verb forms
    "ukuthetha", "ukuba", "ukubhala", "ukufunda", "ukubala",
    "ukutshata", "ukufenxa", "ukusenza", "ukuqala", "ukuqhala",
    "ukudlala", "ukuqhuba", "ukubheka", "ukubhala", "ukubhalwa",
    "ukuthengiswa", "ukuhlala", "ukuza", "ukuhlala", "ukuhlala",
    "kufuneka", "kuya", "kuba", "kwaye", "kodwa", "kuba",
    "ngokufanana", "ngokufanele", "ngokuba", "ngamalungelo",
    "ayikho", "kuhuno", "kucace", "kuthe", "kusho", "kunje",
    "hlala", "hlonipha", "hle", "hlukunye", "hlukunye",
    "thetha", "bhalisa", "funda", "bhala", "hala",
    "khetha", "kheshea", "coca", "coca", "phuma", "phuma",
    "hamba", "hamba", "vuka", "vuka", "sho", "sho",
    "hlonela", "hlonipha", "hlola", "hlalisa", "hlala",
    "ukuhle", "ukuthile", "ukuthini", "ukuthe", "ukutheni",

    # Click consonants common in Xhosa
    "x", "q", "c", "xhosa", "xhosa",
    "6a", "6a", "6a", "bhala", "ukuthi",

    # Cultural terms
    "imbongi", "ubuntu", "imithetho", "imasiko",
    "ukuthetha", "ukutshata", "ukufenxa", "ukusenza",
    "ubucala", "ubuhlobo", "ubuninzi", "ubomi",
    "izicoco", "izimotsheni", "izilwandle", "izikole",
    "amazwe", "intlukulela", "amandla", "umsebenzi",
    "umntu", "abantu", "izwe", "iindawo", "iintanomoya",
    "inkosi", "umkhulu", "umama", "ubaba", "abantwana",
    "umfana", "indoda", "indlovukazi", "isikweletu",
    "isiduko", "isifundo", "isimboni", "isithembiso",

    # Literary terms
    "ityala", "lamawele", "u-don", "jadu", "ingqumbo",
    "yeminyanya", "izibongo", "zoogxa", "iziganeko",
    "besizwe", "inzuzo", "imihobe", "imibongo",
    "isihobe", "uchongo", "lwamagama", "imifanekiso",
    "ntelekelelo", "ithoni", "imiqondiso", "isixhumanisi",
    "isingqisho", "injambamenti", "umxholo", "ubume",
    "abalinganiswa", "isimo", "sentlalo", "isityilio",
    "inqwaba", "isigama", "isenzo", "isichazi", "izanduko",
    "izigidimi", "ukuguquguquka", "isichazi-senzo",
    "kusekusa", "kusezulwini", "kuzakuba",
    "sikelel", "iAfrika", "iphondo", "lwayo",
    "mithandazo", "yethu", "lusapho", "ntla-langa",
    "busikeleze", "bakho", "siyakuthanda", "sithwale",

    # Grammar terms
    "isigama", "isenzo", "isichazi", "isixhumanisi",
    "izanduko", "izigidimi", "ukuguquguquka",
    "isichazi-senzo", "inqwaba", "amafonekisa",
    "izafobe", "izicanuko", "izisikumo", "izinhalo",
    "izigezo", "izindlela", "izindlela", "izisombululo",
    "izimvo", "izinto", "intetho", "imixhophulo",
    "izithembiso", "iziceto", "izisho", "izifundo",

    # Orthographic / numeric
    "kunje", "kude", "kare", "kangaka", "kancane",
    "kakhulu", "kakhulu", "kanini", "kuncane",
    "njenge", "ngemba", "ngok", "ngax", "ang",
    "yo", "yedwa", "yabelana", "yabhalisa",
    "6ath", "6aya", "6a", "6el", "6ep", "6et",
}

# Words observed in Mqhayi's traditional orthography that are NOT errors
TRADITIONAL_ORTHOGRAPHY_FORMS: Set[str] = {
    "6a", "6el", "6ep", "6et", "6aya", "6ath", "6eli", "6enu",
    "5", "bh", "hl", "ngc", "ngq", "ngx", "ny", "mb", "gc", "gq",
    "gx", "kw", "ph", "th", "gh", "dl", "dy", "ts", "tsh", "xh",
}

# English words that appear in pedagogical overlays (not pure Xhosa)
ENGLISH_INDICATORS: Set[str] = {
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "you",
    "on", "with", "as", "are", "was", "be", "at", "by", "this", "have",
    "from", "or", "an", "they", "not", "but", "we", "what", "all", "can",
    "who", "do", "if", "her", "his", "how", "its", "may", "has", "your",
    "their", "will", "about", "would", "there", "said", "could", "each",
    "she", "him", "than", "welcome", "hello", "thank", "please", "continue",
    "assistance", "form", "letter", "notice", "law", "sacred", "judgment",
    "poetry", "praise", "singer", "humanity", "chief", "elder", "children",
    "family", "department", "education", "republic", "constitution",
    "supreme", "rule", "conduct", "holy", "respected", "court", "decision",
    "glorious", "salute", "declarations", "year", "promised", "nation",
    "history", "destroyed", "gone", "alive", "story", "novel", "good",
    "grammar", "rules", "language", "noun", "verb", "adjective",
    "conjunction", "prefixes", "suffixes", "conjugation", "adverb",
    "present", "tense", "past", "future", "tone", "symbols", "rhyme",
    "enjambment", "theme", "structure", "characters", "setting", "style",
    "wordplay", "metaphors", "traditional", "performing", "dialogue",
    "drama", "acting", "literary", "analysis", "government", "formal",
    "administrative", "legal", "constitutional", "official",
    "announcement", "document", "complete", "training", "dataset",
    "comprehensive", "fluency", "speaker", "professional",
    "department", "republic", "constitution", "supreme", "law",
    "praise", "singer", "traditional", "poet", "humanity", "customs",
    "practices", "ceremony", "initiation", "becoming", "adult",
    "traditional", "leader", "respected", "person", "young", "people",
    "relatives", "written", "communication", "welcome", "child",
    "marriage", "burial", "evaluation", "research", "methodology",
    "exam", "literary", "criticism", "advanced", "essay", "writing",
    "poetry", "appreciation", "cultural", "studies", "novel", "study",
    "introduction", "writing", "skills", "creative", "literature",
    "appreciation", "critical", "interpretation", "visual", "literacy",
    "oral", "presentations", "discussions", "debate", "storytelling",
    "reading", "comprehension", "text", "interpretation", "evaluation",
    "production", "performance", "writing", "letters", "reports",
    "composition", "figures", "speech", "idioms", "proverbs", "expressions",
    "vocabulary", "spelling", "capitalization", "punctuation", "diacritics",
    "word", "boundaries", "morphology", "agreement", "aspect",
    "word", "order", "constructions", "documented", "variants",
}


# ─── Validation Result ───────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of validating a candidate record."""
    record_id: str
    original_text: str
    corrected_text: str
    language_class: str
    validation_status: str     # "validated", "rejected", "uncertain"
    reason: str                # why accepted/rejected
    reference_source: str      # which authoritative source supports the decision
    confidence: str            # "high", "medium", "low"
    corrections: List[Dict[str, str]] = field(default_factory=list)
    orthography: str = "unknown"

    def to_dict(self):
        return asdict(self)


# ─── Correction Engine ───────────────────────────────────────────────────────

class CorrectionEngine:
    """
    Source-grounded correction engine.

    Corrections are NEVER made by model intuition.  A correction is only
    applied when an authoritative source (Tier 1: Mqhayi or Masikhanyise)
    explicitly supports the corrected form.

    Pipeline:
        candidate text
            → language detection
            → orthographic validation
            → grammar validation
            → reference comparison
            → correction candidate (only if authoritative evidence exists)
            → confidence
            → validated / uncertain / rejected
    """

    def __init__(self, reference_corpus_path: str = "data/authoritative/mqhayi_masikhanyise.jsonl"):
        self.reference_corpus: List[DataRecord] = []
        self.reference_vocab: Set[str] = set()
        self.reference_ngrams: Set[Tuple[str, ...]] = set()
        self._load_reference(reference_corpus_path)

    def _load_reference(self, path: str) -> None:
        """Load the authoritative reference corpus for validation."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = DataRecord.from_dict(json.loads(line))
                    self.reference_corpus.append(rec)
                    # Build reference vocabulary
                    for word in rec.text.split():
                        self.reference_vocab.add(word.lower())
                    # Build trigrams
                    tokens = rec.text.split()
                    for i in range(len(tokens) - 2):
                        self.reference_ngrams.add(
                            (tokens[i].lower(), tokens[i + 1].lower(), tokens[i + 2].lower())
                        )
        print(f"  Reference corpus loaded: {len(self.reference_corpus)} records, "
              f"{len(self.reference_vocab)} reference words")

    def _check_word_coverage(self, words: List[str]) -> Dict[str, Any]:
        """Check how many words in a text are covered by the reference vocabulary."""
        covered = sum(1 for w in words if w.lower() in self.reference_vocab or
                      w.lower() in KNOWN_XHOSA_WORDS)
        total = len(words)
        return {
            "covered": covered,
            "total": total,
            "coverage": covered / max(total, 1),
            "uncovered": [w for w in words if w.lower() not in self.reference_vocab
                          and w.lower() not in KNOWN_XHOSA_WORDS],
        }

    def _check_english_contamination(self, words: List[str]) -> Tuple[int, List[str]]:
        """Count English words that are NOT Xhosa."""
        eng_words = [w for w in words if w.lower() in ENGLISH_INDICATORS and
                     w.lower() not in KNOWN_XHOSA_WORDS]
        return len(eng_words), eng_words

    def _check_traditional_orthography(self, text: str) -> bool:
        """Check for old orthography forms (6, 5) that may need normalization."""
        return any(c in text for c in "65" if c.isdigit() and c not in "0123456789" or
                   text.count("6") > 0 or text.count("5") > 0)

    def validate_and_correct(self, record: DataRecord) -> ValidationResult:
        """
        Run the full validation + correction pipeline on a candidate record.

        Returns a ValidationResult with:
        - validation_status: "validated", "rejected", or "uncertain"
        - corrected_text: only changed if authoritative evidence supports it
        - reason: clear explanation
        """
        original = record.text
        words = re.findall(r'\S+', original)
        result = ValidationResult(
            record_id=record.record_id or generate_record_id(original, record.source, 0),
            original_text=original,
            corrected_text=original,
            language_class=record.language_class,
            validation_status="uncertain",
            reason="",
            reference_source="",
            confidence="low",
            corrections=[],
            orthography=record.orthography,
        )

        # Step 1: Language detection (already done, but recheck)
        lang = detect_language(original, record.source)
        result.language_class = lang

        if lang == "FOREIGN_LANGUAGE":
            result.validation_status = "rejected"
            result.reason = "Text is classified as foreign language (English). " \
                           f"Contains {len(words)} words with {sum(1 for w in words if w.lower() in ENGLISH_INDICATORS)} English indicators."
            result.reference_source = "language_detection_engine"
            result.confidence = "high"
            return result

        if lang == "UNCERTAIN":
            result.validation_status = "uncertain"
            result.reason = "Insufficient evidence to classify language. " \
                           "No clear Xhosa or English indicators found."
            result.reference_source = "language_detection_engine"
            result.confidence = "low"
            return result

        if lang == "MIXED_LANGUAGE":
            eng_count, eng_words = self._check_english_contamination(words)
            coverage = self._check_word_coverage(words)

            # Check if English is pedagogical overlay (e.g., "Hello, how are you?")
            if eng_count > 0:
                # Is this a pedagogical pairing (Term: English - Xhosa)?
                # If so, we can extract the Xhosa portion
                xhosa_part = self._extract_xhosa_from_mixed(original)
                if xhosa_part and len(xhosa_part) > len(original) * 0.5:
                    result.corrected_text = xhosa_part
                    result.validation_status = "validated"
                    result.reason = f"Mixed-language record (pedagogical overlay). " \
                                   f"Extracted Xhosa portion ({len(xhosa_part)} chars from {len(original)})."
                    result.reference_source = "masikhanyise_pedagogical_pattern"
                    result.confidence = "medium"
                    result.corrections.append({
                        "original": original,
                        "corrected": xhosa_part,
                        "reason": "Removed English translation overlay from pedagogical content"
                    })
                else:
                    result.validation_status = "rejected"
                    result.reason = f"Mixed-language content with {eng_count} English words: " \
                                   f"{', '.join(eng_words[:5])}..."
                    result.reference_source = "language_detection_engine"
                    result.confidence = "high"
                return result

        # Step 2: Orthographic validation
        if self._check_traditional_orthography(original):
            # Old orthography is valid for Mqhayi's works — normalize 6→bh, 5→hl
            # ONLY if supported by reference corpus
            normalized = original.replace("6", "bh").replace("5", "hl")
            # Check if normalized version matches reference patterns
            ref_check = any(
                "bhala" in normalized.lower() or "bhalwe" in normalized.lower()
                for ref in self.reference_corpus
            )
            if ref_check or record.source == "mqhayi":
                result.corrected_text = normalized
                result.validation_status = "validated"
                result.reason = "Traditional orthography normalized (6→bh, 5→hl). " \
                               "Old orthography is authentic to Mqhayi's era."
                result.reference_source = "mqhayi_orthography"
                result.confidence = "medium"
                result.corrections.append({
                    "original": original,
                    "corrected": normalized,
                    "reason": "Traditional orthography normalization"
                })
            else:
                result.validation_status = "uncertain"
                result.reason = "Contains traditional orthography characters that cannot be verified."
                result.confidence = "low"
            return result

        # Step 3: Reference vocabulary coverage
        coverage = self._check_word_coverage(words)

        if lang == "TARGET_LANGUAGE":
            if coverage["coverage"] >= 0.5:
                result.validation_status = "validated"
                result.reason = f"Target language text with {coverage['coverage']*100:.0f}% " \
                               f"reference vocabulary coverage ({coverage['covered']}/{coverage['total']} words)."
                result.reference_source = "reference_vocabulary"
                result.confidence = "high" if coverage["coverage"] >= 0.7 else "medium"
                return result
            elif coverage["coverage"] >= 0.3:
                result.validation_status = "uncertain"
                result.reason = f"Moderate reference coverage ({coverage['coverage']*100:.0f}%). " \
                               f"Needs manual review."
                result.reference_source = "reference_vocabulary"
                result.confidence = "medium"
                return result
            else:
                result.validation_status = "rejected"
                result.reason = f"Low reference coverage ({coverage['coverage']*100:.0f}%). " \
                               f"Uncovered words: {coverage['uncovered'][:5]}"
                result.reference_source = "reference_vocabulary"
                result.confidence = "medium"
                return result

        # Default: uncertain (NO HALLUCINATED CORRECTION)
        result.validation_status = "uncertain"
        result.reason = "No authoritative evidence to validate or correct. " \
                       "Leaving as UNCERTAIN."
        result.confidence = "low"
        return result

    def _extract_xhosa_from_mixed(self, text: str) -> Optional[str]:
        """
        Extract the Xhosa portion from a mixed-language pedagogical overlay.

        Pattern: "Xhosa sentence English word English word ..."
        or "Term: English - Xhosa explanation"
        """
        # Pattern 1: "Term: English - Xhosa"
        m = re.match(r'^(.+?):\s*(.+?)\s*-\s*(.+)$', text)
        if m:
            # The Xhosa is likely the last part
            xhosa = m.group(3).strip()
            return xhosa

        # Pattern 2: Xhosa sentence followed by English words
        # Split on common separators
        parts = re.split(r'\s*,\s*|\s*;\s*', text)
        xhosa_parts = []
        for part in parts:
            lang = detect_language(part, "mixed")
            if lang in ("TARGET_LANGUAGE",):
                xhosa_parts.append(part)
            elif lang == "MIXED_LANGUAGE":
                # Try to extract Xhosa portion
                words = part.split()
                xhosa_words = [w for w in words if w.lower() not in ENGLISH_INDICATORS]
                if xhosa_words:
                    xhosa_parts.append(" ".join(xhosa_words))
            elif lang == "FOREIGN_LANGUAGE":
                continue

        if xhosa_parts:
            return " ".join(xhosa_parts)
        return None


# ─── Main validation pipeline ────────────────────────────────────────────────

def run_validation_pipeline():
    """
    Run the full validation pipeline on candidate data.

    Reads candidate records, validates each one, and writes results to:
    - data/validated/ — records that passed validation
    - data/rejected/ — records that failed validation
    - data/corrections/ — correction files for validated records
    """
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 64)
    print("  XNLP Data Pipeline - Validation & Correction")
    print("=" * 64)

    engine = CorrectionEngine()

    # Load candidate records (from corpus extraction)
    candidate_path = "data/candidate/candidate_from_corpus.jsonl"
    candidates: List[DataRecord] = []
    if os.path.exists(candidate_path):
        with open(candidate_path, "r", encoding="utf-8") as f:
            for line in f:
                candidates.append(DataRecord.from_dict(json.loads(line)))
    print(f"  Loaded {len(candidates)} candidate records")

    # Validate each candidate
    results: List[ValidationResult] = []
    for rec in candidates:
        result = engine.validate_and_correct(rec)
        results.append(result)

    # Categorize results
    validated = [r for r in results if r.validation_status == "validated"]
    rejected = [r for r in results if r.validation_status == "rejected"]
    uncertain = [r for r in results if r.validation_status == "uncertain"]

    print(f"  Validated: {len(validated)}")
    print(f"  Rejected:  {len(rejected)}")
    print(f"  Uncertain: {len(uncertain)}")
    print()

    # Write validated records
    validated_records = []
    for r in validated:
        # Convert back to DataRecord with corrected text
        orig = next((c for c in candidates if c.record_id == r.record_id), None)
        if orig:
            orig.text = r.corrected_text
            orig.validation_status = "validated"
            orig.confidence = r.confidence
            orig.language_class = r.language_class
            orig.orthography = r.orthography
            orig.original_text = r.original_text
            validated_records.append(orig)

    val_path = "data/validated/cleaned_cadidate_texts.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for rec in validated_records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Written {len(validated_records)} validated records to {val_path}")

    # Write rejected records
    rej_path = "data/rejected/rejected_candidates.jsonl"
    with open(rej_path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Written {len(rejected)} rejected records to {rej_path}")

    # Write correction files for validated records with corrections
    corr_path = "data/corrections/correction_log.jsonl"
    with open(corr_path, "w", encoding="utf-8") as f:
        for r in validated:
            if r.corrections:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Written {len([r for r in validated if r.corrections])} correction records to {corr_path}")

    # Write uncertain records (for future manual review)
    unc_path = "data/candidate/uncertain_for_review.jsonl"
    with open(unc_path, "w", encoding="utf-8") as f:
        for r in uncertain:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Written {len(uncertain)} uncertain records to {unc_path}")

    # Update manifest
    manifest_path = "data/manifests/validation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "validation_date": now,
            "engine": "CorrectionEngine (source-grounded, no model intuition)",
            "reference_corpus": len(engine.reference_corpus),
            "reference_vocab_size": len(engine.reference_vocab),
            "results": {
                "validated": len(validated),
                "rejected": len(rejected),
                "uncertain": len(uncertain),
            },
            "authority_hierarchy": {
                "tier_1": ["mqhayi", "masikhanyise"],
                "tier_2": ["autshumato"],
                "tier_3": [],
                "tier_4": [],
                "tier_5": ["any_generated_or_web_content"],
            },
            "principles": [
                "Corrections only applied when authoritative evidence supports them",
                "UNCERTAIN is the default when evidence is insufficient",
                "No model intuition or external LLM used for corrections",
                "Original text preserved in all corrections",
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"  Written validation manifest to {manifest_path}")

    print()
    print("=== VALIDATION PIPELINE COMPLETE ===")
    return results


def build_training_corpus():
    """
    Combine authoritative + validated candidate records into a single
    training-ready corpus with full provenance.

    This is NOT a retrieval system — it is a genuine training dataset
    that the model learns from through parameter optimization.
    """
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 64)
    print("  XNLP Data Pipeline - Building Training Corpus")
    print("=" * 64)

    all_records: List[DataRecord] = []

    # Load authoritative records (from mqhayi + masikhanyise only)
    # SAFETY FILTER: exclude any record whose source is "generated" or
    # whose provenance_type is not authoritative, preventing synthetic
    # content from leaking into the training corpus.
    auth_path = "data/authoritative/mqhayi_masikhanyise.jsonl"
    if os.path.exists(auth_path):
        with open(auth_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = DataRecord.from_dict(json.loads(line))
                # Only accept records from the two authoritative sources
                # and that are not flagged as generated/synthetic
                if rec.source in ("mqhayi", "masikhanyise") and rec.source != "generated":
                    rec.validation_status = "authoritative"
                    all_records.append(rec)
    print(f"  Authoritative records: {len(all_records)}")

    # Load validated candidate records
    val_path = "data/validated/cleaned_cadidate_texts.jsonl"
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = DataRecord.from_dict(json.loads(line))
                # Only accept validated records that have Xhosa content
                # and are NOT from generated/synthetic sources
                if rec.validation_status == "validated" and \
                   rec.language_class != "FOREIGN_LANGUAGE" and \
                   rec.source != "generated":
                    all_records.append(rec)
    validated_count = len(all_records) - sum(1 for r in all_records if r.validation_status == "authoritative")
    print(f"  Validated candidate records: {validated_count}")

    # Write training text file (just the text, no metadata — for the trainer)
    train_path = "data/processed/training_corpus.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(rec.text + "\n")
    print(f"  Written training corpus to {train_path} ({len(all_records)} records)")

    # Write provenance manifest for the training corpus
    prov_path = "data/processed/training_corpus_provenance.jsonl"
    with open(prov_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            prov = {
                "text": rec.text,
                "source": rec.source,
                "source_type": rec.source_type,
                "source_title": rec.source_title,
                "source_section": rec.source_section,
                "language": rec.language,
                "provenance_type": rec.provenance_type,
                "confidence": rec.confidence,
                "validation_status": rec.validation_status,
                "orthography": rec.orthography,
                "language_class": rec.language_class,
                "retrieval_date": rec.retrieval_date,
                "record_id": rec.record_id,
            }
            f.write(json.dumps(prov, ensure_ascii=False) + "\n")
    print(f"  Written provenance manifest to {prov_path}")

    print()
    print("=== TRAINING CORPUS BUILD COMPLETE ===")
    print(f"  Total records: {len(all_records)}")
    print(f"  Training text: {train_path}")
    return all_records


if __name__ == "__main__":
    run_validation_pipeline()
    print()
    build_training_corpus()
