"""
XNLP Data Pipeline — Corpus Deep Extraction
=============================================

Processes the original corpus files to extract ALL authentic isiXhosa
text from the two specified authoritative sources (Mqhayi + Masikhanyise),
stripping English pedagogical overlays and classifying every sentence
with full provenance.

This is NOT retrieval — it is corpus construction for language model training.
The extracted text becomes training data that the neural model learns from
through parameter optimization.
"""

from __future__ import annotations

import json
import re
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, asdict

from data_extractor import DataRecord, detect_language, generate_record_id, SOURCE_TIERS, classify_orthography


# ─── Corpus Section Classifiers ──────────────────────────────────────────────

# Each pattern identifies which work/section a line belongs to
MQHAYI_SECTIONS = [
    # (pattern, work_title, section_name, source_type)
    (r'^[Uu]\s*-?\s*[Dd]on\s*[Jj]adu', "U-Don Jadu", "Novel", "novel_excerpt"),
    (r'[Uu]-[Dd]on\s*[Jj]adu|UDon\s*Jadu', "U-Don Jadu", "Novel", "novel_excerpt"),
    (r'Ityala\s*Lamawele|ityala\s*lamawele', "Ityala Lamawele", "Novel", "novel_excerpt"),
    (r'[Uu]Samson|USamson|u-Samson', "U-Samson", "Novel", "novel_excerpt"),
    (r'[Ii]zibongo\s*zoogxa|izibongo\s*zoogxa', "Izibongo zoogxa", "Praise Poems", "praise_poetry"),
    (r'[Ii]ziganeko\s*zesizwe|iziganeko\s*zesizwe', "Iziganeko zesizwe", "Occasional Poems", "poetry"),
    (r'[Ii]nzuzo|Inzuzo', "Inzuzo", "Poetry Collection", "poetry"),
    (r'[Ii]mhobe|imihobe', "Imihobe nemibongo", "Songs/Lullabies", "song"),
    (r'[Aa]bantu\s*besizwe|Abantu\s*besizwe', "Abantu besizwe", "Biographical Essays", "essay"),
    (r'[Uu]bombo\s*Mfundisi|Ubombo\s*Mfundisi', "Ubombo Mfundisi uJohn Knox Bokwe", "Biography", "biography"),
    (r'[Uu]Mqhayi\s*waseNtab|UMqhayi\s*wase', "UMqhayi waseNtab'ozuko", "Autobiography", "autobiography"),
    (r'[Nn]kosi\s*Sikelel|Nkosi\s*Sikelel', "Nkosi Sikelel' iAfrika", "National Anthem", "traditional_song"),
    (r'[Ii]mbongi|Imbongi', None, "Cultural Term", "cultural_content"),
    (r'[Uu]buntu|ubuntu', None, "Cultural Term", "cultural_content"),
    (r'[Ii]mithetho|imithetho', None, "Cultural Term", "cultural_content"),
    (r'[Ii]masiko|imasiko', None, "Cultural Term", "cultural_content"),
    (r'[Ii]nkosi|inkosi', None, "Cultural Term", "cultural_content"),
    (r'[Uu]mkhulu|umkhulu', None, "Cultural Term", "cultural_content"),
    (r'[Uu]sapho|usapho', None, "Cultural Term", "cultural_content"),
    (r'[Ii]ntetho|intetho', None, "Cultural Term", "cultural_content"),
    (r'[Uu]ku[a-z]|ukuthetha|ukubhala|ukufunda', None, "Cultural Term", "cultural_content"),
    (r'[Uu]mntu|umntu', None, "Cultural Term", "cultural_content"),
    (r'[Aa]bantwana|abantwana', None, "Cultural Term", "cultural_content"),
    (r'[Ii]ncwadi|incwadi', None, "Literature Term", "literature"),
    (r'[Ii]mbhalo|imibhalo', None, "Literature Term", "literature"),
    (r'[Ii]zwi|izwi', None, "Literature Term", "literature"),
    (r'[Ii]zibongo|zibongo', None, "Literature Term", "literature"),
    (r'[Ii]mibongo|imibongo', None, "Literature Term", "literature"),
    (r'[Ii]mihobe|imihobe', None, "Literature Term", "literature"),
    (r'[Ii]sithembiso|isithembiso', None, "Literature Term", "literature"),
    (r'[Ii]ndaba|indaba', None, "Literature Term", "literature"),
    (r'[Ii]ntlawulo|intlawulo', None, "Literature Term", "literature"),
    (r'[Ii]ndlela|indlela', None, "Cultural Term", "cultural_content"),
    (r'[Ii]ndaba|indaba', None, "Cultural Term", "cultural_content"),
    (r'[Ii]mvelo|imvelo', None, "Cultural Term", "cultural_content"),
    (r'[Ii]mveliso|imveliso', None, "Cultural Term", "cultural_content"),
    (r'[Ii]lifa|ilifa', None, "Cultural Term", "cultural_content"),
    (r'[Ii]siko|isiko', None, "Cultural Term", "cultural_content"),
    (r'[Ii]ntsomi|intsomi', None, "Cultural Term", "cultural_content"),
    (r'[Ii]zandla|izandla', None, "Cultural Term", "cultural_content"),
    (r'[Ii]zithembiso|izithembiso', None, "Cultural Term", "cultural_content"),
    (r'[Ii]zikhala|izikhala', None, "Cultural Term", "cultural_content"),
    (r'[Ii]mbhali|imbhali', None, "Literature Term", "literature"),
    (r'[Ii]mbongi|imbongi', None, "Literature Term", "literature"),
    (r'[Ii]mbhoxo|imbhoxo', None, "Literature Term", "literature"),
    (r'[Ii]ndaba|indaba', None, "Literature Term", "literature"),
    (r'[Ii]ntombazane|intombazane', None, "Cultural Term", "cultural_content"),
    (r'[Aa]baku|abaku', None, "Cultural Term", "cultural_content"),
    (r'[Uu]kuthetha|ukuthetha', None, "Cultural Term", "cultural_content"),
    (r'[Uu]kutshata|ukutshata', None, "Cultural Term", "cultural_content"),
    (r'[Uu]kufenxa|ukufenxa', None, "Cultural Term", "cultural_content"),
    (r'[Uu]kusenza|ukusenza', None, "Cultural Term", "cultural_content"),
    (r'[Uu]kubhala|ukubhala', None, "Literature Term", "literature"),
    (r'[Uu]kufunda|ukufunda', None, "Literature Term", "literature"),
]

MASIKHANYISE_SECTIONS = [
    (r'[Ii]zibongo|izibongo', "Izibongo", "Poetry", "poetry_term"),
    (r'[Ii]mibongo|imibongo', "Imibongo", "Poetry", "poetry_term"),
    (r'[Ii]mihobe|imihobe', "Imihobe", "Poetry", "poetry_term"),
    (r'[Ii]sihobe|isihobe', "Isihobe", "Poetry", "poetry_term"),
    (r'[Uu]chongo|uchongo', "Uchongo lwamagama", "Poetic Device", "poetry_term"),
    (r'[Ii]mifanekiso|imifanekiso', "Imifanekiso ntelekelelo", "Poetic Device", "poetry_term"),
    (r'[Ii]thoni|ithoni', "Ithoni", "Poetry Element", "poetry_term"),
    (r'[Ii]miqondiso|imiqondiso', "Imiqondiso", "Poetry Element", "poetry_term"),
    (r'[Ii]singqisho|isingqisho', "Isingqisho", "Poetic Device", "poetry_term"),
    (r'[Ii]njambamenti|injambamenti', "Injambamenti", "Poetic Device", "poetry_term"),
    (r'[Uu]mxholo|umxholo', "Umxholo", "Analysis", "literature_analysis"),
    (r'[Uu]bume|ubume', "Ubume", "Analysis", "literature_analysis"),
    (r'[Aa]blinganiswa|abalinganiswa', "Abalinganiswa", "Analysis", "literature_analysis"),
    (r'[Ii]simo\s*sentlalo|isimo\s*sentlalo', "Isimo sentlalo", "Analysis", "literature_analysis"),
    (r'[Ii]sityilio|isityilio', "Isityilio", "Analysis", "literature_analysis"),
    (r'[Ii]nnqwa|innqwa', "Inqwaba", "Grammar", "grammar_term"),
    (r'[Ii]sigama|isigama', "Isigama", "Grammar", "grammar_term"),
    (r'[Ii]senzo|isenzo', "Isenzo", "Grammar", "grammar_term"),
    (r'[Ii]sichazi|isichazi', "Isichazi", "Grammar", "grammar_term"),
    (r'[Ii]sixhumanisi|isixhumanisi', "Isixhumanisi", "Grammar", "grammar_term"),
    (r'[Ii]zanduko|izanduko', "Izanduko", "Grammar", "grammar_term"),
    (r'[Ii]zigidimi|izigidimi', "Izigidimi", "Grammar", "grammar_term"),
    (r'[Uu]kuguquguquka|ukuguquguquka', "Ukuguquguquka", "Grammar", "grammar_term"),
    (r'[Ii]sichazi[- ]senzo|isichazi-senzo', "Isichazi-senzo", "Grammar", "grammar_term"),
    (r'[Xx]a\s*kusekusa|xa\s*kusekusa', "Xa kusekusa", "Grammar", "grammar_term"),
    (r'[Xx]a\s*kusezulwini|xa\s*kusezulwini', "Xa kusezulwini", "Grammar", "grammar_term"),
    (r'[Xx]a\s*kuzakuba|xa\s*kuzakuba', "Xa kuzakuba", "Grammar", "grammar_term"),
    (r'[Ii]mbongi|imbongi', "Imbongi", "Cultural Content", "cultural_content"),
    (r'[Uu]buntu|ubuntu', "Ubuntu", "Cultural Content", "cultural_content"),
    (r'[Ii]mithetho|imithetho', "Imithetho", "Cultural Content", "cultural_content"),
    (r'[Ii]masiko|imasiko', "Imasiko", "Cultural Content", "cultural_content"),
    (r'[Ii]nkosi|inkosi', "Inkosi", "Cultural Content", "cultural_content"),
    (r'[Uu]mkhulu|umkhulu', "Umkhulu", "Cultural Content", "cultural_content"),
    (r'[Aa]bantwana|abantwana', "Abantwana", "Cultural Content", "cultural_content"),
    (r'[Uu]sapho|usapho', "Usapho", "Cultural Content", "cultural_content"),
    (r'[Uu]kuthetha|ukuthetha', "Ukuthetha", "Cultural Content", "cultural_content"),
    (r'[Uu]kutshata|ukutshata', "Ukutshata", "Cultural Content", "cultural_content"),
    (r'[Uu]kufenxa|ukufenxa', "Ukufenxa", "Cultural Content", "cultural_content"),
    (r'[Uu]kusenza|ukusenza', "Ukusenza", "Cultural Content", "cultural_content"),
    (r'[Ii]dwale|idwale', "Idwala", "Cultural Content", "cultural_content"),
    (r'[Ii]ndaba|indaba', "Indaba", "Cultural Content", "cultural_content"),
    (r'[Ii]zwe|izwe', "Izwe", "Cultural Content", "cultural_content"),
    (r'[Ii]zwe\s*lakithi|izwe\s*lakithi', "Izwe lakithi", "Cultural Content", "cultural_content"),
    (r'[Ii]mkhosi|imikhosi', "Imikhosi", "Cultural Content", "cultural_content"),
    (r'[Ii]ntombazane|intombazane', "Intombazane", "Cultural Content", "cultural_content"),
    (r'[Ii]cici|icici', "Icici", "Cultural Content", "cultural_content"),
    (r'[Ii]sisu|isisu', "Isisu", "Cultural Content", "cultural_content"),
    (r'[Ii]ziko|iziko', "Iziko", "Cultural Content", "cultural_content"),
    (r'[Ii]zibuko|izibuko', "Izibuko", "Cultural Content", "cultural_content"),
    (r'[Ii]zithembiso|izithembiso', "Izithembiso", "Cultural Content", "cultural_content"),
    (r'[Ii]zikhala|izikhala', "Izikhala", "Cultural Content", "cultural_content"),
    (r'[Ii]zingqungquthela|izingqungquthela', "Izingqungquthela", "Cultural Content", "cultural_content"),
]


# ─── Xhosa Text Extraction Helpers ───────────────────────────────────────────

def strip_english_overlay(line: str) -> Tuple[str, str]:
    """
    Strip English translations from pedagogical overlay lines.

    Patterns handled:
    - "Xhosa word/sentence: English - Xhosa explanation"
    - "Xhosa sentence English word English word..."
    - "English word: Xhosa" (rare)

    Returns (xhosa_text, english_removed).
    """
    # Pattern: "Term: English word(s) - Xhosa explanation"
    # e.g., "Isigama: Noun - igama elichaza into"
    m = re.match(r'^(.+?):\s*(.+?)\s*-\s*(.+)$', line.strip())
    if m:
        term = m.group(1).strip()
        english = m.group(2).strip()
        xhosa = m.group(3).strip()
        # If xhosa part is actually English, swap
        eng_words = set(english.lower().split())
        xhosa_words = set(xhosa.lower().split())
        if not eng_words.intersection(ENGLISH_LEXICON) and xhosa_words.intersection(ENGLISH_LEXICON):
            # The "explanation" is actually English
            return xhosa, english
        return xhosa, f"{term}: {english}"

    # Pattern: "Term: English_word(s) Xhosa_text..."
    # e.g., "Imasiko: customs imithetho yendlela yokuphila."
    # The Xhosa term is before the colon, English words follow, mixed with Xhosa
    m2 = re.match(r'^(.+?):\s*(.+)$', line.strip())
    if m2:
        term = m2.group(1).strip()
        rest = m2.group(2).strip()
        # Separate English words from Xhosa in the rest
        rest_words = rest.split()
        xhosa_part_words = []
        english_part_words = []
        for w in rest_words:
            if w.lower() in ENGLISH_LEXICON or _is_english_word(w):
                english_part_words.append(w)
            else:
                xhosa_part_words.append(w)
        if xhosa_part_words and len(" ".join(xhosa_part_words)) > len(" ".join(english_part_words)):
            # Keep the term + Xhosa portion, strip English
            xhosa_text = term + " " + " ".join(xhosa_part_words)
            english_removed = " ".join(english_part_words)
            return xhosa_text, english_removed

    # Pattern: "Xhosa sentence English words..."
    # Split into words, separate English words
    words = line.split()
    if not words:
        return "", ""

    # Find the boundary between Xhosa and English
    xhosa_words = []
    english_words = []
    seen_xhosa = False
    seen_english = False

    for w in words:
        is_english = w.lower() in ENGLISH_LEXICON or _is_english_word(w)
        is_xhosa = _is_xhosa_word(w) or w.lower() in XHOSA_WORDS

        if is_english and not is_xhosa:
            english_words.append(w)
            seen_english = True
        elif is_xhosa:
            xhosa_words.append(w)
            seen_xhosa = True
        elif seen_xhosa and seen_english:
            # Ambiguous - conservatively keep as Xhosa if we've already
            # seen Xhosa text, since removing it might strip valid Xhosa
            xhosa_words.append(w)
        elif seen_xhosa:
            xhosa_words.append(w)
        else:
            english_words.append(w)

    xhosa_text = " ".join(xhosa_words)
    english_text = " ".join(english_words)

    if xhosa_text and len(xhosa_text) > len(english_text):
        return xhosa_text, english_text
    return line.strip(), ""


def _is_english_word(word: str) -> bool:
    """Check if a word is likely English (not Xhosa)."""
    # English words are typically pure ASCII with no Xhosa patterns
    # Xhosa words often have: prefixes (u-, i-, a-, etc.), clicks (c, q, x),
    # apostrophes (Ntab'ozuko), or specific suffixes
    lower = word.lower().strip("'\"-.,;:!?")
    if not lower:
        return False
    if lower in ENGLISH_LEXICON:
        return True
    # Pure English: no Xhosa characters, no apostrophes typical of Xhosa
    # (Xhosa uses ' to mark contractions like Ntab'ozuko)
    has_xhosa_marker = any(c in word for c in "65") or "'" in word or "-" in word
    if has_xhosa_marker:
        return False
    # If it's in our known Xhosa words, it's not English
    if lower in XHOSA_WORDS:
        return False
    # Common English detection
    if lower in {"the", "and", "of", "to", "in", "is", "that", "it", "for",
                 "you", "on", "with", "as", "are", "was", "be", "at", "by",
                 "this", "have", "from", "or", "an", "they", "not", "but"}:
        return True
    # English words tend to end in common English suffixes
    if re.match(r'^[a-zA-Z]+$', lower) and len(lower) > 3:
        # Check if it looks like Xhosa (has typical Xhosa consonant patterns)
        xhosa_patterns = [r'^[uaio](b|e|f|h|i|k|l|m|n|o|p|r|s|t|v|w|x|y|z)',  # u-/i-/a- prefixes
                         r'\b(ng|cw|hl|kh|ph|th|xh)\w+',
                         r'\w+\'\w+',  # contractions
                         r'\w+5\b', r'\w+6\b']
        for pat in xhosa_patterns:
            if re.search(pat, lower):
                return False
        # Likely English if it ends in -tion, -ity, -ous, -ing, -ed, -ly, -er, -al
        if re.search(r'(tion|ity|ous|ing|ed|ly|er|al|ion|ness|ship|ment)$', lower):
            return True
    return False


def _is_xhosa_word(word: str) -> bool:
    """Check if a word has Xhosa characteristics."""
    lower = word.lower().strip("'\"-.,;:!?")
    if not lower:
        return False
    # Xhosa markers
    if "'" in word:  # Xhosa contractions like "Ntab'ozuko"
        return True
    if any(c in word for c in "65"):  # Traditional orthography
        return True
    # Xhosa noun class prefixes
    if re.match(r'^(u|a|i|e|o)(m|b|n|k|g|c|h|t|f|d|s|j|q|x|z)', lower):
        return True
    # Xhosa-specific patterns: ng-, kw-, etc.
    if lower.startswith("ng") or lower.startswith("kw") or lower.startswith("swa"):
        return True
    # Xhosa copula contractions: ya/ye/yo + noun class prefix
    # e.g., yendlela (ya+n Indlela), yokuphila (yo+kuPhila), yamntla (ya+umntla)
    if re.match(r'^(ya|ye|yo)(m|n|k|g|c|h|t|f|d|s|j|q|x|z|b|l|r|v|w)', lower):
        return True
    # Words in our known lexicon
    if lower in XHOSA_WORDS:
        return True
    return False


# Lexicons
ENGLISH_LEXICON: Set[str] = {
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
    "administrative", "legal", "constitutional", "official", "announcement",
    "document", "complete", "training", "dataset", "comprehensive",
    "fluency", "speaker", "professional", "evaluation", "research",
    "methodology", "exam", "writing", "skills", "creative", "criticism",
    "advanced", "essay", "appreciation", "critical", "interpretation",
    "visual", "literacy", "oral", "presentations", "discussions",
    "debate", "storytelling", "reading", "comprehension", "text",
    "interpretation", "evaluation", "production", "performance",
    "letters", "reports", "composition", "figures", "speech",
    "idioms", "proverbs", "expressions", "vocabulary", "spelling",
    "capitalization", "punctuation", "diacritics", "word", "boundaries",
    "morphology", "agreement", "aspect", "order", "constructions",
    "documented", "variants", "humanity", "customs", "practices",
    "ceremony", "initiation", "becoming", "adult", "leader", "person",
    "young", "people", "relatives", "written", "communication",
    "welcome", "child", "marriage", "burial", "department",
    "republic", "constitution", "supreme", "law", "humanity",
    "praise", "singer", "traditional", "poet", "customs", "rules",
    "traditional", "leader", "respected", "person", "young", "people",
    "relatives", "written", "communication", "welcome", "child",
    "marriage", "burial", "evaluation", "research", "methodology",
    "exam", "literary", "criticism", "advanced", "essay", "writing",
    "poem", "structured", "poetic", "form", "poetic", "expression",
    "clever", "use", "attitude", "speaker", "representation", "ideas",
    "sound", "repetition", "continuation", "sentence", "central",
    "organization", "story", "people", "time", "place", "manner",
    "expression", "rules", "language", "naming", "word", "action",
    "describing", "word", "connecting", "beginning", "words",
    "end", "changes", "conjugation", "describing", "verb",
    "now", "then", "before", "later", "when", "do",
    "praise", "singer", "traditional", "poet", "humanity", "through",
    "others", "laws", "traditional", "rules", "customs",
    "cultural", "practices", "chief", "traditional", "leader",
    "elder", "respected", "person", "children", "young", "people",
    "family", "relatives", "naming", "ceremony", "welcoming", "child",
    "marriage", "wedding", "burial", "funeral", "initiation",
    "becoming", "adult", "poetry", "song", "lullaby", "musical",
    "formal", "praise", "poetry", "traditional", "form", "praise",
    "singing", "wordplay", "clever", "metaphors", "figurative",
    "tone", "attitude", "symbols", "representation", "rhyme",
    "sound", "repetition", "enjambment", "continuation", "theme",
    "central", "structure", "organization", "characters", "people",
    "story", "setting", "time", "place", "style", "manner",
    "expression", "plot", "sequence", "events", "conclusion",
    "beginning", "novel", "novel", "utopian", "multiracial", "state",
    "defense", "xhosa", "law", "traditional", "complex", "structures",
    "first", "novel", "adaptation", "biblical", "samson", "historical",
    "biographical", "theses", "analyzing", "collected", "poems",
    "various", "academic", "available", "purchase", "publisher",
    "edition", "print", "digital", "ebook", "pdf", "isbn",
    "author", "year", "grade", "content", "focus", "literary",
    "analysis", "introduction", "novel", "study", "essay",
    "writing", "poetry", "appreciation", "drama", "study",
    "language", "appreciation", "exam", "preparation", "foundation",
    "phase", "senior", "further", "education", "training",
}

XHOSA_WORDS: Set[str] = {
    # Greetings
    "molo", "sawubona", "molweni", "ndiyabulela", "ndikhona", "kakuhle",
    "unjani", "ndiyaphila", "ndiyavuya", "ndixhomeke", "ndizelwe",
    "ndanduluka", "ndisenza", "ndiya", "ndine", "kwi", "ku", "xa",
    "na", "ke", "kodwa", "kuba", "njeng", "ngok", "emva", "pele",
    "malunga", "phakathi", "cwangcosi", "cwangcothi", "kweli", "kule",
    "kwabo", "kwe", "kulo", "uku", "uku", "ngo", "nga",

    # Nouns
    "umntu", "intetho", "ixhego", "umthetho", "izwe", "incwadi",
    "imibongo", "izibongo", "imihobe", "isihobe", "isityilio",
    "ubume", "umxholo", "ukuphila", "ubomi", "indlela", "imvelo",
    "imveliso", "inkosi", "umkhulu", "umama", "ubaba", "abantwana",
    "usapho", "abalinganiswa", "isimo", "into", "izinto",
    "indaba", "imibhalo", "izwi", "amagama", "igama", "isenzo",
    "isichazi", "isixhumanisi", "izanduko", "izigidimi",
    "izafobe", "izicanuko", "inzwelo", "isaziso", "ifomu", "ibhalo",
    "ubuhlobo", "intlonipho", "ubuninzi", "umsebenzi", "isikolo",
    "isithembiso", "indodla", "imvelo", "imveliso", "ilifa",
    "isiko", "intsomi", "izandla", "izithembiso", "izikhala",
    "izingqungquthela", "intombazane", "icici", "isisu", "iziko",
    "izibuko", "abaku", "izwe", "izwe", "izikhali",

    # Verbs
    "ukuthetha", "ukuba", "ukubhala", "ukufunda", "ukubala",
    "ukutshata", "ukufenxa", "ukusenza", "ukuqala", "ukuqhala",
    "ukudlala", "ukuqhuba", "ukubheka", "ukubhalwa", "ukuthengiswa",
    "ukuhlala", "ukuza", "hlonela", "hlonipha", "hlola", "hlalisa",
    "khetha", "kheshea", "coca", "xela", "qukutha", "phuma", "hamba",
    "vuka", "sho", "hlonipha", "hlala", "hluka", "hlula",
    "hle", "hlukunye", "thetha", "bhalisa", "funda", "bhala",
    "hala", "khetha", "phatha", "dlula", "dlulisa",
    "kufuneka", "kuya", "kwaye", "kubuye", "kuhlala", "kuthi",
    "kuhlala", "kuba", "ku", "ngokufanana", "ngokufanele",
    "ayikho", "kuhuno", "kucace", "kuthe", "kusho", "kunje",
    "hlala", "hle", "hlukunye", "bhalisa", "coca", "phuma",
    "hamba", "vuka", "sho", "hlonipha", "hlala",
    "hluka", "hlula", "thetha", "bhalisa", "funda", "bhala",

    # Cultural terms
    "imbongi", "ubuntu", "imithetho", "imasiko", "ukuthetha",
    "ukutshata", "ukufenxa", "ukusenza", "ubucala", "ubuhlobo",
    "ubuninzi", "ubomi", "izicoco", "izimotsheni", "izilwandle",
    "izikole", "amazwe", "intlukulela", "amandla", "umsebenzi",
    "umfana", "indoda", "indlovukazi", "isiduko", "isifundo",
    "isimboni", "isithembiso", "izisho", "izifundo",
    "umntu", "abantu", "izwe", "iindawo", "iintanomoya",
    "inkosi", "umkhulu", "umama", "ubaba", "abantwana",
    "usapho", "ukuthetha", "ukutshata", "ukufenxa", "ukusenza",
    "ukubhala", "ukufunda", "inkundla", "indlela",

    # Literary terms
    "ityala", "lamawele", "u-don", "jadu", "ingqumbo", "yeminyanya",
    "izibongo", "zoogxa", "iziganeko", "besizwe", "inzuzo",
    "imihobe", "imibongo", "isihobe", "uchongo", "lwamagama",
    "imifanekiso", "ntelekelelo", "ithoni", "imiqondiso",
    "isixhumanisi", "isingqisho", "injambamenti", "umxholo",
    "ubume", "abalinganiswa", "isimo", "sentlalo", "isityilio",
    "inqwaba", "isigama", "isenzo", "isichazi", "izanduko",
    "izigidimi", "ukuguquguquka", "isichazi-senzo",
    "kusekusa", "kusezulwini", "kuzakuba",
    "sikelel", "iAfrika", "iphondo", "lwayo",
    "mithandazo", "yethu", "lusapho", "ntla-langa",
    "busikeleze", "bakho", "siyakuthanda", "sithwale",
    "kakhulu", "kanini", "kancane", "kunje", "kude", "pele",
    "ngokufanana", "ngokufanele", "ngamalungelo",
    "uku", "ukuthi", "ukuba", "ukuze",
    "6a", "6el", "6ep", "6et", "6aya", "6ath",
    "6eli", "6enu", "6eth", "6end", "6ez",
    "bhala", "bhalwe", "bhali", "bhalisa",
    "ukuthi", "ukuba", "ukuze", "uku",
    "xa", "na", "ke", "kodwa", "kuba", "njeng",
    "ngok", "emva", "pele", "malunga", "phakathi",
    "cwangcosi", "cwangcothi", "kweli", "kule",
    "kwabo", "kwe", "kulo", "kwi", "ku",
    "ndiy", "ndiya", "ndine", "ndiya", "ndax",
    "ndiqala", "ndafunda", "ndabhala", "ndanduluka",
    "ndisenza", "ndizelwe", "ndixhomeke", "ndiyabulela",
    "ndikhona", "ndiyavuya", "ndiyaxhalisa", "ndiyaxolisa",
    "ndiyaphila", "ndiwena", "ndin", "kwi", "ku",
}


def classify_mqhayi_line(line: str) -> Tuple[str, str, str]:
    """Classify a Mqhayi line by work and section."""
    line_lower = line.lower()
    for pattern, work, section, source_type in MQHAYI_SECTIONS:
        if re.search(pattern, line_lower):
            return work or "General Mqhayi", section or "Unspecified", source_type
    return "Unknown Mqhayi", "Unclassified", "unclassified"


def classify_masikhanyise_line(line: str) -> Tuple[str, str, str]:
    """Classify a Masikhanyise line by section."""
    line_lower = line.lower()
    for pattern, title, section, source_type in MASIKHANYISE_SECTIONS:
        if re.search(pattern, line_lower):
            return title, section, source_type
    return "Masikhanyise", "Unclassified", "unclassified"


def extract_from_corpus_file(filepath: str, source: str) -> List[DataRecord]:
    """
    Process a single corpus file, extracting every authentic Xhosa sentence
    with provenance tracking.

    Strips English overlays from pedagogical content.
    Identifies and flags purely generated/synthetic content.
    """
    records: List[DataRecord] = []
    now = datetime.now(timezone.utc).isoformat()

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header/section markers
        if line.startswith("===") or line.startswith("—") or line.isdigit():
            continue

        # Extract Xhosa text, stripping English overlays
        xhosa_text, english_removed = strip_english_overlay(line)

        # Skip if no meaningful Xhosa text
        if not xhosa_text or len(xhosa_text) < 5:
            continue

        # Detect language class
        lang_class = detect_language(xhosa_text, source)

        # Classify section
        if source == "mqhayi":
            work, section, source_type = classify_mqhayi_line(xhosa_text)
        else:
            work, section, source_type = classify_masikhanyise_line(xhosa_text)

        # Determine confidence and validation status
        if english_removed:
            # Had English overlay -> candidate, needs validation
            confidence = "medium"
            validation_status = "candidate"
        else:
            # Pure text from corpus -> authoritative
            confidence = "high"
            validation_status = "authoritative"

        # Determine record-level source (do NOT mutate the filepath-derived source)
        record_source = source
        record_source_type = source_type
        is_generated = False

        # Only apply the repetitive check to mqhayi_basic.txt (known synthetic file)
        # For mqhayi_complete.txt and masikhanyise_complete.txt, all Xhosa text
        # is from the specified authoritative sources, even if it has English overlays
        is_basic_file = os.path.basename(filepath) == "mqhayi_basic.txt"

        if is_basic_file:
            # mqhayi_basic.txt is entirely synthetic (generated by create_xhosa_dataset.py)
            # ALL content from this file is candidate/generated, not authoritative
            is_generated = True
        elif _is_repetitive_generated(xhosa_text):
            is_generated = True

        if is_generated:
            confidence = "low"
            validation_status = "candidate"
            record_source = "generated"
            record_source_type = "synthetic"

        # Skip purely English lines
        if lang_class == "FOREIGN_LANGUAGE":
            continue

        records.append(DataRecord(
            text=xhosa_text,
            source=record_source,
            source_type=record_source_type,
            source_title=work,
            source_section=section,
            language="isiXhosa",
            provenance_type="authoritative" if record_source in ("mqhayi", "masikhanyise") else "generated",
            confidence=confidence,
            validation_status=validation_status,
            orthography=classify_orthography(xhosa_text),
            language_class=lang_class,
            retrieval_date=now,
            generator="corpus_file_extraction" if record_source == "generated" else "",
            original_text=line,
        ))

    return records


def _is_repetitive_generated(text: str) -> bool:
    """Detect lines that are repetitive/generated (same phrase repeated)."""
    words = text.split()
    if len(words) < 6:
        return False

    # Check for repetitive word patterns
    # e.g., "bemihla ngemihla" repeated
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = {}
    for bg in bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1

    # If any bigram appears more than 1/3 of total bigrams, it's repetitive
    max_count = max(bigram_counts.values())
    if max_count > len(bigrams) * 0.3 and len(bigrams) > 4:
        return True

    # Check for specific known patterns
    repetitive_patterns = [
        "bemihla ngemihla",
        "ngobhalwa ngamagama",
        "ekubhalwa ngobomi",
        "bembihla ngemihla",
        "kwe nto bemihla",
    ]
    for pat in repetitive_patterns:
        if text.lower().count(pat) >= 2:
            return True

    return False


# ─── Corpus consolidation ────────────────────────────────────────────────────

def run_corpus_extraction():
    """
    Process all three corpus files, extract authentic Xhosa text,
    strip English overlays, classify by source/provenance, and produce
    a unified reference corpus with full provenance tracking.
    """
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 64)
    print("  XNLP Data Pipeline - Corpus Deep Extraction")
    print("=" * 64)

    source_files = [
        ("corpus/mqhayi_complete.txt", "mqhayi"),
        ("corpus/masikhanyise_complete.txt", "masikhanyise"),
        ("corpus/mqhayi_basic.txt", "mqhayi"),
    ]

    all_records: List[DataRecord] = []
    seen_texts: Set[str] = set()

    for filepath, source in source_files:
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping")
            continue
        print(f"  Processing {filepath}...")
        records = extract_from_corpus_file(filepath, source)

        # Deduplicate
        unique = []
        for rec in records:
            text_key = rec.text.lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                rec.record_id = generate_record_id(rec.text, rec.source, len(all_records))
                rec.token_count = len(rec.text.split())
                unique.append(rec)

        print(f"    {len(records)} lines -> {len(unique)} unique records")
        all_records.extend(unique)

    # Separate by provenance
    authoritative = [r for r in all_records if r.validation_status == "authoritative"]
    candidate = [r for r in all_records if r.validation_status == "candidate"]

    print()
    print(f"  Total unique records: {len(all_records)}")
    print(f"  Authoritative: {len(authoritative)}")
    print(f"  Candidate: {len(candidate)}")
    print()

    # Language classification breakdown
    lang_dist = {}
    for rec in all_records:
        lang_dist[rec.language_class] = lang_dist.get(rec.language_class, 0) + 1
    print("  Language distribution:")
    for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {count}")

    # Source distribution
    src_dist = {}
    for rec in all_records:
        src_dist[rec.source] = src_dist.get(rec.source, 0) + 1
    print()
    print("  Source distribution:")
    for src, count in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    # Write authoritative records
    auth_path = "data/authoritative/mqhayi_masikhanyise.jsonl"
    with open(auth_path, "w", encoding="utf-8") as f:
        for rec in sorted(authoritative, key=lambda r: (r.source, r.source_type, r.text)):
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    print(f"\n  Written {len(authoritative)} authoritative records to {auth_path}")

    # Write candidate records
    cand_path = "data/candidate/candidate_from_corpus.jsonl"
    with open(cand_path, "w", encoding="utf-8") as f:
        for rec in sorted(candidate, key=lambda r: (r.source, r.text)):
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Written {len(candidate)} candidate records to {cand_path}")

    # Write manifest
    manifest = {
        "manifest_version": "1.0",
        "created": now,
        "corpus_source_files": [f for f, _ in source_files],
        "sources": dict(SOURCE_TIERS),
        "record_counts": {
            "authoritative": len(authoritative),
            "candidate": len(candidate),
            "total_unique": len(all_records),
            "total_lines_processed": sum(1 for f, _ in source_files if os.path.exists(f)),
        },
        "source_distribution": src_dist,
        "language_distribution": lang_dist,
        "confidence_distribution": {
            "high": sum(1 for r in all_records if r.confidence == "high"),
            "medium": sum(1 for r in all_records if r.confidence == "medium"),
            "low": sum(1 for r in all_records if r.confidence == "low"),
        },
        "orthography_distribution": {
            "modern": sum(1 for r in all_records if r.orthography == "modern"),
            "traditional": sum(1 for r in all_records if r.orthography == "traditional"),
            "mixed": sum(1 for r in all_records if r.orthography == "mixed"),
            "unknown": sum(1 for r in all_records if r.orthography == "unknown"),
        },
        "extraction_metadata": {
            "deduplication": "case-insensitive text dedup on stripped English overlay",
            "english_stripping": "Pedagogical English translations removed; Xhosa portion preserved",
            "repetitive_pattern_detection": "Lines with repeated bigrams classified as generated/candidate",
            "provenance": "Full source work + section + title tracking for every record",
        },
    }
    manifest_path = "data/manifests/reference_corpus_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Written manifest to {manifest_path}")

    # Write dataset report
    report = {
        "report_date": now,
        "total_records": len(all_records),
        "authoritative_records": len(authoritative),
        "candidate_records": len(candidate),
        "total_tokens": sum(r.token_count for r in all_records),
        "authoritative_tokens": sum(r.token_count for r in authoritative),
        "candidate_tokens": sum(r.token_count for r in candidate),
        "language_distribution": lang_dist,
        "source_distribution": src_dist,
        "confidence_distribution": manifest["confidence_distribution"],
        "orthography_distribution": manifest["orthography_distribution"],
        "validation_status_distribution": {
            "authoritative": len(authoritative),
            "candidate": len(candidate),
        },
        "recommendation": (
            "Corpus extracted from Mqhayi and Masikhanyise sources. "
            "English pedagogical overlays stripped. Repetitive/generated content "
            "classified as candidate. Authoritative records ready for direct training. "
            "Candidate records require validation before training use."
        ),
    }
    report_path = "data/reports/dataset_report_phase1.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Written dataset report to {report_path}")

    print()
    print("=== CORPUS EXTRACTION COMPLETE ===")
    return all_records


if __name__ == "__main__":
    records = run_corpus_extraction()

    # Now run validation on the candidate data
    print()
    from data_validator import CorrectionEngine, run_validation_pipeline, build_training_corpus

    # Re-validate the new candidate data
    engine = CorrectionEngine("data/authoritative/mqhayi_masikhanyise.jsonl")
    candidates = []
    cand_path = "data/candidate/candidate_from_corpus.jsonl"
    if os.path.exists(cand_path):
        with open(cand_path, "r", encoding="utf-8") as f:
            for line in f:
                candidates.append(DataRecord.from_dict(json.loads(line)))

    print(f"\n  Validating {len(candidates)} candidate records...")
    results = []
    for rec in candidates:
        result = engine.validate_and_correct(rec)
        results.append(result)

    validated = [r for r in results if r.validation_status == "validated"]
    rejected = [r for r in results if r.validation_status == "rejected"]
    uncertain = [r for r in results if r.validation_status == "uncertain"]

    print(f"  Validated: {len(validated)}")
    print(f"  Rejected:  {len(rejected)}")
    print(f"  Uncertain: {len(uncertain)}")

    # Write validation results
    val_path = "data/validated/cleaned_cadidate_texts.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for r in validated:
            # Convert to DataRecord with corrected text
            orig = next((c for c in candidates if c.record_id == r.record_id), None)
            if orig:
                orig.text = r.corrected_text
                orig.validation_status = "validated"
                orig.confidence = r.confidence
                orig.language_class = r.language_class
                f.write(json.dumps(orig.to_dict(), ensure_ascii=False) + "\n")

    rej_path = "data/rejected/rejected_candidates.jsonl"
    with open(rej_path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # Build the final training corpus
    build_training_corpus()
