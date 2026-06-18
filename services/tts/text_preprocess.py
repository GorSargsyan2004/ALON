from __future__ import annotations

import re
from typing import List, Tuple


_CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "can't": "cannot",
    "won't": "will not",
    "didn't": "did not",
    "i'd": "i would",
    "i'll": "i will",
    "i've": "i have",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
}

_PRONUNCIATIONS = {
    "alon": "Ah-lahn",
    "°C": "degree of celsius",
    "/h": "per hour",
    "Alon": "Ah-lahn",
    "1.": "first",
    "2.": "second",
    "3.": "third",
    "4.": "forth",
    "5.": "fifth",
    "6.": "sixth"
}


_ENERGETIC = {"laugh", "laughs", "chuckle", "giggle", "excited", "gasp"}
_CALM = {"sigh", "sighs", "exhale", "whispers", "whisper", "calm"}
_SERIOUS = {"angry", "mad", "stern", "serious", "annoyed"}


def expand_contractions(text: str) -> str:
    if not text:
        return ""
    # Normalize curly apostrophes to plain ones so contractions match.
    text = text.replace("’", "'").replace("‘", "'")

    def replace(match: re.Match) -> str:
        src = match.group(0)
        repl = _CONTRACTIONS.get(src.lower(), src)
        if src.isupper():
            return repl.upper()
        if src[0].isupper():
            return repl[0].upper() + repl[1:]
        return repl

    pattern = r"\b(" + "|".join(map(re.escape, _CONTRACTIONS.keys())) + r")\b"
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def apply_pronunciations(text: str) -> str:
    if not text:
        return ""
    out = text
    for src, repl in _PRONUNCIATIONS.items():
        out = re.sub(rf"\b{re.escape(src)}\b", repl, out, flags=re.IGNORECASE)
    return out


def apply_unit_expansions(text: str) -> str:
    if not text:
        return ""
    out = text
    out = re.sub(r"°\s*c", " degrees celsius", out, flags=re.IGNORECASE)
    out = re.sub(r"°\s*f", " degrees fahrenheit", out, flags=re.IGNORECASE)
    out = out.replace("/h", " per hour")
    return out


def extract_stage_directions(text: str) -> Tuple[str, List[str]]:
    if not text:
        return "", []

    tags: List[str] = []

    def grab_tokens(content: str):
        for tok in re.findall(r"[a-zA-Z']+", content.lower()):
            tags.append(tok)

    def repl_md_keep(match: re.Match) -> str:
        return match.group(1)

    def repl_stage_star(match: re.Match) -> str:
        grab_tokens(match.group(1))
        return " "

    def repl_paren(match: re.Match) -> str:
        grab_tokens(match.group(1))
        return " "

    # Preserve basic math multiplication before stage-direction parsing.
    text = re.sub(r"(?<=\d)\s*\*\s*(?=\d)", " times ", text)

    # Markdown emphasis first (**bold**, __bold__), then single markers.
    text = re.sub(r"\*\*([^*]+)\*\*", repl_md_keep, text)
    text = re.sub(r"__([^_]+)__", repl_md_keep, text)
    text = re.sub(r"_([^_]+)_", repl_md_keep, text)
    # Treat single-star segments as stage directions only when not math-adjacent.
    text = re.sub(r"(?<!\d)\*([^*]+)\*(?!\d)", repl_stage_star, text)
    text = re.sub(r"\(([^)]+)\)", repl_paren, text)
    # Remove any leftover markdown markers so TTS never reads asterisks.
    text = re.sub(r"[*_`~#]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, tags


def choose_voice_from_tags(tags: List[str], default_voice: str) -> str:
    tag_set = set(tags or [])
    if tag_set & _SERIOUS:
        return "serious"
    if tag_set & _CALM:
        return "calm"
    if tag_set & _ENERGETIC:
        return "energetic"
    return (default_voice or "neutral").lower()


def preprocess_for_tts(text: str, default_voice: str) -> dict:
    original = text or ""
    stripped, tags = extract_stage_directions(original)
    expanded = expand_contractions(stripped)
    expanded = apply_unit_expansions(expanded)
    expanded = apply_pronunciations(expanded)
    clean = re.sub(r"\s+", " ", expanded).strip()

    voice = choose_voice_from_tags(tags, default_voice)

    changed = {
        "tags_removed": len(tags),
        "contractions_expanded": int(expanded != stripped),
    }

    return {
        "clean_text": clean,
        "voice": voice,
        "tags": tags,
        "changed": changed,
    }
