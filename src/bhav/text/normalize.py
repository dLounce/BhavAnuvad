"""Single text normalizer for Devanagari (Hindi) and Tamil.

Every string that crosses a module boundary passes through here. Keeping one
implementation means ASR output, MT input, and TTS input can never disagree
about digits, punctuation, or Unicode form.
"""

from __future__ import annotations

import re
import unicodedata

LANGS = ("hi", "ta")

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")

# Danda and full stop are interchangeable in practice; collapse to one form per script.
_PUNCT_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
}

_HI_RANGE = r"\u0900-\u097F"
_TA_RANGE = r"\u0B80-\u0BFF"

_ALLOWED = {
    "hi": re.compile(rf"[^{_HI_RANGE}0-9a-zA-Z\s.,!?'\"\-]"),
    "ta": re.compile(rf"[^{_TA_RANGE}0-9a-zA-Z\s.,!?'\"\-]"),
}

_WS = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([.,!?])\1+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?])")


def normalize(text: str, lang: str, *, strip_punct: bool = False) -> str:
    """Normalize a Hindi or Tamil string to the project's canonical form.

    NFC Unicode, ASCII digits, collapsed whitespace, script-foreign characters
    dropped. `strip_punct=True` is for WER/CER scoring, not for TTS input.
    """
    if lang not in LANGS:
        raise ValueError(f"unsupported lang {lang!r}, expected one of {LANGS}")
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)

    text = text.translate(_DEVANAGARI_DIGITS).translate(_TAMIL_DIGITS)
    text = text.replace("\u0964", ".").replace("\u0965", ".")  # danda, double danda

    text = _ALLOWED[lang].sub(" ", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)

    if strip_punct:
        text = re.sub(r"[.,!?'\"\-]", " ", text)

    return _WS.sub(" ", text).strip()


def normalize_for_scoring(text: str, lang: str) -> str:
    """Lowercased, punctuation-free form used by WER/CER/chrF metrics only."""
    return normalize(text, lang, strip_punct=True).lower()


def detect_script(text: str) -> str | None:
    """Return 'hi' or 'ta' from character counts, or None if neither dominates."""
    hi = len(re.findall(rf"[{_HI_RANGE}]", text))
    ta = len(re.findall(rf"[{_TA_RANGE}]", text))
    if hi == ta == 0:
        return None
    return "hi" if hi > ta else "ta"
