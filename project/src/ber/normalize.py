"""Raw-preserving normalization views.

Rules (master.md 8.1-8.4):
- normalize_text: NFKC -> casefold -> non-alnum/non-mark chars to spaces ->
  collapse whitespace. Raw strings are always kept alongside these views.
- latin_accent_fold: strips combining marks ONLY when their base character is
  a Latin letter after NFD decomposition. Indic scripts (whose vowel signs
  are combining marks) are preserved byte-for-byte.
- tokens: alnum runs of length >= 2 in normalized text.
- trigrams: char 3-grams of tokens with length >= 4.
- is_indic: any character in U+0900-U+0DFF.
"""

from __future__ import annotations

import unicodedata

_INDIC_MIN = 0x0900
_INDIC_MAX = 0x0DFF


def _punct_to_space_table() -> dict[int, str]:
    """Translate table mapping every non-alnum/non-mark codepoint to space."""
    table: dict[int, str] = {}
    for cp in range(0x110000):
        c = chr(cp)
        if c.isalnum() or unicodedata.category(c).startswith("M"):
            continue
        table[cp] = " "
    return table


_PUNCT_TABLE = _punct_to_space_table()


def _to_space(ch: str) -> str:
    if ch.isalnum() or unicodedata.category(ch).startswith("M"):
        return ch
    return " "


def normalize_text(text: str) -> str:
    """NFKC + casefold + punctuation-to-space + whitespace collapse."""
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    folded = nfkc.casefold()
    mapped = folded.translate(_PUNCT_TABLE)
    if "  " in mapped or mapped != mapped.strip():
        return " ".join(mapped.split())
    return mapped


def latin_accent_fold(text: str) -> str:
    """Remove combining marks that decorate Latin letters; keep all others.

    Implemented without any external transliteration library (stdlib only).
    Fast path: pure-ASCII input is returned unchanged.
    """
    if not text or text.isascii():
        return text
    nfd = unicodedata.normalize("NFD", text)
    if nfd.isascii():
        return text  # no decomposable accents at all
    out: list[str] = []
    for ch in nfd:
        if unicodedata.category(ch) == "Mn":
            # keep the mark unless the previous emitted base is a Latin letter
            if out and out[-1].isascii() and out[-1].isalnum():
                continue
            # mark with no Latin base: preserve it (e.g., stray diacritic)
            out.append(ch)
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def fold_view(normalized: str) -> str:
    """Accent-folded view of an already-normalized string (Latin only)."""
    return latin_accent_fold(normalized)


def tokens(normalized: str) -> list[str]:
    """Alnum tokens of length >= 2, in order, from normalized text."""
    if not normalized:
        return []
    return [t for t in normalized.split() if len(t) >= 2]


def trigrams(normalized: str) -> list[str]:
    """Char 3-grams of every token with length >= 4, in token order."""
    result: list[str] = []
    for tok in tokens(normalized):
        if len(tok) < 4:
            continue
        for i in range(len(tok) - 2):
            result.append(tok[i : i + 3])
    return result


def is_indic(text: str) -> bool:
    """True if any character falls in the U+0900-U+0DFF block."""
    return any(_INDIC_MIN <= ord(c) <= _INDIC_MAX for c in text)
