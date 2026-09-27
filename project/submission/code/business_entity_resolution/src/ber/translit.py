"""Rule-based Indic -> Latin transliteration and a consonant skeleton.

The Unicode Indic blocks U+0900..U+0D7F (Devanagari, Bengali, Gurmukhi,
Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) share the ISCII-derived
parallel layout: the same offset inside each 128-code-point block is the same
letter. One offset table therefore covers every script. The skeleton (vowels
dropped, phonetically close consonants merged, repeats collapsed) lets
"प्राइवेट लिमिटेड" and "Private Limited" compare as "prvt lmtd".
"""

from __future__ import annotations

_BASE, _END = 0x0900, 0x0D80

_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
           0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h"}
_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x46: "e",
          0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_DIGIT0 = 0x66


def transliterate(text: str) -> str:
    """Indic characters -> rough Latin; everything else passes through."""
    out: list[str] = []
    pending_a = False  # inherent vowel of the last consonant
    for ch in text:
        cp = ord(ch)
        if _BASE <= cp < _END:
            off = (cp - _BASE) % 0x80
            if off in _CONS:
                if pending_a:
                    out.append("a")
                out.append(_CONS[off])
                pending_a = True
                continue
            if off in _SIGNS:
                out.append(_SIGNS[off])
                pending_a = False
                continue
            if off == _VIRAMA:
                pending_a = False
                continue
            if off == _NUKTA:
                continue
            if pending_a and off not in _NASAL:
                out.append("a")
            pending_a = False
            if off in _VOWELS:
                out.append(_VOWELS[off])
            elif off in _NASAL:
                out.append(_NASAL[off])
            elif _DIGIT0 <= off < _DIGIT0 + 10:
                out.append(str(off - _DIGIT0))
            continue
        # non-Indic char ends a word: word-final schwa is dropped
        pending_a = False
        out.append(ch)
    return "".join(out)


# voicing pairs merged too: Tamil writes b/p, d/t, g/k with one letter
_MERGE = str.maketrans({"c": "k", "q": "k", "w": "v", "z": "j", "x": "k", "f": "p",
                        "b": "p", "d": "t", "g": "k"})
_VOWEL_SET = frozenset("aeiouy")


def skeleton_token(tok: str) -> str:
    t = tok.replace("ph", "p").replace("sh", "s").replace("ch", "k").replace("th", "t") \
           .replace("dh", "d").replace("bh", "b").replace("kh", "k").replace("gh", "g") \
           .replace("jh", "j")
    t = t.translate(_MERGE)
    if not t:
        return ""
    head, rest = t[0], [c for c in t[1:] if c not in _VOWEL_SET and c != "h"]
    s = [head] + rest
    collapsed = [s[0]]
    for c in s[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
    return "".join(collapsed)


def skeleton(normalized: str) -> list[str]:
    """Consonant-skeleton tokens of an already-normalized string."""
    toks = transliterate(normalized).split()
    return [sk for sk in (skeleton_token(t) for t in toks) if sk]
