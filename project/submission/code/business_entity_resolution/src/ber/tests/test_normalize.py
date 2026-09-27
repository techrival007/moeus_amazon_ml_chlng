"""Raw-preserving normalization views and record ingestion.

Contract from master.md sections 8.1-8.4, 21.3:
- normalize_text: NFKC, casefold, non-alnum/non-mark chars to space, collapse
  whitespace. Never destroys Indic text.
- latin_accent_fold: removes combining marks only when their base character
  is a Latin letter (post-NFD); leaves every other script untouched.
- tokens: alnum runs of length >= 2 from normalized text.
- trigrams: char 3-grams of tokens with length >= 4.
- is_indic: any character in U+0900-U+0DFF (script diagnostic, not language ID).
- "NA"/"N/A"-like strings are literal text, never silently converted to null;
  missingness is decided only by an empty/blank raw field.
"""

import pytest

from ber.normalize import (
    is_indic,
    latin_accent_fold,
    normalize_text,
    tokens,
    trigrams,
)


class TestNormalizeText:
    def test_casefold_and_punctuation_to_space(self):
        assert normalize_text("B+ Retail Inc") == "b retail inc"
        assert normalize_text("  Hello,   World!  ") == "hello world"

    def test_preserves_indic_script(self):
        s = "राम मार्केटिंग प्राइवेट लिमिटेड"
        assert normalize_text(s) == s.casefold()  # casefold is identity here

    def test_preserves_digits(self):
        assert normalize_text("1795 Westchester Drive, Apt. 2G") == "1795 westchester drive apt 2g"

    def test_nfkc_folds_compat_characters(self):
        assert normalize_text("①②③") == "123"  # NFKC compatibility digits
        assert normalize_text("ﬁne Café") == "fine café"  # ligature folded, accent KEPT in base view

    def test_empty_and_blank(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestLatinAccentFold:
    def test_folds_latin_accents(self):
        assert latin_accent_fold(normalize_text("Béque")) == "beque"
        assert latin_accent_fold(normalize_text("CHAMPS ÉLYSÉES")) == "champs elysees"
        assert latin_accent_fold(normalize_text("Nétwork Cáre")) == "network care"

    def test_does_not_touch_ascii(self):
        assert latin_accent_fold("orelee s barbershop") == "orelee s barbershop"

    def test_does_not_strip_devanagari_marks(self):
        # Devanagari vowel signs are combining marks; they must survive
        s = "बिजनेस प्राइवेट लिमिटेड"
        assert latin_accent_fold(s) == s

    def test_mixed_latin_and_indic(self):
        s = normalize_text("Café बिजनेस")
        out = latin_accent_fold(s)
        assert "cafe" in out
        assert "बिजनेस" in out

    def test_fold_then_unfold_is_prefix_stable(self):
        # folding a second time changes nothing
        once = latin_accent_fold(normalize_text("École Élémentaire"))
        assert latin_accent_fold(once) == once


class TestTokensAndTrigrams:
    # tokens/trigrams take NORMALIZED text as input

    def test_tokens_drop_length_one(self):
        assert tokens(normalize_text("b retail inc")) == ["retail", "inc"]
        assert tokens(normalize_text("unit 2g floor")) == ["unit", "2g", "floor"]

    def test_tokens_split_on_punctuation(self):
        # punctuation becomes a space, so "H.No.16" -> "h no 16"
        assert tokens(normalize_text("H.No.16-11-23/37/A, 2Nd Floor")) == [
            "no", "16", "11", "23", "37", "2nd", "floor",
        ]

    def test_tokens_empty(self):
        assert tokens(normalize_text("---")) == []
        assert tokens("") == []

    def test_trigrams_of_long_tokens_only(self):
        # "services" = s e r v i c e s -> ser, erv, rvi, vic, ice, ces
        assert trigrams(normalize_text("global services")) == [
            "glo", "lob", "oba", "bal",  # global (6 chars -> 4)
            "ser", "erv", "rvi", "vic", "ice", "ces",  # services (8 chars -> 6)
        ]

    def test_trigram_order_follows_token_order(self):
        assert trigrams(normalize_text("abcde xyzw")) == ["abc", "bcd", "cde", "xyz", "yzw"]

    def test_short_tokens_have_no_trigrams(self):
        assert trigrams(normalize_text("pc llc b2b")) == []

    def test_trigrams_empty(self):
        assert trigrams("") == []


class TestIsIndic:
    def test_detects_devanagari(self):
        assert is_indic("राम मार्केटिंग")
        assert is_indic("कंपनी")

    def test_detects_other_indic_blocks(self):
        assert is_indic("குளோபல்")  # Tamil U+0B80-U+0BFF

    def test_ascii_is_not_indic(self):
        assert not is_indic("Global Business Inc")
        assert not is_indic("")

    def test_latin_accents_are_not_indic(self):
        assert not is_indic("Béque Nétwork")
