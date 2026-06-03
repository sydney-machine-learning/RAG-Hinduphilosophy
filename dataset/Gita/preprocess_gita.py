"""
Preprocessing script for two problematic Gita text files:
  1. 455_gita_roman.txt   - Gita Press PDF conversion with Devanagari font-encoding artifacts
  2. bhagavad-gita-by-mahesh-yogi.txt - custom Sanskrit transliteration + English translation

Strategy:
  File 1: Remove pure-Sanskrit lines (low ASCII ratio), fix PDF ligature/math-symbol artifacts,
          normalize diacritics to plain ASCII.
  File 2: Decode verse numbers, keep only English prose lines, strip Sanskrit transliteration blocks.
"""

import re
import unicodedata
from pathlib import Path

GITA_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def ascii_ratio(line):
    return sum(1 for c in line if ord(c) < 128) / max(len(line), 1)


def strip_diacritics(text):
    """NFKD-decompose and drop combining marks: a-ring -> a, e-acute -> e, etc."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def collapse_blank_lines(text, max_consecutive=2):
    return re.sub(r"\n{%d,}" % (max_consecutive + 1), "\n" * max_consecutive, text)


# ---------------------------------------------------------------------------
# File 1: 455_gita_roman.txt
# Problem: Gita Press PDF -> OCR/font-mapping converted Devanagari to Mac PDF
#          private-area chars and used math symbols (integral, partial, delta)
#          as stand-ins for Sanskrit diacritics.
# ---------------------------------------------------------------------------

# Math symbols repurposed as Sanskrit diacritics in this PDF font mapping
MATH_TO_ASCII = {
    "∫": "i",   # integral sign used for long-i
    "∂": "d",   # partial differential used for retroflex-d
    "∆": "n",   # increment used for velar nasal
    "≈": "h",   # almost-equal used for visarga
    "◊": "",    # lozenge - decorative separator, remove
    "∑": "S",   # summation sign
    "≥": "",    # greater-or-equal - remove
    "≤": "",    # less-or-equal - remove
    "∞": "",    # infinity - remove
}

# PDF ligatures that should be letter sequences
LIGATURE_FIX = {
    "ﬂ": "fl",
    "ﬁ": "fi",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "Œ": "OE",
    "œ": "oe",
    "Æ": "AE",
    "æ": "ae",
    "ƒ": "fi",   # f-hook used as fi ligature in some Mac PDF fonts
}

# Typographic / symbol replacements
TYPO_FIX = {
    "—": "--",  # em dash
    "–": "-",   # en dash
    "⁄": "/",   # fraction slash
    "µ": "u",   # micro sign used as mu
}

# PDF artifacts with no text value - remove entirely
REMOVE_CHARS = set("¢¤§¨¶❖")


def fix_line_gita_press(line):
    for src, dst in MATH_TO_ASCII.items():
        line = line.replace(src, dst)
    for src, dst in LIGATURE_FIX.items():
        line = line.replace(src, dst)
    for src, dst in TYPO_FIX.items():
        line = line.replace(src, dst)
    line = "".join(c for c in line if c not in REMOVE_CHARS)
    line = strip_diacritics(line)
    line = "".join(c for c in line if ord(c) < 128)
    return line


def preprocess_455_gita(src, dst):
    content = src.read_text(encoding="utf-8")
    content = content.replace("\x0c", "\n")   # form feed -> blank line
    lines = content.split("\n")
    kept, skipped = 0, 0
    clean_lines = []

    for line in lines:
        stripped = line.rstrip()
        if stripped:
            ratio = ascii_ratio(stripped)
            if ratio < 0.60:
                skipped += 1
                continue
            stripped = fix_line_gita_press(stripped)
            kept += 1
        clean_lines.append(stripped)

    result = collapse_blank_lines("\n".join(clean_lines))
    dst.write_text(result, encoding="utf-8")

    return {"lines_kept": kept, "lines_skipped_sanskrit": skipped, "output_chars": len(result)}


# ---------------------------------------------------------------------------
# File 2: bhagavad-gita-by-mahesh-yogi.txt
# Problem: File interleaves Sanskrit transliteration (custom encoding using
#          ASCII symbols: ; / 0-9 as phoneme markers) with English translation.
#          Verse numbers encoded as 00!00, 00@00, 00#00 (! = 1, @ = 2, etc.)
# ---------------------------------------------------------------------------

VERSE_DIGIT = {"!": "1", "@": "2", "#": "3", "$": "4", "%": "5",
               "^": "6", "&": "7", "*": "8", "(": "9", ")": "0"}

VERSE_RE = re.compile(r"^00([!@#$%^&*()]+)00$")
PAGE_RE  = re.compile(r"^Page \d+$")

# Sanskrit-marker Unicode codepoints specific to this file:
# curly quotes used as vowel diacritics, and other special chars
SANSKRIT_CODEPOINTS = frozenset([
    0x2018,  # left single quotation mark (used as Sanskrit virama)
    0x2019,  # right single quotation mark (used as Sanskrit virama)
    0x201c,  # left double quotation mark
    0x201d,  # right double quotation mark
    0x201e,  # double low-9 quotation mark
    0x00e5,  # a-ring (aa vowel in this transliteration)
    0x00ca,  # E-circumflex (E with macron approx)
    0x221a,  # square root sign (used as Sanskrit char)
    0x00a2,  # cent sign (Sanskrit marker)
    0x0192,  # f-hook (fi-ligature artifact)
    0x00ae,  # registered trademark (Sanskrit marker in this encoding)
    0x25ca,  # lozenge
    0x221e,  # infinity
    0x2248,  # almost equal
    0x00cc,  # I-grave (Sanskrit long-I marker)
    0x2265,  # greater-or-equal
])


def decode_verse_number(s):
    m = VERSE_RE.match(s.strip())
    if m:
        digits = "".join(VERSE_DIGIT.get(c, c) for c in m.group(1))
        return "[Verse %s]" % digits
    return None


def is_sanskrit_transliteration(line):
    """
    Identify Sanskrit transliteration lines in the Mahesh Yogi file.

    Markers of the custom ITRANS-like encoding used here:
      - Digit embedded inside a word: v7, t9, n9 (vowel length markers)
      - Digit immediately before a letter: e5, u+e5
      - Semicolon after a letter: v;c, smvet;
      - Line starts with / (Sanskrit verb prefix in this scheme)
      - Line ends with digit+letter: t9
      - Curly quotes used as vowel diacritics
      - Other Sanskrit-specific Unicode chars
      - ITRANS consonant clusters: lowercase+UPPERCASE+lowercase (svR, inT)
      - More than 5% non-ASCII chars
    """
    s = line.strip()
    if not s:
        return False
    if re.match(r"^0+$", s):       # standalone 0 separators
        return True
    if VERSE_RE.match(s):          # verse number lines - handled separately
        return True
    if PAGE_RE.match(s):           # Page N markers
        return True
    if re.search(r"[a-zA-Z][0-9][a-zA-Z]", s):   # digit inside word
        return True
    if re.search(r"[0-9][a-zA-Z]", s):            # digit before letter
        return True
    if re.search(r"[a-zA-Z];", s):                # semicolon after letter
        return True
    if re.match(r"^/", s):                        # starts with /
        return True
    if re.search(r"[a-zA-Z][0-9]$", s):           # ends with digit after letter
        return True
    if any(ord(c) in SANSKRIT_CODEPOINTS for c in s):
        return True
    # ITRANS consonant clusters: lowercase+UPPERCASE+lowercase (svR, inT, kiLb)
    if re.search(r"[a-z][A-Z][a-z]", s):
        return True
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii / len(s) >= 0.05:
        return True
    return False


def preprocess_mahesh_yogi(src, dst):
    lines = src.read_text(encoding="utf-8").splitlines()
    clean_lines = []
    verses_decoded, lines_skipped, lines_kept = 0, 0, 0

    for line in lines:
        stripped = line.strip()

        verse_label = decode_verse_number(stripped)
        if verse_label:
            clean_lines.append(verse_label)
            verses_decoded += 1
            continue

        if is_sanskrit_transliteration(line):
            lines_skipped += 1
            continue

        clean_lines.append(stripped)
        lines_kept += 1

    result = collapse_blank_lines("\n".join(clean_lines))
    dst.write_text(result, encoding="utf-8")

    return {
        "verses_decoded": verses_decoded,
        "lines_kept": lines_kept,
        "lines_skipped_sanskrit": lines_skipped,
        "output_chars": len(result),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    files = {
        "455_gita_roman.txt": (
            GITA_DIR / "455_gita_roman.txt",
            GITA_DIR / "455_gita_roman_clean.txt",
            preprocess_455_gita,
        ),
        "bhagavad-gita-by-mahesh-yogi.txt": (
            GITA_DIR / "bhagavad-gita-by-mahesh-yogi.txt",
            GITA_DIR / "bhagavad-gita-by-mahesh-yogi_clean.txt",
            preprocess_mahesh_yogi,
        ),
    }

    for name, (src, dst, fn) in files.items():
        print("\n" + "=" * 60)
        print("Processing: %s" % name)
        stats = fn(src, dst)
        print("  Output -> %s" % dst.name)
        for k, v in stats.items():
            label = k.replace("_", " ").capitalize()
            print("  %s: %s" % (label, "{:,}".format(v) if isinstance(v, int) else v))
