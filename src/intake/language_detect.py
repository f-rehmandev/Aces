"""
Language family detection — spec §11.1.

This is a lightweight, deterministic classifier: it looks at Unicode
character ranges and returns the dominant script family. That's enough
for §11.1's needs (choosing a normalizer, choosing a prompt template)
without pulling in a heavy ML dependency.

Honest limitation: it does NOT distinguish e.g. Spanish from English,
because both are Latin-script. It distinguishes *script families*, which
is what the downstream code actually branches on.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# Ordered by specificity: if any of these ranges fires, that family wins.
_SCRIPT_RANGES: list[tuple[str, re.Pattern]] = [
    ("cjk",         re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")),
    ("korean",      re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")),
    ("arabic",      re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff]")),
    ("hebrew",      re.compile(r"[\u0590-\u05ff]")),
    ("cyrillic",    re.compile(r"[\u0400-\u04ff]")),
    ("devanagari",  re.compile(r"[\u0900-\u097f]")),
    ("thai",        re.compile(r"[\u0e00-\u0e7f]")),
    ("greek",       re.compile(r"[\u0370-\u03ff]")),
]

_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")


@dataclass
class LanguageGuess:
    family: str          # "latin" | "cjk" | "arabic" | ...
    confidence: float    # 0..1 — rough; see docstring
    dominant_script: str # same as family for readability


def detect_language(text: str) -> LanguageGuess:
    """
    Return the dominant script family of `text`.

    Families: latin, cjk, korean, arabic, hebrew, cyrillic, devanagari,
    thai, greek, or "unknown" if no letters are present.
    """
    if not text or not text.strip():
        return LanguageGuess(family="unknown", confidence=0.0,
                             dominant_script="unknown")

    counts: dict[str, int] = {}

    for name, rx in _SCRIPT_RANGES:
        n = len(rx.findall(text))
        if n:
            counts[name] = n

    latin_count = len(_LATIN_RE.findall(text))
    if latin_count:
        counts["latin"] = latin_count

    if not counts:
        return LanguageGuess(family="unknown", confidence=0.0,
                             dominant_script="unknown")

    family = max(counts, key=counts.get)
    total = sum(counts.values())
    confidence = counts[family] / total if total else 0.0

    return LanguageGuess(
        family=family,
        confidence=round(confidence, 3),
        dominant_script=family,
    )


def is_latin(text: str) -> bool:
    """Convenience: is the dominant script Latin-based?"""
    return detect_language(text).family == "latin"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assert detect_language("find the best laptop prices").family == "latin"
    assert detect_language("مرحبا بالعالم").family == "arabic"
    assert detect_language("你好世界").family == "cjk"
    assert detect_language("안녕하세요").family == "korean"
    assert detect_language("привет мир").family == "cyrillic"
    assert detect_language("नमस्ते दुनिया").family == "devanagari"
    assert detect_language("γειά σου").family == "greek"
    assert detect_language("").family == "unknown"
    assert detect_language("12345").family == "unknown"

    # Mixed: dominant script wins
    g = detect_language("Hello 你好 你好 你好")
    assert g.family == "cjk", g

    print("Language detector OK.")