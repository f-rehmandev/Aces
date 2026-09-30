"""
PII redactor — spec §11.1.

Replaces emails and phone numbers in a *user prompt* with stable tokens
(``<EMAIL_1>``, ``<PHONE_1>``) before the prompt is cached or sent to an
LLM. Keeps a mapping so callers can restore the originals if needed.

Important: this is NOT the same as the output-side PII guard from §27.1.
That one operates on scraped records before they reach deliverables.
This one operates on prompts only.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Conservative email match.
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Phone candidate: an optional country code, an optional parens area code,
# then two or more digit groups with optional separators. Groups may be
# 1-4 digits so long trailing numbers (e.g. "1234567") still match as one.
#
# The real filter is the "must contain at least 9 digits" check in
# _phone_sub, not the regex itself.
_PHONE_CANDIDATE_RE = re.compile(
    r"(?:\+\d{1,3}[\s\-.]?)?"        # optional +CC
    r"(?:\(\d{1,4}\)[\s\-.]?)?"      # optional (area)
    r"\d{2,4}"                        # first group: at least 2 digits
    r"(?:[\s\-.]?\d{1,4}){1,5}"      # one to five more groups
)

# Reject ISO dates mistaken for phones.
_ISO_DATE_RE = re.compile(r"^\d{4}[\-/]\d{2}[\-/]\d{2}$")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RedactionResult:
    original: str
    redacted: str
    replacements: dict[str, str] = field(default_factory=dict)
    # replacements maps token -> original value, e.g. {"<EMAIL_1>": "a@b.com"}

    def restore(self, text: str) -> str:
        """Put the originals back (inverse of redaction)."""
        for token, original in self.replacements.items():
            text = text.replace(token, original)
        return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact(text: str) -> RedactionResult:
    """
    Replace emails and phones in `text` with stable tokens.

    Tokens are assigned in first-seen order, one counter per kind:
        <EMAIL_1>, <EMAIL_2>, ...
        <PHONE_1>, <PHONE_2>, ...

    Duplicate values reuse the same token.
    """
    if not text:
        return RedactionResult(original=text, redacted=text)

    replacements: dict[str, str] = {}
    reverse: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {"EMAIL": 0, "PHONE": 0}

    def _token(kind: str, original: str) -> str:
        key = (kind, original)
        if key in reverse:
            return reverse[key]
        counters[kind] += 1
        tok = f"<{kind}_{counters[kind]}>"
        reverse[key] = tok
        replacements[tok] = original
        return tok

    # --- emails first (so digit runs inside an email can't be phone-matched) ---
    def _email_sub(match: re.Match) -> str:
        return _token("EMAIL", match.group(0))

    redacted = EMAIL_RE.sub(_email_sub, text)

    # --- phones ---
    def _phone_sub(match: re.Match) -> str:
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        if len(digits) < 9:
            return value
        if _ISO_DATE_RE.match(value.strip()):
            return value
        return _token("PHONE", value)

    redacted = _PHONE_CANDIDATE_RE.sub(_phone_sub, redacted)

    return RedactionResult(
        original=text,
        redacted=redacted,
        replacements=replacements,
    )


def has_pii(text: str) -> bool:
    """Cheap check: does the text contain any email or phone-shaped token?"""
    return bool(EMAIL_RE.search(text) or _PHONE_CANDIDATE_RE.search(text))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = redact(
        "Contact Alice at alice@example.com or +92 300 1234567. "
        "Also bob@example.org, phone 0300-1234567."
    )
    print("redacted:", r.redacted)
    print("tokens:", r.replacements)
    assert "<EMAIL_1>" in r.redacted
    assert "<EMAIL_2>" in r.redacted
    assert "<PHONE_1>" in r.redacted
    assert "<PHONE_2>" in r.redacted
    assert r.replacements["<PHONE_1>"] == "+92 300 1234567"
    assert r.replacements["<PHONE_2>"] == "0300-1234567"

    # Round trip
    assert r.restore(r.redacted) == r.original

    # No PII
    r2 = redact("find the best laptop prices in Pakistan")
    assert r2.redacted == r2.original
    assert r2.replacements == {}

    # ISO dates are not phones
    r3 = redact("created on 2026-09-22")
    assert "2026-09-22" in r3.redacted

    # Duplicate reuse
    r4 = redact("a@b.com and a@b.com")
    assert r4.redacted.count("<EMAIL_1>") == 2
    assert len(r4.replacements) == 1

    print("PII redactor OK.")