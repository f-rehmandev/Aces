"""
Secret scrubbing — spec §52.

Before anything goes to a log, an error message, or a user-visible string,
it should pass through `scrub()`. This replaces anything that looks like a
credential with a placeholder.

Design:
    - Pattern-based, not allow-list-based: we never know all the provider
      key formats, so we cover the common ones and everything else that
      looks high-entropy.
    - Preserves enough of the original for debugging.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_NAMED_PATTERNS: list[tuple[str, re.Pattern]] = [
    # OpenAI keys: sk-..., sk-proj-...
    # Must NOT match Anthropic (sk-ant-) or OpenRouter (sk-or-v1-) keys,
    # which have their own patterns further down.
    ("OPENAI_KEY", re.compile(
        r"\bsk-(?!ant-|or-v1-)(?:proj-)?[A-Za-z0-9_\-]{20,}\b"
    )),

    # Anthropic keys: sk-ant-...
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),

    # Google API keys: AIza...
    ("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),

    # OpenRouter: sk-or-v1-...
    ("OPENROUTER_KEY", re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{20,}\b")),

    # Groq keys: gsk_...
    ("GROQ_KEY", re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b")),

    # Supabase service-role JWT
    ("SUPABASE_KEY", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),

    # ScraperAPI keys
    ("SCRAPERAPI_KEY", re.compile(r"\bscraperapi[_-]?key\s*[:=]\s*['\"]?([A-Za-z0-9]{20,})", re.IGNORECASE)),

    # Bearer tokens
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE)),

    # Generic "api_key=..." style
    ("GENERIC_API_KEY", re.compile(
        r"\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})",
        re.IGNORECASE,
    )),

    # Password assignments in connection strings
    ("PASSWORD", re.compile(
        r"\b(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{6,})",
        re.IGNORECASE,
    )),
]

_PEM_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

_HIGH_ENTROPY_HEX_RE = re.compile(r"\b[a-fA-F0-9]{32,}\b")
_B64_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{40,}\b")


def _looks_like_base64_secret(s: str) -> bool:
    """
    Heuristic to avoid redacting URL path segments. Real base64 secrets
    virtually always contain mixed case AND at least one digit. A URL
    slug like "online-pharmacy/product/panadol-tablet-500mg" is lowercase
    only, so it fails this check.
    """
    if len(s) < 40:
        return False
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    return has_upper and has_lower and has_digit


@dataclass
class ScrubResult:
    original: str
    scrubbed: str
    redactions: list[str]

    @property
    def was_modified(self) -> bool:
        return self.original != self.scrubbed


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def scrub(text: str, extra_secrets: Iterable[str] = ()) -> ScrubResult:
    """Scrub secrets from `text`."""
    if not text:
        return ScrubResult(original=text or "", scrubbed=text or "", redactions=[])

    redactions: list[str] = []
    out = text

    # --- explicit literal secrets (highest priority) ---
    for secret in extra_secrets:
        if secret and len(secret) >= 8 and secret in out:
            out = out.replace(secret, "<EXPLICIT_SECRET_REDACTED>")
            redactions.append("EXPLICIT_SECRET")

    # --- named patterns ---
    for name, rx in _NAMED_PATTERNS:
        if rx.search(out):
            if name in ("GENERIC_API_KEY", "SCRAPERAPI_KEY", "PASSWORD"):
                def _replace_value(m: re.Match) -> str:
                    prefix = m.group(1)
                    return f"{prefix}=<{name}_REDACTED>"
                out = rx.sub(_replace_value, out)
            else:
                out = rx.sub(f"<{name}_REDACTED>", out)
            redactions.append(name)

    # --- PEM private keys ---
    if _PEM_KEY_RE.search(out):
        out = _PEM_KEY_RE.sub("<PRIVATE_KEY_REDACTED>", out)
        redactions.append("PRIVATE_KEY")

    # --- high-entropy hex / base64 ---
    if _HIGH_ENTROPY_HEX_RE.search(out):
        out = _HIGH_ENTROPY_HEX_RE.sub("<HEX_SECRET_REDACTED>", out)
        redactions.append("HEX_SECRET")

    def _b64_sub(m: re.Match) -> str:
        val = m.group(0)
        return "<B64_SECRET_REDACTED>" if _looks_like_base64_secret(val) else val

    if _B64_CANDIDATE_RE.search(out):
        new_out = _B64_CANDIDATE_RE.sub(_b64_sub, out)
        if new_out != out:
            redactions.append("B64_SECRET")
        out = new_out

    return ScrubResult(original=text, scrubbed=out, redactions=redactions)


def contains_secret(text: str, extra_secrets: Iterable[str] = ()) -> bool:
    """Cheap check — True if `scrub()` would change the text."""
    return scrub(text, extra_secrets=extra_secrets).was_modified


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("OpenAI error: sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "OPENAI_KEY"),
        ("Key is AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "GOOGLE_KEY"),
        ("GROQ API returned 401 for gsk_abcdefghijklmnopqrstuvwxyz0123456789ABCD", "GROQ_KEY"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "BEARER_TOKEN"),
        ("api_key=abcdEFGH1234567890abcdEFGH1234567890", "GENERIC_API_KEY"),
        ("password=hunter2hunter2", "PASSWORD"),
    ]
    for text, expect in cases:
        r = scrub(text)
        assert r.was_modified, f"not scrubbed: {text!r}"
        assert expect in r.redactions, f"expected {expect} in {r.redactions} for {text!r}"

    # Anthropic key must NOT be caught by OPENAI_KEY
    r = scrub("anthropic: sk-ant-abcdefghijklmnop0123456789")
    assert "ANTHROPIC_KEY" in r.redactions, r.redactions
    assert "OPENAI_KEY" not in r.redactions, r.redactions

    # OpenRouter key must NOT be caught by OPENAI_KEY
    r = scrub("openrouter: sk-or-v1-abcdefghijklmnopqrstuvwxyz0123456789")
    assert "OPENROUTER_KEY" in r.redactions, r.redactions
    assert "OPENAI_KEY" not in r.redactions, r.redactions

    # PEM
    pem = """-----BEGIN RSA PRIVATE KEY-----
    MIIEowIBAAKCAQEA...
    -----END RSA PRIVATE KEY-----"""
    r = scrub(pem)
    assert "PRIVATE_KEY" in r.redactions
    assert "MIIEowIBAAKCAQEA" not in r.scrubbed

    # Explicit literal
    r = scrub("call with SECRET_TOKEN_ABC123", extra_secrets=["SECRET_TOKEN_ABC123"])
    assert "EXPLICIT_SECRET" in r.redactions

    # Clean text untouched
    r = scrub("the quick brown fox jumps over the lazy dog")
    assert not r.was_modified
    assert r.redactions == []

    # Empty
    r = scrub("")
    assert r.scrubbed == ""

    # contains_secret
    assert contains_secret("api_key=abcdefghijklmnop12345678")
    assert not contains_secret("hello world")

    print("Secret scrubbing OK.")