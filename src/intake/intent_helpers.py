"""
Deterministic intent-inference helpers — spec §11.1 steps.

These take what the LLM already gave us (or the raw user prompt) and produce
the last few §11.1 pipeline steps without another LLM call:
    - navigation inference (pagination / detail pages / max pages)
    - output format inference (xlsx / csv / json / parquet)
    - compliance pre-screening (access-controlled resources → refuse)
    - PII-aware prompt assembly (redact for the LLM, restore afterwards)

Kept separate from command_parser so they're cheap to unit-test.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from src.intake.pii_redactor import redact, RedactionResult
from src.intake.language_detect import detect_language


# ---------------------------------------------------------------------------
# Navigation inference (§11.1 step 6)
# ---------------------------------------------------------------------------

# Phrases that imply "more than one page" work.
# NOTE: bare singular "page" is intentionally NOT a signal — it appears in
# prompts like "just this one page" that mean the opposite.
_MULTIPAGE_HINTS = re.compile(
    r"\b(all|every|each|paginate|pagination|next\s+page|"
    r"across\s+multiple\s+pages?|scroll|load\s+more|browse|pages)\b",
    re.IGNORECASE,
)

# Explicit single-page phrases override _MULTIPAGE_HINTS.
_SINGLE_PAGE_HINTS = re.compile(
    r"\b(single\s+page|one\s+page|just\s+(this|the)\s+page|this\s+one\s+page|"
    r"only\s+(this|one)\s+page|first\s+page\s+only|just\s+the\s+first\s+page)\b",
    re.IGNORECASE,
)

_DETAIL_PAGE_HINTS = re.compile(
    r"\b(detail|details|product\s+page|each\s+item|per\s+item|full\s+description|"
    r"specifications?|reviews?)\b",
    re.IGNORECASE,
)

# "up to N pages" / "N pages" / "pages=N"
_PAGE_NUMBER_RE = re.compile(
    r"\b(?:up\s+to\s+|first\s+|max\s+)?(\d{1,4})\s+pages?\b",
    re.IGNORECASE,
)


@dataclass
class NavigationHint:
    pagination: str        # "auto" | "none"
    detail_pages: str      # "required" | "optional" | "none"
    max_pages: int


def infer_navigation(prompt: str, llm_hints: Optional[dict] = None) -> NavigationHint:
    """
    Returns navigation defaults inferred from the prompt text. If the LLM
    provided navigation hints, they take precedence for any field they set.
    """
    llm_hints = llm_hints or {}

    text = prompt or ""
    # Negative hint wins — explicit single-page phrasing is unambiguous.
    if _SINGLE_PAGE_HINTS.search(text):
        pagination = "none"
    elif _MULTIPAGE_HINTS.search(text):
        pagination = "auto"
    else:
        pagination = "none"

    # LLM hints take precedence for any field they explicitly set.
    if llm_hints.get("pagination"):
        pagination = str(llm_hints["pagination"])

    if _DETAIL_PAGE_HINTS.search(text):
        detail_pages = "required"
    elif pagination == "auto":
        detail_pages = "optional"
    else:
        detail_pages = "none"

    if llm_hints.get("detail_pages"):
        detail_pages = str(llm_hints["detail_pages"])

    max_pages = 5
    m = _PAGE_NUMBER_RE.search(prompt or "")
    if m:
        max_pages = max(1, min(int(m.group(1)), 500))
    if llm_hints.get("max_pages"):
        try:
            max_pages = max(1, min(int(llm_hints["max_pages"]), 500))
        except (TypeError, ValueError):
            pass

    return NavigationHint(
        pagination=pagination,
        detail_pages=detail_pages,
        max_pages=max_pages,
    )


# ---------------------------------------------------------------------------
# Output format inference (§11.1 step 7)
# ---------------------------------------------------------------------------

_FORMAT_HINTS = {
    "parquet": re.compile(r"\bparquet\b", re.IGNORECASE),
    "json":    re.compile(r"\bjson\b|\bjsonl\b", re.IGNORECASE),
    "csv":     re.compile(r"\bcsv\b|\bcomma[-\s]?separated\b", re.IGNORECASE),
    "xlsx":    re.compile(r"\bxlsx\b|\bexcel\b|\bspreadsheet\b|\bworkbook\b", re.IGNORECASE),
}


def infer_output_format(prompt: str, default: str = "xlsx") -> str:
    """Return a valid Output.format value, preferring the most specific hint."""
    for fmt in ("parquet", "json", "csv", "xlsx"):
        if _FORMAT_HINTS[fmt].search(prompt or ""):
            return fmt
    return default


# ---------------------------------------------------------------------------
# Compliance pre-screening (§11.1 step 9)
# ---------------------------------------------------------------------------

# Phrases that indicate the user is asking us to access something behind
# access control — §15.7 and §60 say we refuse these.
_FORBIDDEN_PATTERNS = [
    (re.compile(r"\b(log\s*in|login|sign\s*in|signin)\b.*\b(as|with)\b.*\b(me|my|password)\b",
                re.IGNORECASE), "logging in with a user's credentials"),
    (re.compile(r"\bbypass\b.*\b(login|paywall|captcha|auth)\b", re.IGNORECASE),
     "bypassing access control"),
    (re.compile(r"\b(scrape|get|extract)\b.*\b(behind|through|past)\b.*\b(paywall|login|auth)\b",
                re.IGNORECASE), "accessing a resource behind a paywall or login"),
    (re.compile(r"\bcrack\b|\bsteal\b.*\b(credentials?|passwords?|cookies?)\b",
                re.IGNORECASE), "acquiring or using third-party credentials"),
    (re.compile(r"\bsolve\b.*\bcaptcha\b", re.IGNORECASE),
     "solving CAPTCHAs programmatically"),
]


@dataclass
class ComplianceHint:
    allowed: bool
    refusal_reason: str = ""
    user_authorization_declared: bool = False


def infer_compliance(prompt: str) -> ComplianceHint:
    """
    Returns a refusal if the prompt clearly asks for forbidden behavior.
    Otherwise marks the task as "user has not declared authorization" (the
    default) — the compliance gate in §60 will still run downstream.
    """
    text = prompt or ""
    for rx, reason in _FORBIDDEN_PATTERNS:
        if rx.search(text):
            return ComplianceHint(allowed=False, refusal_reason=reason)
    return ComplianceHint(allowed=True, user_authorization_declared=False)


# ---------------------------------------------------------------------------
# PII-safe prompt assembly (§11.1 step 2)
# ---------------------------------------------------------------------------

@dataclass
class SafePrompt:
    redacted: str
    redaction: RedactionResult

    def restore(self, text: str) -> str:
        return self.redaction.restore(text)


def make_safe_prompt(prompt: str) -> SafePrompt:
    """Redact PII from a user prompt before it goes to the LLM."""
    r = redact(prompt or "")
    return SafePrompt(redacted=r.redacted, redaction=r)


# ---------------------------------------------------------------------------
# Language detection pass-through (§11.1 step 1)
# ---------------------------------------------------------------------------

def infer_language(prompt: str) -> str:
    """
    Return the language family string for `constraints.language`.
    "latin" is treated as unknown-but-close-to-English for now; the
    downstream normalizer will still use local logic.
    """
    return detect_language(prompt or "").family


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Navigation
    assert infer_navigation("get all items across pages").pagination == "auto"
    assert infer_navigation("just this one page").pagination == "none"
    assert infer_navigation("first page only please").pagination == "none"
    assert infer_navigation("find laptop prices").pagination == "none"
    assert infer_navigation("get the first 3 pages of products").max_pages == 3
    assert infer_navigation("get the detail page for each item").detail_pages == "required"

    # Output
    assert infer_output_format("give me a csv") == "csv"
    assert infer_output_format("as parquet please") == "parquet"
    assert infer_output_format("nothing special") == "xlsx"

    # Compliance
    assert infer_compliance("scrape public prices").allowed
    assert not infer_compliance("bypass the login wall on example.com").allowed
    assert not infer_compliance("solve captcha on this site").allowed

    # PII
    sp = make_safe_prompt("email me at a@b.com")
    assert "<EMAIL_1>" in sp.redacted
    assert sp.restore(sp.redacted) == "email me at a@b.com"

    # Language
    assert infer_language("hello world") == "latin"
    assert infer_language("你好") == "cjk"

    print("Intent helpers OK.")