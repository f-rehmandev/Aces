"""
Network provider layer — spec §15.5.

The scraper engine must not know which transport it used. Everything that
can fetch a URL (direct HTTP, Playwright, ScraperAPI, future providers)
implements `NetworkProvider`. The `NetworkManager` picks a route.

Design:
    - `NetworkRoute` is a plain data object describing *how* to fetch.
    - `FetchResult` is what a provider returns — HTML + metadata.
    - `ProviderError` is the normalized exception type every provider raises.
    - Nothing in this module imports a specific provider.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


# ---------------------------------------------------------------------------
# Transport / provider identification
# ---------------------------------------------------------------------------

class Transport(str, Enum):
    DIRECT_HTTP = "direct_http"
    BROWSER = "browser"
    PROVIDER_API = "provider_api"
    LOCAL_SOLVER = "local_solver"   # Byparr / FlareSolverr-compatible


class ProviderName(str, Enum):
    DIRECT = "direct"
    CURL_CFFI = "curl_cffi"
    PLAYWRIGHT = "playwright"
    SELENIUMBASE_CDP = "seleniumbase_cdp"
    CAMOUFOX = "camoufox"
    BYPARR = "byparr"
    HITL = "hitl"
    SCRAPERAPI = "scraperapi"
    SCRAPINGANT = "scrapingant"
    WEBSCRAPINGAPI = "webscrapingapi"
    ZENROWS = "zenrows"
    ZENSCRAPE = "zenscrape"
    APIFY = "apify"




class Tier(str, Enum):
    """
    Which tier in the fallback chain a request was served by.

    Tier 0  — fast HTTP with browser TLS fingerprint (curl_cffi).
    Tier 1  — local browsers: SeleniumBase UC → Camoufox → Playwright → Byparr.
    Tier 2  — ScraperAPI (paid, budget-limited).
    Tier 3  — backup providers (ScrapingAnt → WebScrapingAPI → ZenRows →
              Zenscrape → Apify).

    "none" is used on the returned FetchResult when *every* tier failed.
    """
    TIER_0 = "tier_0"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    NONE = "none"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass
class NetworkRoute:
    """
    A decision about how to fetch. The caller passes this to a provider.
    Only `provider` matters for routing; the rest is metadata a provider
    may consult.
    """
    provider: str = ProviderName.DIRECT.value
    transport: str = Transport.DIRECT_HTTP.value
    # provider-specific options (rendering, country, premium, etc.)
    options: dict = field(default_factory=dict)
    # budget gate: caller decides whether this route is allowed
    estimated_credits: int = 0
    timeout_seconds: int = 60

    # --- tiered-fallback controls ---
    tier_policy: str = "auto"                # auto | local_only | no_local
    allowed_tiers: list[str] = field(default_factory=list)
    max_tier: str = Tier.TIER_3.value

    # --- session / pacing ---
    session_cookies: dict = field(default_factory=dict)
    jitter: bool = True
    # Query terms used by the bot-shell heuristic. Populated by callers
    # that know what the user is looking for (e.g. the extractor's field
    # names, or the search query).
    query_terms: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    html: str = ""
    status_code: int = 0
    provider: str = ""
    transport: str = ""
    bytes_received: int = 0
    credits_used: int = 0
    latency_ms: int = 0
    # set when the provider reports a *definite* block
    blocked: bool = False
    block_reason: str = ""
    # which tier served this request
    tier_used: str = Tier.NONE.value
    # per-attempt trail: one dict per provider tried, in order
    tier_attempts: list[dict] = field(default_factory=list)
    # arbitrary provider metadata (proxy IP, region, etc.)
    metadata: dict = field(default_factory=dict)

    def is_empty(self, min_chars: int = 500) -> bool:
        """True when the response is suspiciously small — usually a block."""
        return len(self.html or "") < min_chars

    @property
    def is_ok(self) -> bool:
        return self.status_code == 200 and not self.blocked and bool(self.html)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Normalized error every provider raises. Never leaks provider internals."""
    def __init__(self, provider: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.retryable = retryable


class ProviderNotConfigured(ProviderError):
    """Raised when the provider exists but its key/URL is missing."""
    def __init__(self, provider: str, missing: str):
        super().__init__(provider, f"{provider} not configured: missing {missing}")
        self.missing = missing


class BudgetExceeded(ProviderError):
    """Raised when a fetch would exceed the provider's remaining budget."""
    def __init__(self, provider: str, needed: int, available: int):
        super().__init__(
            provider,
            f"budget exceeded: need {needed} credits, have {available}",
            retryable=False,
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class NetworkProvider(Protocol):
    name: str

    async def fetch(self, url: str, route: NetworkRoute) -> FetchResult: ...

    def is_configured(self) -> bool: ...


# ---------------------------------------------------------------------------
# Block-signal detection (spec §16.1)
# ---------------------------------------------------------------------------

# Very short list — enough to catch the obvious cases without false positives
_BLOCK_MARKERS = (
    "enable javascript",
    "please enable javascript",
    "just a moment",
    "checking your browser",
    "attention required!",
    "cf-browser-verification",
    "access denied",
    "are you a robot",
)


def looks_blocked(html: str, min_chars: int = 500) -> tuple[bool, str]:
    """
    Cheap heuristic: does this HTML look like a block page?
    Returns (blocked, reason). `reason` is empty when not blocked.
    """
    if html is None:
        return True, "no html"
    stripped = html.strip()
    if len(stripped) < min_chars:
        lower = stripped.lower()
        for marker in _BLOCK_MARKERS:
            if marker in lower:
                return True, f"block marker: {marker!r}"
        return True, f"body too small ({len(stripped)} chars)"
    return False, ""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = NetworkRoute(provider="direct", transport="direct_http")
    assert r.provider == "direct"

    fr = FetchResult(url="https://x", html="<html>" + "a" * 1000 + "</html>",
                     status_code=200, provider="direct", transport="direct_http")
    assert fr.is_ok
    assert not fr.is_empty()

    small = FetchResult(url="https://x", html="hi", status_code=200)
    assert small.is_empty()

    blocked, reason = looks_blocked("Please enable JavaScript to continue")
    assert blocked and "javascript" in reason.lower()

    blocked, reason = looks_blocked("a" * 5000)
    assert not blocked and reason == ""

    blocked, reason = looks_blocked("")
    assert blocked and "small" in reason

    err = ProviderError("scraperapi", "bad request", retryable=False)
    assert err.provider == "scraperapi" and not err.retryable

    try:
        raise ProviderNotConfigured("scraperapi", "SCRAPERAPI_KEY")
    except ProviderNotConfigured as e:
        assert "SCRAPERAPI_KEY" in str(e)

    try:
        raise BudgetExceeded("scraperapi", needed=5, available=1)
    except BudgetExceeded as e:
        assert "5" in str(e) and "1" in str(e)

    print("Network types OK.")