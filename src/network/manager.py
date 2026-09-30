"""
NetworkManager — tiered provider orchestration (spec §15.5D).

Walks a tiered chain of fetch providers, stopping at the first one
that returns usable (non-blocked) HTML:

    Tier 1 (local, $0)
        1a  seleniumbase       — CDP mode, maximum stealth
        1b  playwright         — fast baseline via the injected scraper
        1c  byparr             — local Turnstile solver

    Tier 2 (paid, budget-limited)
        2a  scraperapi         — SCRAPER_API_KEY

    Tier 3 (fallback free tiers)
        3a  scrapingant        — 10K credits/month
        3b  webscrapingapi     — 1K calls/month
        3c  zenrows            — 5K credits (~200 protected)
        3d  zenscrape          — 1K credits/month
        3e  apify              — $5 compute credits/month

Fallback is PER REQUEST. Each call starts at Tier 1 again. There is no
global circuit-breaker state — if Docker was down 5 minutes ago and is
back now, the very next fetch tries Tier 1.

Backward compatibility:
    NetworkManager(scraper, scraperapi)     ← the old two-arg form
is interpreted as:
    - `scraper`     → the Playwright provider (Tier 1b)
    - `scraperapi`  → the ScraperAPI provider (Tier 2a)

Every other provider is an optional keyword argument. Both call forms
work; nothing in the existing codebase has to change.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from src.network.block_codes import (
    classify_provider_error, should_fall_through,
)
from src.network.bot_shell import looks_like_bot_shell
from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, ProviderName,
    Tier, Transport, looks_blocked,
)

# Providers are imported lazily inside build_production_manager() so
# this module stays import-safe when optional dependencies are missing.

logger = logging.getLogger("network_manager")


# ---------------------------------------------------------------------------
# Tier ordering
# ---------------------------------------------------------------------------

# Tier 0 — fast HTTP with browser TLS fingerprint (curl_cffi)
_TIER_0_ORDER = ("curl_cffi",)

# Tier 1 — local browsers, escalating stealth
_TIER_1_ORDER = ("seleniumbase", "camoufox", "playwright", "byparr")

_TIER_2_ORDER = ("scraperapi",)

_TIER_3_ORDER = (
    "scrapingant", "webscrapingapi", "zenrows", "zenscrape", "apify",
)

_TIER_BY_NAME: dict[str, str] = {}
for _n in _TIER_0_ORDER:
    _TIER_BY_NAME[_n] = Tier.TIER_0.value
for _n in _TIER_1_ORDER:
    _TIER_BY_NAME[_n] = Tier.TIER_1.value
for _n in _TIER_2_ORDER:
    _TIER_BY_NAME[_n] = Tier.TIER_2.value
for _n in _TIER_3_ORDER:
    _TIER_BY_NAME[_n] = Tier.TIER_3.value

_TIER_RANK = {
    Tier.TIER_0.value: 0,
    Tier.TIER_1.value: 1,
    Tier.TIER_2.value: 2,
    Tier.TIER_3.value: 3,
}

# Providers that must not receive render/premium defaults in `_route_for`.
_NO_RENDER_PROVIDERS = frozenset({"playwright", "curl_cffi"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_configured(provider: Any) -> bool:
    """
    True if the provider claims it can serve requests. Providers that
    don't implement is_configured() (e.g. a bare scraper wrapper) are
    assumed usable.
    """
    fn = getattr(provider, "is_configured", None)
    if not callable(fn):
        return True
    try:
        return bool(fn())
    except Exception:
        return False


def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


class _PlaywrightAdapter:
    """
    Wraps a Playwright-style scraper (`await fetch_html(url, timeout_ms)`)
    so it satisfies the `await fetch(url, route)` provider protocol, and
    applies the block / bot-shell heuristics that NetworkManager used to
    apply inline.
    """
    def __init__(self, scraper: Any):
        self.scraper = scraper

    def is_configured(self) -> bool:
        return True

    async def fetch(self, url: str, route: NetworkRoute) -> FetchResult:
        timeout_ms = int((route.timeout_seconds or 60) * 1000)
        html = await self.scraper.fetch_html(url, timeout=timeout_ms)
        html = html or ""

        blocked, reason = looks_blocked(html)
        if not blocked:
            shell, shell_reason = looks_like_bot_shell(
                html, query_terms=route.query_terms or None,
            )
            if shell:
                blocked = True
                reason = f"bot shell: {shell_reason}"

        return FetchResult(
            url=url,
            html=html,
            status_code=200 if html else 0,
            provider=ProviderName.PLAYWRIGHT.value,
            transport=Transport.BROWSER.value,
            bytes_received=len(html),
            blocked=blocked,
            block_reason=reason,
        )


# ---------------------------------------------------------------------------
# NetworkManager
# ---------------------------------------------------------------------------

class NetworkManager:
    """
    Tiered fetch orchestrator. See module docstring for the chain.
    """

    def __init__(
        self,
        scraper: Any = None,
        scraperapi: Any = None,
        *,
        curl_cffi: Any = None,
        seleniumbase: Any = None,
        camoufox: Any = None,
        byparr: Any = None,
        scrapingant: Any = None,
        webscrapingapi: Any = None,
        zenrows: Any = None,
        zenscrape: Any = None,
        apify: Any = None,
        hitl: Any = None,
        jitter: Optional[Jitter] = None,
        default_route: Optional[NetworkRoute] = None,
    ):
        self._providers: dict[str, Any] = {}
        if curl_cffi is not None:
            self._providers["curl_cffi"] = curl_cffi
        if seleniumbase is not None:
            self._providers["seleniumbase"] = seleniumbase
        if camoufox is not None:
            self._providers["camoufox"] = camoufox
        if scraper is not None:
            # Wrap the raw scraper in the adapter so it speaks the
            # provider protocol. Already-wrapped providers (things with
            # a .fetch method) pass through unchanged.
            if callable(getattr(scraper, "fetch", None)):
                self._providers["playwright"] = scraper
            else:
                self._providers["playwright"] = _PlaywrightAdapter(scraper)
        if byparr is not None:
            self._providers["byparr"] = byparr
        if scraperapi is not None:
            self._providers["scraperapi"] = scraperapi
        if scrapingant is not None:
            self._providers["scrapingant"] = scrapingant
        if webscrapingapi is not None:
            self._providers["webscrapingapi"] = webscrapingapi
        if zenrows is not None:
            self._providers["zenrows"] = zenrows
        if zenscrape is not None:
            self._providers["zenscrape"] = zenscrape
        if apify is not None:
            self._providers["apify"] = apify
        if hitl is not None:
            self._providers["hitl"] = hitl

        self._jitter = jitter
        self._default_route = default_route or NetworkRoute()

        # --- provider credit meter (§41.4) ---
        # Accumulated across every successful provider fetch in this
        # manager instance. Playwright/direct fetches report zero credits,
        # so only paid providers (ScraperAPI, ScrapingAnt, ...) contribute.
        self.total_provider_credits: int = 0
        self.provider_credits_by_provider: dict[str, int] = {}
        self.provider_call_count: int = 0

    # ------------------------------------------------------------------
    # Chain building
    # ------------------------------------------------------------------
    def _build_chain(
        self, route: NetworkRoute,
    ) -> tuple[list[tuple[str, str, Any]], dict[str, str]]:
        """
        Returns (chain, skipped_reasons).

        `skipped_reasons[name]` records why a provider that exists in
        `self._providers` was excluded from the chain — currently only
        "provider missing key" when is_configured() returned False.
        Used downstream for backward-compat metadata.
        """
        policy = (route.tier_policy or "auto").lower()
        allowed = [str(t).lower() for t in (route.allowed_tiers or [])]
        max_tier = (route.max_tier or Tier.TIER_3.value).lower()
        max_rank = _TIER_RANK.get(max_tier, 3)

        # If the caller wants to skip all automated providers and jump
        # straight to HITL, honor that. This is a legitimate shortcut
        # when the user already knows a target needs a human.
        skip_automated = bool((route.options or {}).get("skip_automated", False))

        _LOCAL_TIERS = {Tier.TIER_0.value, Tier.TIER_1.value}

        def _passes(tier_value: str) -> bool:
            if allowed:
                return tier_value in allowed
            if policy == "local_only":
                return tier_value in _LOCAL_TIERS
            if policy == "no_local":
                return tier_value not in _LOCAL_TIERS
            return True

        chain: list[tuple[str, str, Any]] = []
        skipped: dict[str, str] = {}

        # If skip_automated is set, jump straight past every automated
        # tier. HITL (appended below) will be the only chain entry.
        if not skip_automated:
            for tier_value, names in (
                (Tier.TIER_0.value, _TIER_0_ORDER),
                (Tier.TIER_1.value, _TIER_1_ORDER),
                (Tier.TIER_2.value, _TIER_2_ORDER),
                (Tier.TIER_3.value, _TIER_3_ORDER),
            ):
                if not _passes(tier_value):
                    continue
                if _TIER_RANK.get(tier_value, 99) > max_rank:
                    continue
                for name in names:
                    provider = self._providers.get(name)
                    if provider is None:
                        continue
                    if not _provider_configured(provider):
                        skipped[name] = "provider missing key"
                        continue
                    chain.append((tier_value, name, provider))

        # HITL is a special final tier — only included if the caller
        # explicitly opted in via route.options['hitl'] == True.
        # It sits after all automated providers because it's slow and
        # requires a human.
        hitl_provider = self._providers.get("hitl")
        hitl_enabled = bool((route.options or {}).get("hitl", False))
        if hitl_provider is not None and hitl_enabled:
            chain.append(("hitl", "hitl", hitl_provider))

        return chain, skipped

    # ------------------------------------------------------------------
    # Per-provider route shaping
    # ------------------------------------------------------------------
    def _route_for(
        self,
        base: NetworkRoute,
        provider_name: str,
        last_failure_reason: str,
    ) -> NetworkRoute:
        route = copy.copy(base)
        route.options = dict(base.options or {})
        route.session_cookies = dict(base.session_cookies or {})
        route.query_terms = list(base.query_terms or [])

        # Tier 2/3 providers all default to render=True. Whether the
        # previous tier said "bot shell" decides if we ask for premium
        # proxies (residential tier), same heuristic the old manager
        # used for ScraperAPI.
        if provider_name not in _NO_RENDER_PROVIDERS:
            route.options.setdefault("render", True)
            if last_failure_reason and last_failure_reason.startswith("bot shell"):
                route.options["premium"] = True
            else:
                route.options.setdefault("premium", False)

        return route

    # ------------------------------------------------------------------
    # Usage meter (§41.4)
    # ------------------------------------------------------------------
    def reset_usage(self) -> None:
        """Zero the provider-credit meter. Called by PipelineRunner per run."""
        self.total_provider_credits = 0
        self.provider_credits_by_provider = {}
        self.provider_call_count = 0

    def _record_provider_usage(self, result: FetchResult) -> None:
        """
        Accumulate credits from a successful fetch. Fetches that report
        zero credits (Playwright, direct HTTP, or an unconfigured
        provider) still bump `provider_call_count` but contribute no
        credits to the total.
        """
        self.provider_call_count += 1
        credits = int(getattr(result, "credits_used", 0) or 0)
        if credits <= 0:
            return
        self.total_provider_credits += credits
        provider = (result.provider or "unknown").lower()
        self.provider_credits_by_provider[provider] = (
            self.provider_credits_by_provider.get(provider, 0) + credits
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        timeout: Optional[int] = None,
        query_terms: Optional[list[str]] = None,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        """
        Fetch a URL through the tier chain. Returns a FetchResult whose
        `tier_attempts` list records every provider that was tried.

        A run that exhausts every tier returns a FetchResult with
        `blocked=True` and `tier_used="none"`.
        """
        route = route or self._default_route

        # Backward compat: timeout is interpreted as SECONDS at this
        # level (the old manager was inconsistent — it passed the same
        # value as both ms and seconds; we standardize on seconds).
        if timeout is not None:
            route = copy.copy(route)
            route.options = dict(route.options or {})
            route.timeout_seconds = int(timeout)

        # Backward compat: query_terms come in as a call arg, get
        # smuggled onto the route so the Playwright adapter can use them.
        if query_terms:
            route = copy.copy(route)
            route.options = dict(route.options or {})
            route.query_terms = list(query_terms)

        chain, skipped_reasons = self._build_chain(route)

        if not chain:
            return FetchResult(
                url=url,
                provider="",
                tier_used=Tier.NONE.value,
                blocked=True,
                block_reason="no providers configured",
            )

        # Optional jitter before the first attempt
        if route.jitter and self._jitter is not None:
            try:
                await self._jitter.sleep_for(_domain_of(url))
            except Exception as e:
                logger.debug(f"jitter skipped: {type(e).__name__}: {e}")

        attempts: list[dict] = []
        last_result: Optional[FetchResult] = None
        last_failure_reason = ""
        last_provider_name = ""

        for tier_value, provider_name, provider in chain:
            provider_route = self._route_for(
                route, provider_name, last_failure_reason,
            )

            try:
                result = await provider.fetch(url, provider_route)
            except ProviderError as e:
                attempts.append({
                    "provider": provider_name,
                    "tier": tier_value,
                    "ok": False,
                    "error": str(e),
                })
                last_failure_reason = str(e)
                last_provider_name = provider_name
                continue
            except Exception as e:
                attempts.append({
                    "provider": provider_name,
                    "tier": tier_value,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })
                last_failure_reason = f"{type(e).__name__}: {e}"
                last_provider_name = provider_name
                continue

            if result is None:
                attempts.append({
                    "provider": provider_name,
                    "tier": tier_value,
                    "ok": False,
                    "error": "returned None",
                })
                last_failure_reason = f"{provider_name} returned None"
                last_provider_name = provider_name
                continue

            if result.blocked:
                attempts.append({
                    "provider": provider_name,
                    "tier": tier_value,
                    "ok": False,
                    "blocked": True,
                    "block_reason": result.block_reason,
                })
                last_failure_reason = result.block_reason or "blocked"
                last_provider_name = provider_name
                last_result = result
                continue

            # ----- Success -----
            result.tier_used = tier_value
            result.tier_attempts = attempts + [{
                "provider": provider_name,
                "tier": tier_value,
                "ok": True,
            }]
            self._attach_backward_compat_metadata(
                result, skipped_reasons, route,
            )
            logger.info(
                f"NetworkManager: {url} served by {provider_name} "
                f"({tier_value}) after {len(attempts)} failed attempt(s)"
            )
            self._record_provider_usage(result)
            return result
            return result

        # ----- Every provider failed -----
        logger.warning(
            f"NetworkManager: all tiers exhausted for {url} "
            f"({len(attempts)} attempt(s))"
        )
        if last_result is not None:
            last_result.tier_used = Tier.NONE.value
            last_result.tier_attempts = attempts
            last_result.blocked = True
            last_result.block_reason = (
                last_result.block_reason
                or last_failure_reason
                or "all tiers exhausted"
            )
            self._attach_backward_compat_metadata(
                last_result, skipped_reasons, route,
            )
            return last_result

        return FetchResult(
            url=url,
            provider="",
            tier_used=Tier.NONE.value,
            tier_attempts=attempts,
            blocked=True,
            block_reason=last_failure_reason or "all tiers exhausted",
        )

    # ------------------------------------------------------------------
    # Backward-compat metadata
    #
    # The pre-tier manager populated a handful of metadata keys that
    # existing tests and call sites still read. We keep emitting them
    # for the two-provider (playwright → scraperapi) chain so nothing
    # breaks; new code should read `tier_attempts` instead.
    # ------------------------------------------------------------------
    @staticmethod
    def _attach_backward_compat_metadata(
        result: FetchResult,
        skipped_reasons: dict[str, str],
        route: NetworkRoute,
    ) -> None:
        """
        Emit the pre-tier metadata keys that older call sites and tests
        still read. New code should prefer `result.tier_attempts`.
        """
        meta = result.metadata
        attempts = result.tier_attempts or []
        tried = [a.get("provider") for a in attempts]

        # Playwright-side diagnostics
        pw_attempt = next(
            (a for a in attempts if a.get("provider") == "playwright"),
            None,
        )
        if pw_attempt is not None:
            if pw_attempt.get("block_reason"):
                meta.setdefault(
                    "playwright_block_reason", pw_attempt["block_reason"],
                )
            if pw_attempt.get("error"):
                meta.setdefault(
                    "playwright_error", pw_attempt["error"],
                )

        # "fallback_skipped": Playwright was tried but no provider
        # followed. Distinguish "provider existed but unconfigured"
        # from "no provider object at all".
        if pw_attempt is not None and len(attempts) == 1:
            if skipped_reasons:
                # Any is fine — we only track one reason today.
                meta.setdefault(
                    "fallback_skipped",
                    next(iter(skipped_reasons.values())),
                )
            else:
                meta.setdefault("fallback_skipped", "no provider configured")

        # If a non-Playwright provider was the final attempt
        if (
            len(attempts) >= 2
            and tried[-1] is not None
            and tried[-1] != "playwright"
        ):
            last = attempts[-1]
            meta.setdefault("fallback_from", "playwright")

            if last.get("ok"):
                pass   # provider succeeded — nothing more to add
            elif last.get("blocked"):
                meta.setdefault("fallback_failed", "provider also blocked")
            elif last.get("error"):
                # Provider raised; the result we're returning is the
                # earlier provider's (usually Playwright's).
                if result.provider == "playwright":
                    meta.setdefault("fallback_error", last["error"])

        # Premium flag used on the last attempt
        if "premium" in (route.options or {}):
            meta.setdefault("premium_used", bool(route.options["premium"]))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Fakes ---------------------------------------------------------------

    class FakeScraper:
        def __init__(self, html="", raises=None):
            self.html = html
            self.raises = raises
            self.calls = 0
        async def fetch_html(self, url, timeout=None):
            self.calls += 1
            if self.raises:
                raise self.raises
            return self.html

    class FakeProvider:
        def __init__(self, result=None, raises=None, configured=True):
            self.result = result
            self.raises = raises
            self.configured = configured
            self.calls = 0
            self.last_route = None
        def is_configured(self):
            return self.configured
        async def fetch(self, url, route):
            self.calls += 1
            self.last_route = route
            if self.raises:
                raise self.raises
            return self.result

    def _good_html(n=2500):
        return (
            "<html><head><title>Test</title></head>"
            "<body><h1>Test Content</h1><p>real page</p>"
            + "x" * n
            + "</body></html>"
        )

    def _good_result():
        return FetchResult(
            url="https://x", html=_good_html(),
            status_code=200, provider="scraperapi",
            transport="provider_api", bytes_received=2500, credits_used=5,
        )

    async def run():
        # 1. Healthy Playwright — no fallback
        scraper = FakeScraper(_good_html())
        provider = FakeProvider()
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "playwright"
        assert not r.blocked
        assert r.tier_used == "tier_1"
        assert provider.calls == 0
        assert r.tier_attempts[-1]["provider"] == "playwright"
        assert r.tier_attempts[-1]["ok"] is True

        # 2. Blocked Playwright — fallback to ScraperAPI
        scraper = FakeScraper("Please enable JavaScript to continue")
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "scraperapi"
        assert r.tier_used == "tier_2"
        assert r.credits_used == 5
        assert r.metadata["fallback_from"] == "playwright"
        assert provider.calls == 1
        # Two attempts recorded
        assert len(r.tier_attempts) == 2
        assert r.tier_attempts[0]["provider"] == "playwright"
        assert r.tier_attempts[0]["ok"] is False
        assert r.tier_attempts[1]["provider"] == "scraperapi"
        assert r.tier_attempts[1]["ok"] is True

        # 3. Provider not configured
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(configured=False)
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "playwright"
        assert r.blocked is True
        assert provider.calls == 0

        # 4. No provider at all
        m = NetworkManager(FakeScraper("Please enable JavaScript"), None)
        r = await m.fetch("https://x")
        assert r.provider == "playwright"
        assert r.blocked is True

        # 5. Provider raises → Playwright result comes back with metadata
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(raises=RuntimeError("scraperapi down"))
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.blocked is True
        assert len(r.tier_attempts) == 2
        assert r.tier_attempts[-1]["ok"] is False
        assert "scraperapi down" in r.tier_attempts[-1]["error"]

        # 6. Three-tier chain — SeleniumBase fails, Playwright fails,
        #    ScraperAPI succeeds.
        class FakeSeleniumBase:
            def __init__(self): self.calls = 0
            def is_configured(self): return True
            async def fetch(self, url, route):
                self.calls += 1
                return FetchResult(
                    url=url, html="", status_code=0,
                    provider="seleniumbase_cdp",
                    blocked=True, block_reason="chrome crashed",
                )

        sb = FakeSeleniumBase()
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider, seleniumbase=sb)
        r = await m.fetch("https://x")
        assert r.provider == "scraperapi"
        assert r.tier_used == "tier_2"
        assert sb.calls == 1
        assert len(r.tier_attempts) == 3
        assert [a["provider"] for a in r.tier_attempts] == [
            "seleniumbase", "playwright", "scraperapi",
        ]

        # 7. Tier 3 provider used when Tier 1 + 2 fail
        class FakeScrapingAnt:
            def __init__(self): self.calls = 0
            def is_configured(self): return True
            async def fetch(self, url, route):
                self.calls += 1
                return FetchResult(
                    url=url, html=_good_html(), status_code=200,
                    provider="scrapingant",
                    transport="provider_api", bytes_received=2500,
                )

        sa = FakeScrapingAnt()
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(configured=False)   # ScraperAPI unavailable
        m = NetworkManager(scraper, provider, scrapingant=sa)
        r = await m.fetch("https://x")
        assert r.provider == "scrapingant"
        assert r.tier_used == "tier_3"
        assert sa.calls == 1

        # 8. All tiers fail → blocked=True, tier_used="none"
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(raises=RuntimeError("scraperapi down"))
        sa = FakeScrapingAnt()
        # Also make scrapingant fail
        async def _fail_fetch(url, route):
            raise RuntimeError("scrapingant down")
        sa.fetch = _fail_fetch
        m = NetworkManager(scraper, provider, scrapingant=sa)
        r = await m.fetch("https://x")
        assert r.tier_used == "none"
        assert r.blocked is True
        assert len(r.tier_attempts) == 3

        # 9. tier_policy="local_only" restricts to Tier 1
        scraper = FakeScraper(_good_html())
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider)
        r = await m.fetch(
            "https://x",
            route=NetworkRoute(tier_policy="local_only"),
        )
        assert r.provider == "playwright"
        assert provider.calls == 0

        # 10. max_tier="tier_1" — no fallback
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider)
        r = await m.fetch(
            "https://x",
            route=NetworkRoute(max_tier="tier_1"),
        )
        assert provider.calls == 0
        assert r.blocked is True

        # 11. Bot-shell detection triggers premium on the next tier
        shell_html = (
            "<html><head><title></title></head><body>"
            + "<script>x</script>" * 200
            + "</body></html>"
        )
        scraper = FakeScraper(shell_html)
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x", query_terms=["panadol"])
        assert provider.calls == 1
        assert provider.last_route.options.get("premium") is True

        # 12. Simple small-page block → premium=False
        scraper = FakeScraper("Please enable JavaScript to continue")
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider)
        await m.fetch("https://x")
        assert provider.last_route.options.get("premium") is False

        # 13. Cookie injection is passed through
        scraper = FakeScraper(_good_html())
        provider = FakeProvider(result=_good_result())
        m = NetworkManager(scraper, provider)
        route = NetworkRoute(session_cookies={"li_at": "xyz"})
        await m.fetch("https://x", route=route)
        # Playwright returned good HTML, so no provider call. But the
        # route was still passed. Verify with a fresh fail case:
        scraper = FakeScraper("Please enable JavaScript")
        captured = {}
        class CookieCaptureProvider:
            def is_configured(self): return True
            async def fetch(self, url, route):
                captured["cookies"] = dict(route.session_cookies)
                return _good_result()
        m = NetworkManager(scraper, CookieCaptureProvider())
        await m.fetch(
            "https://x",
            route=NetworkRoute(session_cookies={"li_at": "xyz"}),
        )
        assert captured["cookies"] == {"li_at": "xyz"}

        # 14. Jitter is called (using a fake)
        calls: list[str] = []
        class FakeJitter:
            async def sleep_for(self, domain):
                calls.append(domain)
                return 0.01
        scraper = FakeScraper(_good_html())
        m = NetworkManager(scraper, jitter=FakeJitter())
        await m.fetch("https://example.com/page")
        assert calls == ["example.com"]

        # 15. Jitter skipped when route.jitter=False
        calls.clear()
        scraper = FakeScraper(_good_html())
        m = NetworkManager(scraper, jitter=FakeJitter())
        await m.fetch(
            "https://example.com/page",
            route=NetworkRoute(jitter=False),
        )
        assert calls == []

        # 16. No providers at all → clean failure
        m = NetworkManager()
        r = await m.fetch("https://x")
        assert r.blocked is True
        assert r.tier_used == "none"
        assert "no providers configured" in r.block_reason

        print("NetworkManager OK.")

    asyncio.run(run())



# ---------------------------------------------------------------------------
# Production factory
# ---------------------------------------------------------------------------

def build_production_manager(
    scraper: Any = None,
    *,
    enable_jitter: bool = True,
    enable_curl_cffi: bool = True,
    enable_seleniumbase: bool = True,
    enable_camoufox: bool = True,
) -> "NetworkManager":
    """
    Construct a NetworkManager with every provider it can find in the
    environment. Missing API keys silently disable that provider — no
    exception, no warning, the manager just runs with what it has.

    `scraper` — an existing Playwright-style scraper. If None, the
    caller is expected to pass one later (this factory doesn't create
    ScraperEngine itself to avoid circular imports with scraper.engine).

    Each provider is imported lazily so a failure to import one (e.g.
    SeleniumBase not installed) doesn't break the whole chain.

    Usage from production code:

        from src.scraper.engine import ScraperEngine
        from src.network.manager import build_production_manager

        manager = build_production_manager(ScraperEngine())
        result = await manager.fetch(url)
    """
    # --- Tier 0: curl_cffi (fast HTTP with browser TLS fingerprint) ---
    curl_cffi_provider = None
    if enable_curl_cffi:
        try:
            from src.network.curl_cffi_driver import CurlCffiProvider
            ccp = CurlCffiProvider()
            if ccp.is_configured():
                curl_cffi_provider = ccp
        except Exception:
            curl_cffi_provider = None

    # --- Tier 1a: SeleniumBase (optional — needs the package installed) ---
    seleniumbase_provider = None
    if enable_seleniumbase:
        try:
            from src.network.seleniumbase_cdp import SeleniumBaseCDPProvider
            sb = SeleniumBaseCDPProvider()
            if sb.is_configured():
                seleniumbase_provider = sb
        except Exception:
            seleniumbase_provider = None

    # --- Tier 1b: Camoufox (Firefox-based anti-detection) ---
    camoufox_provider = None
    if enable_camoufox:
        try:
            from src.network.camoufox import CamoufoxProvider
            cfp = CamoufoxProvider()
            if cfp.is_configured():
                camoufox_provider = cfp
        except Exception:
            camoufox_provider = None

    # --- Tier 1c: Byparr (URL only — no key) ---
    byparr_provider = None
    try:
        from src.network.byparr import ByparrProvider
        bp = ByparrProvider()
        # Byparr doesn't probe its own reachability here — a failed
        # connection during fetch() will be handled by the tier chain.
        if bp.is_configured():
            byparr_provider = bp
    except Exception:
        byparr_provider = None

    # --- Tier 2: ScraperAPI ---
    scraperapi_provider = None
    try:
        from src.network.scraperapi import ScraperAPIProvider
        sp = ScraperAPIProvider()
        if sp.is_configured():
            scraperapi_provider = sp
    except Exception:
        scraperapi_provider = None

    # --- Tier 3a: ScrapingAnt ---
    scrapingant_provider = None
    try:
        from src.network.scrapingant import ScrapingAntProvider
        sa = ScrapingAntProvider()
        if sa.is_configured():
            scrapingant_provider = sa
    except Exception:
        scrapingant_provider = None

    # --- Tier 3b: WebScrapingAPI ---
    webscrapingapi_provider = None
    try:
        from src.network.webscrapingapi import WebScrapingAPIProvider
        wsa = WebScrapingAPIProvider()
        if wsa.is_configured():
            webscrapingapi_provider = wsa
    except Exception:
        webscrapingapi_provider = None

    # --- Tier 3c: ZenRows ---
    zenrows_provider = None
    try:
        from src.network.zenrows import ZenRowsProvider
        zr = ZenRowsProvider()
        if zr.is_configured():
            zenrows_provider = zr
    except Exception:
        zenrows_provider = None

    # --- Tier 3d: Zenscrape ---
    zenscrape_provider = None
    try:
        from src.network.zenscrape import ZenscrapeProvider
        zs = ZenscrapeProvider()
        if zs.is_configured():
            zenscrape_provider = zs
    except Exception:
        zenscrape_provider = None

    # --- Tier 3e: Apify ---
    apify_provider = None
    try:
        from src.network.apify import ApifyProvider
        ap = ApifyProvider()
        if ap.is_configured():
            apify_provider = ap
    except Exception:
        apify_provider = None

    # --- Tier 4: Human-in-the-Loop (opt-in only) ---
    hitl_provider = None
    try:
        from src.network.hitl_solver import HITLSolver
        hs = HITLSolver()
        if hs.is_configured():
            hitl_provider = hs
    except Exception:
        hitl_provider = None

    # --- Jitter (on by default for production) ---
    jitter = None
    if enable_jitter:
        try:
            from src.network.jitter import Jitter
            jitter = Jitter()
        except Exception:
            jitter = None

    return NetworkManager(
        scraper=scraper,
        scraperapi=scraperapi_provider,
        curl_cffi=curl_cffi_provider,
        seleniumbase=seleniumbase_provider,
        camoufox=camoufox_provider,
        byparr=byparr_provider,
        scrapingant=scrapingant_provider,
        webscrapingapi=webscrapingapi_provider,
        zenrows=zenrows_provider,
        zenscrape=zenscrape_provider,
        apify=apify_provider,
        hitl=hitl_provider,
        jitter=jitter,
    )