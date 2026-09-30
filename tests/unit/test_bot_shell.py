"""
Network manager — spec §15.5D.

Chooses how to fetch a URL. Current policy:

    1. Try Playwright (through the supplied ScraperEngine).
    2. If the result looks blocked — either by size/marker heuristics
       (`looks_blocked`) or by product-shape absence (`looks_like_bot_shell`)
       — and ScraperAPI is configured, escalate.
    3. When we detected a *bot shell* specifically, escalate with
       `premium=True` (residential proxy + JS rendering, 25 credits).
       For smaller/ambiguous pages, plain rendering (5 credits) suffices.
    4. Return whichever result is usable.

Later milestones will make the route decision smarter (per-domain history,
health checks, cost estimation) without changing this interface.
"""

from __future__ import annotations
import logging
from typing import Optional

from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, Transport, looks_blocked,
)
from src.network.bot_shell import looks_like_bot_shell
from src.security.secrets import scrub


logger = logging.getLogger("network_manager")


class NetworkManager:
    """
    Coordinates Playwright and (optionally) ScraperAPI.

    `scraper` must expose `async fetch_html(url, timeout=None) -> str`.
    `scraperapi` may be None — the manager then behaves like a
    Playwright-only path.
    """

    def __init__(self, scraper, scraperapi=None):
        self.scraper = scraper
        self.scraperapi = scraperapi

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        timeout: Optional[int] = None,
        query_terms: Optional[list[str]] = None,
    ) -> FetchResult:
        """
        Try Playwright first; escalate to ScraperAPI when the result looks
        blocked OR the page shape says it's a bot shell.

        `query_terms` lets the bot-shell heuristic check whether the user's
        target keywords appear in the page's visible text. Passing None
        disables the query-term check.
        """
        # --- 1. Playwright attempt ---
        html = ""
        playwright_error = ""
        try:
            html = await self.scraper.fetch_html(url, timeout=timeout) or ""
        except Exception as e:
            # Scrub before logging or storing — provider keys and other
            # secrets can easily appear in network/SDK exception messages.
            raw_error = f"{type(e).__name__}: {e}"
            playwright_error = scrub(raw_error).scrubbed
            logger.warning(f"Playwright fetch failed on {url}: {playwright_error}")

        # --- 2. Block/shell classification ---
        blocked, reason = looks_blocked(html)
        if not blocked:
            shell, shell_reason = looks_like_bot_shell(
                html, query_terms=query_terms,
            )
            if shell:
                blocked = True
                reason = f"bot shell: {shell_reason}"

        primary = FetchResult(
            url=url,
            html=html,
            status_code=200 if html else 0,
            provider="playwright",
            transport=Transport.BROWSER.value,
            bytes_received=len(html),
            blocked=blocked,
            block_reason=reason,
        )
        if playwright_error:
            primary.metadata["playwright_error"] = playwright_error

        if not blocked:
            return primary

        logger.info(
            f"Playwright result for {url} looks blocked "
            f"({reason}, {len(html)} chars) — considering fallback"
        )

        # --- 3. Fallback path ---
        if self.scraperapi is None:
            primary.metadata["fallback_skipped"] = "no provider configured"
            return primary

        if not self.scraperapi.is_configured():
            primary.metadata["fallback_skipped"] = "provider missing key"
            return primary

        # When we already know the page is a bot shell (not just small),
        # go straight to the premium residential-proxy route. This is the
        # only thing that works against Amazon / Walmart / CVS-class
        # defences. Otherwise a plain render is enough.
        use_premium = reason.startswith("bot shell")

        route = NetworkRoute(
            provider="scraperapi",
            transport=Transport.PROVIDER_API.value,
            options={"render": True, "premium": use_premium},
            timeout_seconds=timeout or 60,
        )

        try:
            fallback = await self.scraperapi.fetch(url, route)
        except ProviderError as e:
            safe = scrub(str(e)).scrubbed
            logger.warning(f"ScraperAPI fallback failed on {url}: {safe}")
            primary.metadata["fallback_error"] = safe
            return primary
        except Exception as e:
            safe = scrub(f"{type(e).__name__}: {e}").scrubbed
            logger.warning(f"ScraperAPI fallback raised on {url}: {safe}")
            primary.metadata["fallback_error"] = safe
            return primary

        # Carry Playwright-side diagnostics across to whichever result
        # we end up returning, so a caller inspecting the final result
        # can see what happened on the primary attempt.
        fallback.metadata["playwright_block_reason"] = reason
        fallback.metadata["fallback_from"] = "playwright"
        fallback.metadata["premium_used"] = use_premium
        if playwright_error:
            fallback.metadata["playwright_error"] = playwright_error

        if fallback.blocked:
            logger.info(
                f"ScraperAPI also returned a blocked page for {url}: "
                f"{fallback.block_reason}"
            )
            fallback.metadata["fallback_failed"] = "provider also blocked"
        else:
            logger.info(
                f"ScraperAPI fallback succeeded on {url} "
                f"({fallback.bytes_received} chars, "
                f"{fallback.credits_used} credits, "
                f"premium={use_premium})"
            )

        return fallback


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    from src.network.types import ProviderNotConfigured

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

    async def run():
        good_html = "<html><body>" + "y" * 2000 + "</body></html>"

        # 1. Playwright healthy
        scraper = FakeScraper(good_html)
        provider = FakeProvider()
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "playwright"
        assert not r.blocked
        assert provider.calls == 0

        # 2. Playwright returns a bot shell (medium size, no title/headings)
        shell = (
            "<html><head><title></title></head><body>"
            + "<script>x</script>" * 200
            + "</body></html>"
        )
        scraper = FakeScraper(shell)
        provider = FakeProvider(result=FetchResult(
            url="https://x", html="<html>" + "z" * 2000 + "</html>",
            status_code=200, provider="scraperapi",
            transport="provider_api", bytes_received=2000, credits_used=25,
        ))
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x", query_terms=["panadol"])
        assert r.provider == "scraperapi"
        assert r.credits_used == 25
        # premium was escalated because it was a bot shell, not a small page
        assert provider.last_route.options.get("premium") is True

        # 3. Playwright blocked by size — fallback without premium
        scraper = FakeScraper("Please enable JavaScript to continue")
        provider = FakeProvider(result=FetchResult(
            url="https://x", html="<html>" + "z" * 2000 + "</html>",
            status_code=200, provider="scraperapi",
            transport="provider_api", bytes_received=2000, credits_used=5,
        ))
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "scraperapi"
        assert r.credits_used == 5
        # small-page block -> plain render, not premium
        assert provider.last_route.options.get("premium") is False

        # 4. No provider configured
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(configured=False)
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "playwright"
        assert r.blocked is True
        assert provider.calls == 0

        # 5. Provider raises -> keep Playwright result with metadata
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(raises=RuntimeError("scraperapi down"))
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "playwright"
        assert "fallback_error" in r.metadata

        # 6. Provider also blocked
        scraper = FakeScraper("Please enable JavaScript")
        provider = FakeProvider(result=FetchResult(
            url="https://x", html="Please enable JavaScript",
            status_code=200, provider="scraperapi",
            transport="provider_api", bytes_received=26,
            credits_used=5, blocked=True, block_reason="block marker",
        ))
        m = NetworkManager(scraper, provider)
        r = await m.fetch("https://x")
        assert r.provider == "scraperapi"
        assert "fallback_failed" in r.metadata

        print("NetworkManager OK.")

    asyncio.run(run())