"""
Camoufox provider — Tier 1b.

Camoufox is a Firefox-based anti-detection browser built on a patched
Firefox engine (not a JavaScript injection like playwright-stealth).
It evades fingerprinting at the C++ layer, which makes it harder for
modern anti-bot systems (Cloudflare, DataDome, PerimeterX) to detect.

Free and open-source. No API key. Requires:
    1. pip install 'camoufox[geoip]'
    2. python -m camoufox fetch     (downloads the patched browser)

The binary download is a one-time step; the Dockerfile runs it during
the image build so containers ship with Camoufox ready.

If the package is not installed, `is_configured()` returns False and
the NetworkManager skips this provider cleanly — same pattern as the
SeleniumBase provider.

Fetch logic is injected via `fetch_fn` so tests never launch a real
browser.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

from src.network.block_codes import (
    BlockKind,
    classify_provider_error,
)
from src.network.types import (
    FetchResult,
    NetworkRoute,
    ProviderError,
    ProviderName,
    ProviderNotConfigured,
    Transport,
)


logger = logging.getLogger("network.camoufox")


# Phrases that mean a challenge page slipped through
_UNSOLVED_MARKERS = (
    "just a moment",
    "checking your browser",
    "cf-chl-",
    "challenge-platform",
    "please verify you are human",
)


# (url, cookies, timeout_seconds, headless) -> (html, status_code, error)
FetchFn = Callable[
    [str, dict, int, bool],
    Awaitable[tuple[str, int, str]],
]


# ---------------------------------------------------------------------------
# Default fetch implementation
# ---------------------------------------------------------------------------

async def _default_fetch(
    url: str,
    cookies: dict,
    timeout_seconds: int,
    headless: bool,
) -> tuple[str, int, str]:
    """
    Open the URL in Camoufox and return the rendered HTML.

    Returns (html, status_code, error). Never raises — every failure
    comes back as a non-empty `error` string.
    """
    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError as e:
        return "", 0, f"camoufox not installed: {e}"

    try:
        async with AsyncCamoufox(headless=headless) as browser:
            context = await browser.new_context()
            page = await context.new_page()

            # Inject cookies before navigating — provide domain + path
            # so Playwright doesn't require us to be on the target page.
            if cookies:
                domain = urlparse(url).netloc
                cookie_list = [
                    {
                        "name": str(name),
                        "value": str(value),
                        "domain": domain,
                        "path": "/",
                    }
                    for name, value in cookies.items()
                ]
                try:
                    await context.add_cookies(cookie_list)
                except Exception as e:
                    logger.debug(f"camoufox cookie injection failed: {e}")

            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout_seconds * 1000,
            )
            # Give the page a beat to finish rendering dynamic content.
            await page.wait_for_timeout(1200)

            html = await page.content() or ""
            return html, 200, ""
    except Exception as e:
        return "", 0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class CamoufoxProvider:
    """
    Tier 1b — Firefox-based anti-detection browser.

    Stronger stealth than playwright-stealth, weaker than
    SeleniumBase's UC Mode in some scenarios (and vice versa — the
    two complement each other), so the manager tries both.
    """

    name = ProviderName.CAMOUFOX.value

    def __init__(
        self,
        headless: bool = True,
        fetch_fn: Optional[FetchFn] = None,
    ):
        self.headless = headless
        self._fetch_fn = fetch_fn or _default_fetch
        self._injected = fetch_fn is not None

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        if self._injected:
            return True
        try:
            import camoufox  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        if not self.is_configured():
            raise ProviderNotConfigured(
                self.name,
                "camoufox (pip install 'camoufox[geoip]')",
            )

        route = route or NetworkRoute()
        timeout_seconds = int(route.timeout_seconds or 60)
        cookies = dict(route.session_cookies or {})

        try:
            html, status, err = await self._fetch_fn(
                url, cookies, timeout_seconds, self.headless,
            )
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"camoufox fetch error: {type(e).__name__}: {e}",
                retryable=kind in (
                    BlockKind.TIMEOUT, BlockKind.NETWORK_ERROR,
                ),
            ) from e

        if err and not html:
            kind = classify_provider_error(RuntimeError(err))
            raise ProviderError(
                self.name,
                err,
                retryable=kind in (
                    BlockKind.TIMEOUT, BlockKind.NETWORK_ERROR,
                ),
            )

        # Challenge-page detection
        blocked = False
        block_reason = ""
        if not html:
            blocked = True
            block_reason = "camoufox returned empty html"
        else:
            lower = html[:5000].lower()
            for marker in _UNSOLVED_MARKERS:
                if marker in lower:
                    blocked = True
                    block_reason = f"camoufox unsolved challenge: {marker!r}"
                    break

        return FetchResult(
            url=url,
            html=html,
            status_code=status or 200,
            provider=self.name,
            transport=Transport.BROWSER.value,
            bytes_received=len(html.encode("utf-8", errors="ignore")),
            blocked=blocked,
            block_reason=block_reason,
        )


# ---------------------------------------------------------------------------
# Smoke test — injectable fake fetch, no browser
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def run():
        async def ok_fetch(url, cookies, timeout, headless):
            return "<html>ok</html>", 200, ""

        async def empty_fetch(url, cookies, timeout, headless):
            return "", 0, "RuntimeError: browser crashed"

        async def challenge_fetch(url, cookies, timeout, headless):
            return (
                '<html><title>Just a moment...</title>'
                '<body>cf-chl-xyz</body></html>',
                200, "",
            )

        # Happy path
        p = CamoufoxProvider(fetch_fn=ok_fetch)
        assert p.is_configured()
        r = await p.fetch("https://example.com/", NetworkRoute())
        assert r.provider == "camoufox"
        assert r.transport == "browser"
        assert r.status_code == 200
        assert not r.blocked

        # Empty response with error → ProviderError
        p = CamoufoxProvider(fetch_fn=empty_fetch)
        try:
            await p.fetch("https://example.com/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert "browser crashed" in str(e)

        # Challenge page → blocked
        p = CamoufoxProvider(fetch_fn=challenge_fetch)
        r = await p.fetch("https://example.com/", NetworkRoute())
        assert r.blocked is True
        assert "unsolved challenge" in r.block_reason

        # Cookie pass-through
        captured = {}
        async def capture_fetch(url, cookies, timeout, headless):
            captured["cookies"] = dict(cookies)
            captured["url"] = url
            captured["timeout"] = timeout
            return "<html>ok</html>", 200, ""
        p = CamoufoxProvider(fetch_fn=capture_fetch)
        await p.fetch(
            "https://linkedin.example/",
            NetworkRoute(
                session_cookies={"li_at": "abc"},
                timeout_seconds=45,
            ),
        )
        assert captured["cookies"] == {"li_at": "abc"}
        assert captured["timeout"] == 45

        print("CamoufoxProvider OK.")

    asyncio.run(run())