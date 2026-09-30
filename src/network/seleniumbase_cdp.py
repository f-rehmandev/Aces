"""
SeleniumBase CDP provider — spec §15.3 (Tier 1a).

SeleniumBase's UC Mode (undetected-chromedriver) combined with CDP Mode
is the current state of the art for local anti-detection. It's the
"maximum stealth" path per SeleniumBase's own architecture docs.

Trade-offs:
    - Slower than Playwright (Selenium startup is heavier)
    - Requires SeleniumBase + Chrome/Chromium installed on the host
    - Runs sync (we wrap with asyncio.to_thread)

Since it's optional, the class is import-safe: if SeleniumBase isn't
installed, `is_configured()` returns False and the manager skips it
cleanly. This lets the same code run on dev machines that don't have
SeleniumBase installed.

Public API:

    provider = SeleniumBaseCDPProvider()
    if provider.is_configured():
        result = await provider.fetch(url, route)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

from src.network.block_codes import (
    BlockKind, classify_provider_error,
)
from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, ProviderNotConfigured,
    ProviderName, Transport,
)


logger = logging.getLogger("network.seleniumbase_cdp")


# Phrases that mean the returned HTML is still a challenge page
_UNSOLVED_MARKERS = (
    "just a moment",
    "checking your browser",
    "cf-chl-",
    "challenge-platform",
    "please verify you are human",
)


def _is_seleniumbase_available() -> bool:
    """Soft check — don't import SeleniumBase unless we need it."""
    try:
        import seleniumbase  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The sync worker (runs in a thread)
# ---------------------------------------------------------------------------

def _sync_fetch(
    url: str,
    session_cookies: dict,
    timeout_seconds: int,
    uc: bool = True,
    headless: bool = True,
) -> tuple[str, int, str]:
    """
    Runs in a worker thread. Returns (html, status_code, error_message).

    Never raises — errors come back through the third tuple element so
    the caller can turn them into a FetchResult without a try/except
    dance.
    """
    try:
        from seleniumbase import SB
    except Exception as e:
        return "", 0, f"seleniumbase import failed: {e}"

    try:
        # SB() is a context manager; it starts Chrome, does the
        # undetected handshake, and closes the browser on exit.
        with SB(uc=uc, headless=headless, test=False) as sb:
            sb.set_page_load_timeout(timeout_seconds)

            # Inject cookies if the caller supplied any. Selenium
            # requires us to have the domain first, so navigate
            # briefly to set the cookie context.
            if session_cookies:
                try:
                    sb.open(url)
                    for name, value in session_cookies.items():
                        sb.add_cookie({"name": str(name), "value": str(value)})
                except Exception as e:
                    logger.debug(f"cookie injection failed: {e}")

            # uc_open_with_reconnect disconnects from the initial
            # challenge for `reconnect_time` seconds. That's what lets
            # Cloudflare's JS challenge complete before we reconnect.
            try:
                sb.uc_open_with_reconnect(url, reconnect_time=4)
            except Exception:
                # Fall back to a plain open — some sites don't need
                # the reconnect dance.
                sb.open(url)

            # Give the page a beat to finish rendering dynamic content.
            sb.sleep(1.5)

            html = sb.get_page_source() or ""
            return html, 200, ""
    except Exception as e:
        return "", 0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class SeleniumBaseCDPProvider:
    name = ProviderName.SELENIUMBASE_CDP.value

    def __init__(
        self,
        headless: bool = True,
        uc: bool = True,
        fetch_fn: Optional[Callable] = None,
    ):
        self.headless = headless
        self.uc = uc
        # Inject a fetch fn in tests to avoid the real browser
        self._fetch_fn = fetch_fn or _sync_fetch

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        # If a fake fetch_fn was injected, we're always "configured".
        if self._fetch_fn is not _sync_fetch:
            return True
        return _is_seleniumbase_available()

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        if not self.is_configured():
            raise ProviderNotConfigured(
                self.name,
                "seleniumbase (pip install seleniumbase)",
            )

        route = route or NetworkRoute()
        cookies = dict(route.session_cookies or {})
        timeout = int(route.timeout_seconds or 60)

        started = time.monotonic()
        try:
            html, status, err = await asyncio.to_thread(
                self._fetch_fn,
                url,
                cookies,
                timeout,
                self.uc,
                self.headless,
            )
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"thread error: {type(e).__name__}: {e}",
                retryable=kind in (
                    BlockKind.TIMEOUT, BlockKind.NETWORK_ERROR,
                ),
            ) from e

        latency_ms = int((time.monotonic() - started) * 1000)

        if err and not html:
            kind = classify_provider_error(RuntimeError(err))
            raise ProviderError(
                self.name, err,
                retryable=kind in (
                    BlockKind.TIMEOUT, BlockKind.NETWORK_ERROR,
                ),
            )

        # Detect unsolved challenge pages
        blocked = False
        block_reason = ""
        if not html:
            blocked = True
            block_reason = "seleniumbase returned empty html"
        else:
            lower = html[:5000].lower()
            for marker in _UNSOLVED_MARKERS:
                if marker in lower:
                    blocked = True
                    block_reason = f"seleniumbase unsolved challenge: {marker!r}"
                    break

        return FetchResult(
            url=url,
            html=html,
            status_code=status or 200,
            provider=self.name,
            transport=Transport.BROWSER.value,
            bytes_received=len(html.encode("utf-8", errors="ignore")),
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=block_reason,
        )


# ---------------------------------------------------------------------------
# Smoke test — injectable fetch_fn, no real browser launched
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    def _fake_ok(url, cookies, timeout, uc, headless):
        return "<html>content from seleniumbase</html>", 200, ""

    def _fake_empty(url, cookies, timeout, uc, headless):
        return "", 0, "RuntimeError: chrome crashed"

    def _fake_challenge(url, cookies, timeout, uc, headless):
        return (
            '<html><title>Just a moment...</title>'
            '<body>cf-chl-xyz</body></html>',
            200, "",
        )

    async def run():
        # 1. Happy path
        p = SeleniumBaseCDPProvider(fetch_fn=_fake_ok)
        assert p.is_configured()
        r = await p.fetch("https://example.com/", NetworkRoute())
        assert r.provider == "seleniumbase_cdp"
        assert r.transport == "browser"
        assert r.status_code == 200
        assert not r.blocked
        assert "content from seleniumbase" in r.html

        # 2. Empty response with error → ProviderError
        p = SeleniumBaseCDPProvider(fetch_fn=_fake_empty)
        try:
            await p.fetch("https://example.com/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert "chrome crashed" in str(e)

        # 3. Challenge page → blocked=True, no exception
        p = SeleniumBaseCDPProvider(fetch_fn=_fake_challenge)
        r = await p.fetch("https://example.com/", NetworkRoute())
        assert r.blocked is True
        assert "unsolved challenge" in r.block_reason

        # 4. Not configured when SeleniumBase isn't installed
        #    (uses the real _sync_fetch and checks the soft import)
        p = SeleniumBaseCDPProvider()   # real fetch_fn
        configured = p.is_configured()
        # On this test machine SeleniumBase may or may not be installed;
        # either way, is_configured() must return a bool without raising.
        assert isinstance(configured, bool)

        # 5. Missing when not configured → ProviderNotConfigured
        if not configured:
            try:
                await p.fetch("https://example.com/", NetworkRoute())
                raise AssertionError("expected ProviderNotConfigured")
            except ProviderNotConfigured as e:
                assert "seleniumbase" in str(e).lower()

        # 6. Cookie injection is passed through to the fetch_fn
        captured = {}
        def capture_fn(url, cookies, timeout, uc, headless):
            captured["cookies"] = dict(cookies)
            captured["url"] = url
            return "<html>ok</html>", 200, ""
        p = SeleniumBaseCDPProvider(fetch_fn=capture_fn)
        route = NetworkRoute(session_cookies={"li_at": "xyz"})
        await p.fetch("https://linkedin.example/", route)
        assert captured["cookies"] == {"li_at": "xyz"}
        assert captured["url"] == "https://linkedin.example/"

        print("SeleniumBase CDP provider OK.")

    asyncio.run(run())