"""
curl_cffi HTTP driver — Tier 0.

Fast async HTTP with a browser-mimicking TLS fingerprint. Clears the
"your TLS handshake says you're a Python client" class of bot blocks
without launching a browser at all.

Free, no API key. Uses curl_cffi (a Python binding around
curl-impersonate), which sends a Firefox/Chrome-style TLS ClientHello,
HTTP/2 settings, and header ordering.

Session factory is injected so tests never touch the network and so a
future deployment can swap in a session with custom settings.

When curl_cffi is not installed, `is_configured()` returns False and the
provider is skipped in the tier chain — same pattern as seleniumbase.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from src.network.block_codes import (
    BlockKind,
    classify_provider_error,
    should_fall_through,
)
from src.network.types import (
    FetchResult,
    NetworkRoute,
    ProviderError,
    ProviderName,
    ProviderNotConfigured,
    Transport,
    looks_blocked,
)


logger = logging.getLogger("network.curl_cffi")


DEFAULT_IMPERSONATE = "chrome"


class _SessionLike(Protocol):
    async def __aenter__(self) -> "_SessionLike": ...
    async def __aexit__(self, *args) -> None: ...
    async def get(self, url: str, **kwargs): ...


def _default_session_factory() -> _SessionLike:
    """Lazy import — the module is optional at import-time."""
    from curl_cffi.requests import AsyncSession
    return AsyncSession()


class CurlCffiProvider:
    """
    Tier 0 — fast HTTP with browser TLS fingerprint.

    No API key. No configuration required beyond `pip install curl-cffi`.
    If the package is missing, the provider reports itself as
    unconfigured and the NetworkManager skips it cleanly.
    """

    name = ProviderName.CURL_CFFI.value

    def __init__(
        self,
        default_impersonate: str = DEFAULT_IMPERSONATE,
        session_factory: Optional[callable] = None,
    ):
        self.default_impersonate = default_impersonate
        self._session_factory = session_factory or _default_session_factory
        self._injected = session_factory is not None

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        # Tests inject a session factory — trust it.
        if self._injected:
            return True
        try:
            import curl_cffi  # noqa: F401
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
                "curl-cffi (pip install curl-cffi)",
            )

        route = route or NetworkRoute()
        options = route.options or {}

        # Caller can override the impersonation profile per route.
        impersonate = str(
            options.get("curl_impersonate", self.default_impersonate)
        )

        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        kwargs: dict = {
            "headers": headers,
            "impersonate": impersonate,
            "timeout": route.timeout_seconds,
            "allow_redirects": True,
        }
        cookies = route.session_cookies or {}
        if cookies:
            kwargs["cookies"] = dict(cookies)

        started = time.monotonic()
        try:
            async with self._session_factory() as session:
                response = await session.get(url, **kwargs)
            latency_ms = int((time.monotonic() - started) * 1000)
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"transport error: {type(e).__name__}: {e}",
                retryable=should_fall_through(kind),
            ) from e

        status = int(getattr(response, "status_code", 0) or 0)
        html = getattr(response, "text", "") or ""

        blocked, block_reason = looks_blocked(html)

        return FetchResult(
            url=url,
            html=html,
            status_code=status,
            provider=self.name,
            transport=Transport.DIRECT_HTTP.value,
            bytes_received=len(html.encode("utf-8", errors="ignore")),
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=block_reason,
            metadata={"impersonate": impersonate},
        )


# ---------------------------------------------------------------------------
# Smoke test — injectable fake session, no network
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class _FakeResponse:
        def __init__(self, status=200, text=""):
            self.status_code = status
            self.text = text

    class _FakeSession:
        def __init__(self, response):
            self.response = response
            self.last_kwargs = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            self.last_kwargs = kwargs
            return self.response

    async def run():
        # Happy path
        fake = _FakeSession(_FakeResponse(
            200, "<html><body>" + ("x" * 1000) + "</body></html>",
        ))
        p = CurlCffiProvider(session_factory=lambda: fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.provider == "curl_cffi"
        assert r.status_code == 200
        assert not r.blocked
        assert r.metadata["impersonate"] == "chrome"
        assert fake.last_kwargs["impersonate"] == "chrome"
        assert fake.last_kwargs["timeout"] == 60

        # Block detection
        fake2 = _FakeSession(_FakeResponse(
            200, "Please enable JavaScript to continue",
        ))
        p2 = CurlCffiProvider(session_factory=lambda: fake2)
        r2 = await p2.fetch("https://x.example/", NetworkRoute())
        assert r2.blocked is True
        assert "javascript" in r2.block_reason.lower()

        # Custom impersonate via route
        fake3 = _FakeSession(_FakeResponse(200, "x" * 1000))
        p3 = CurlCffiProvider(session_factory=lambda: fake3)
        await p3.fetch(
            "https://x.example/",
            NetworkRoute(options={"curl_impersonate": "firefox133"}),
        )
        assert fake3.last_kwargs["impersonate"] == "firefox133"

        # Cookies pass through
        fake4 = _FakeSession(_FakeResponse(200, "x" * 1000))
        p4 = CurlCffiProvider(session_factory=lambda: fake4)
        await p4.fetch(
            "https://x.example/",
            NetworkRoute(session_cookies={"li_at": "abc"}),
        )
        assert fake4.last_kwargs["cookies"] == {"li_at": "abc"}

        # Transport error → ProviderError
        class _BoomSession:
            async def __aenter__(self):
                raise RuntimeError("connection refused")

            async def __aexit__(self, *args):
                return None

            async def get(self, *args, **kwargs):
                raise RuntimeError("connection refused")

        p5 = CurlCffiProvider(session_factory=lambda: _BoomSession())
        try:
            await p5.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert "connection refused" in str(e)

        # Not configured when curl_cffi isn't installed and no factory injected
        p6 = CurlCffiProvider()
        # On this dev machine curl_cffi is probably not installed; either
        # way is_configured() must return a bool without raising.
        assert isinstance(p6.is_configured(), bool)

        print("CurlCffiProvider OK.")

    asyncio.run(run())