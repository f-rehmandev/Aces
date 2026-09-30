"""
Apify provider — Tier 3e.

Apify runs "actors" (cloud-hosted scrapers) rather than fetching URLs
directly. Free tier: $5 credit/month, 10 GB data transfer. When the
credit is spent, the account pauses until the next monthly cycle.

We use the `apify/website-content-crawler` actor configured to return
cleaned HTML for a single URL. This gives us a FetchResult shaped the
same way as our other providers.

API docs: https://docs.apify.com/api/v2
Endpoint (run-sync-get-dataset-items):
    POST https://api.apify.com/v2/acts/{actorId}/run-sync-get-dataset-items?token={token}
Auth: `token` query parameter.

Response on success: dataset items as JSON (default). The Website
Content Crawler returns one item per URL with fields including `html`
(when outputFormat="html"), `text`, `url`, `title`, etc.

Response on error: JSON `{"error": {"type": "...", "message": "..."}}`.
The sync endpoint has a 300-second timeout.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Protocol

from src.network.block_codes import (
    BlockKind, classify_http_status, classify_provider_error,
    should_fall_through,
)
from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, ProviderNotConfigured,
    ProviderName, Transport, looks_blocked,
)


logger = logging.getLogger("network.apify")


DEFAULT_ENDPOINT = "https://api.apify.com/v2"
DEFAULT_ACTOR = "apify/website-content-crawler"


class HttpTransport(Protocol):
    async def post_json(
        self, url: str, params: dict, body: dict, timeout: float,
    ) -> tuple[int, str]: ...


class HttpxTransport:
    async def post_json(
        self, url: str, params: dict, body: dict, timeout: float,
    ) -> tuple[int, str]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, params=params, json=body)
            return r.status_code, r.text or ""


def _extract_html_from_items(items: list) -> tuple[str, dict]:
    """
    Apify's Website Content Crawler returns one item per URL. We want the
    `html` field (present when outputFormat="html"). Fall back to
    wrapping `text` in a minimal HTML shell if `html` is absent.

    Returns (html, metadata) where metadata holds useful extras from the
    item (title, url, crawl info).
    """
    if not items or not isinstance(items, list):
        return "", {}

    item = items[0]
    if not isinstance(item, dict):
        return "", {}

    metadata: dict = {}
    for key in ("title", "url", "description", "crawlerType"):
        v = item.get(key)
        if v:
            metadata[key] = v

    html = item.get("html")
    if isinstance(html, str) and html:
        return html, metadata

    # Some configurations return "text" (markdown or plain)
    text = item.get("text") or item.get("markdown")
    if isinstance(text, str) and text:
        # Wrap in a minimal HTML shell so downstream extractors have
        # something to parse. The text may contain markdown; leave it
        # as-is inside <pre> — the LLM extractor handles it.
        return f"<html><body><pre>{text}</pre></body></html>", metadata

    return "", metadata


class ApifyProvider:
    """Tier 3e — Apify (via the Website Content Crawler actor)."""

    name = ProviderName.APIFY.value

    def __init__(
        self,
        api_token: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        actor: str = DEFAULT_ACTOR,
        transport: Optional[HttpTransport] = None,
    ):
        if api_token is None:
            api_token = os.getenv("APIFY_API_TOKEN", "")
        self.api_token = api_token
        self.endpoint = endpoint.rstrip("/")
        self.actor = actor
        self._transport = transport or HttpxTransport()

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.api_token)

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        if not self.is_configured():
            raise ProviderNotConfigured(self.name, "APIFY_API_TOKEN")

        route = route or NetworkRoute()
        options = route.options or {}

        # Allow callers to override the actor via route.options.
        actor = options.get("apify_actor") or self.actor
        actor_path = actor.replace("/", "~")

        params = {"token": self.api_token}

        # Website Content Crawler input for a single URL.
        #  - maxCrawlPages: 1        → one page only
        #  - maxCrawlDepth: 0        → no link following
        #  - outputFormat: html      → return cleaned HTML
        #  - crawlerType: playwright → for JS-heavy pages (default)
        #
        # Apify returns JSON by default from run-sync-get-dataset-items.
        # The `format` query param can request html/csv/jsonl instead; we
        # stick with JSON because the actor's dataset items carry richer
        # metadata (title, url, etc.) that we want.
        body = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": 1,
            "maxCrawlDepth": 0,
            "outputFormat": "html",
            # The actor's accepted values changed upstream — the old
            # "playwright" is rejected. "playwright:adaptive" is the
            # current default (auto-picks between browser and HTTP).
            "crawlerType": options.get(
                "crawler_type", "playwright:adaptive",
            ),
        }

        # Apify's proxy config: turn on if the caller asked for premium.
        if options.get("premium"):
            body["proxyConfiguration"] = {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
            }
        else:
            body["proxyConfiguration"] = {"useApifyProxy": True}

        # Optional country (Apify exposes proxy country under this key)
        if options.get("country"):
            body["proxyConfiguration"]["apifyProxyCountry"] = str(
                options["country"]
            ).upper()

        api_url = (
            f"{self.endpoint}/acts/{actor_path}/run-sync-get-dataset-items"
        )

        started = time.monotonic()
        try:
            status, text = await self._transport.post_json(
                api_url, params, body, timeout=min(300, route.timeout_seconds),
            )
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"transport error: {type(e).__name__}: {e}",
                retryable=should_fall_through(kind),
            ) from e

        latency_ms = int((time.monotonic() - started) * 1000)

        if status >= 400:
            kind = classify_http_status(status)
            raise ProviderError(
                self.name,
                f"HTTP {status} from Apify: {text[:300]}",
                retryable=should_fall_through(kind),
            )

        # The sync endpoint returns dataset items (JSON array) on success.
        # Some error shapes come back as a JSON object with `error`.
        try:
            parsed = json.loads(text) if text else []
        except json.JSONDecodeError:
            # Not JSON — treat as raw HTML (some actors bypass the
            # dataset wrapper).
            parsed = None

        if isinstance(parsed, dict) and parsed.get("error"):
            err_msg = str(parsed["error"])
            raise ProviderError(
                self.name,
                f"apify actor error: {err_msg}",
                retryable=True,
            )

        if isinstance(parsed, list):
            html, extra_meta = _extract_html_from_items(parsed)
        else:
            # Fallback: treat the raw text as HTML
            html = text
            extra_meta = {}

        blocked = False
        block_reason = ""
        if not html:
            blocked = True
            block_reason = "apify returned no html content"
        else:
            blocked, block_reason = looks_blocked(html)

        metadata: dict = {"apify_actor": actor}
        metadata.update(extra_meta)

        return FetchResult(
            url=url,
            html=html,
            status_code=status,
            provider=self.name,
            transport=Transport.PROVIDER_API.value,
            bytes_received=len(html.encode("utf-8", errors="ignore")),
            credits_used=1,   # approximate — Apify bills by compute unit
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=block_reason,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class FakeTransport:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict, dict, float]] = []
        async def post_json(self, url, params, body, timeout):
            self.calls.append((url, params, body, timeout))
            return self.response

    async def run():
        # 1. Not configured
        p = ApifyProvider(api_token="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except ProviderNotConfigured:
            pass

        # 2. Happy path — dataset item with `html` field
        items = json.dumps([{
            "url": "https://x.example/",
            "title": "Example",
            "html": "<html><body><h1>" + "a" * 600 + "</h1></body></html>",
        }])
        fake = FakeTransport((200, items))
        p = ApifyProvider(api_token="k", transport=fake)
        assert p.is_configured()
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.provider == "apify"
        assert r.status_code == 200
        assert not r.blocked
        assert r.bytes_received > 600
        assert "<h1>" in r.html
        assert r.metadata["apify_actor"] == DEFAULT_ACTOR
        assert r.metadata["title"] == "Example"

        url_called, params, body, _ = fake.calls[0]
        assert url_called.endswith("/run-sync-get-dataset-items")
        assert "apify~website-content-crawler" in url_called
        assert params["token"] == "k"
        assert body["startUrls"] == [{"url": "https://x.example/"}]
        assert body["maxCrawlPages"] == 1
        assert body["maxCrawlDepth"] == 0
        assert body["outputFormat"] == "html"

        # 3. Fallback: item with only `text` (no `html`)
        items = json.dumps([{
            "url": "https://x.example/",
            "title": "TextOnly",
            "text": "some plain text content " * 40,
        }])
        fake = FakeTransport((200, items))
        p = ApifyProvider(api_token="k", transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert "some plain text content" in r.html
        assert r.blocked is False
        assert r.metadata["title"] == "TextOnly"

        # 4. Empty dataset → blocked
        fake = FakeTransport((200, "[]"))
        p = ApifyProvider(api_token="k", transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True
        assert "no html" in r.block_reason.lower()

        # 5. Actor error in JSON body
        fake = FakeTransport((200, json.dumps({
            "error": {"type": "actor-failed", "message": "out of credits"},
        })))
        p = ApifyProvider(api_token="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert "out of credits" in str(e)

        # 6. HTTP 429 → ProviderError, retryable
        fake = FakeTransport((429, "rate limit"))
        p = ApifyProvider(api_token="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 7. HTTP 401 → ProviderError, retryable
        fake = FakeTransport((401, "bad token"))
        p = ApifyProvider(api_token="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 8. Transport exception
        class Boom:
            async def post_json(self, *a, **k):
                raise RuntimeError("network unreachable")
        p = ApifyProvider(api_token="k", transport=Boom())
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 9. Premium option sets residential proxy group
        items = json.dumps([{"html": "x" * 1000, "url": "u", "title": "t"}])
        fake = FakeTransport((200, items))
        p = ApifyProvider(api_token="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"premium": True}))
        _, _, body, _ = fake.calls[0]
        assert body["proxyConfiguration"]["apifyProxyGroups"] == ["RESIDENTIAL"]

        # 10. Country option uppercased
        fake = FakeTransport((200, items))
        p = ApifyProvider(api_token="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"country": "pk"}))
        _, _, body, _ = fake.calls[0]
        assert body["proxyConfiguration"]["apifyProxyCountry"] == "PK"

        # 11. Custom actor override
        fake = FakeTransport((200, items))
        p = ApifyProvider(api_token="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"apify_actor": "apify/web-scraper"}))
        url_called, _, _, _ = fake.calls[0]
        assert "apify~web-scraper" in url_called

        print("Apify provider OK.")

    asyncio.run(run())