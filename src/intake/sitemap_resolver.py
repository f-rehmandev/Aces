"""
Sitemap input resolver — spec §10 / §14.1.

Takes a sitemap URL, fetches the XML, parses it (reusing our sitemap parser),
follows a sitemap index one level deep, and returns a TaskSpec whose
target.start_urls is the list of canonicalized URLs.

Design:
    - Fetching is injected via a `fetcher: Callable[[str], str]` so tests
      don't need network. Default fetcher uses `requests` (no browser needed).
    - Child sitemaps in a <sitemapindex> are followed with a bounded count.
    - A URL cap prevents a 100k-URL sitemap from blowing up memory.
    - URLs are canonicalized and deduplicated.
"""

from __future__ import annotations
from typing import Callable, Optional
import logging

from src.core.task_spec import TaskSpec, Target
from src.intake.resolver import InputResolution
from src.navigation.sitemap import parse_sitemap
from src.navigation.url_normalizer import normalize_url

logger = logging.getLogger("sitemap_resolver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _default_http_fetch(url: str) -> str:
    """Sitemaps are plain XML — no browser needed, so we use requests."""
    import requests
    headers = {"User-Agent": "ACES-Project/1.0 (student portfolio project)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text


class SitemapResolver:
    """See module docstring for the model."""

    def __init__(
        self,
        fetcher: Optional[Callable[[str], str]] = None,
        max_child_sitemaps: int = 20,
        max_urls: int = 10_000,
    ):
        self._fetch = fetcher or _default_http_fetch
        self.max_child_sitemaps = max_child_sitemaps
        self.max_urls = max_urls

    def resolve(self, sitemap_url: str) -> InputResolution:
        warnings: list[str] = []

        # --- fetch root sitemap ---
        try:
            xml = self._fetch(sitemap_url)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch sitemap {sitemap_url}: {e}") from e

        try:
            result = parse_sitemap(xml)
        except ValueError as e:
            warnings.append(f"malformed sitemap XML: {e}")
            result = None

        raw_urls: list[str] = []

        if result is None:
            pass

        elif result.is_index:
            children = result.child_sitemaps[: self.max_child_sitemaps]
            if len(result.child_sitemaps) > self.max_child_sitemaps:
                warnings.append(
                    f"truncated to {self.max_child_sitemaps} child sitemaps "
                    f"(had {len(result.child_sitemaps)})"
                )
            for child in children:
                try:
                    child_result = parse_sitemap(self._fetch(child))
                except Exception as e:
                    warnings.append(f"failed to fetch/parse child {child}: {e}")
                    continue
                if child_result.is_index:
                    warnings.append(f"nested sitemap index not followed: {child}")
                    continue
                for entry in child_result.entries:
                    raw_urls.append(entry.url)
                    if len(raw_urls) >= self.max_urls:
                        break
                if len(raw_urls) >= self.max_urls:
                    break

        else:
            for entry in result.entries:
                raw_urls.append(entry.url)
                if len(raw_urls) >= self.max_urls:
                    break

        if len(raw_urls) >= self.max_urls:
            warnings.append(f"hit max_urls cap ({self.max_urls})")

        # --- canonicalize + dedupe ---
        seen: set[str] = set()
        urls: list[str] = []
        for u in raw_urls:
            normalized = normalize_url(u).normalized
            if not normalized or not normalized.startswith(("http://", "https://")):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)

        spec = TaskSpec(
            natural_language_prompt=f"sitemap: {sitemap_url}",
            target=Target(start_urls=urls),
        )
        return InputResolution(
            spec=spec,
            source_description=f"sitemap: {sitemap_url}",
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Smoke test (uses a fake fetcher — no network)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    URLSET = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://x.com/a</loc></url>
      <url><loc>https://x.com/b?utm_source=ignored</loc></url>
    </urlset>
    """
    responses = {"https://x.com/sitemap.xml": URLSET}
    resolver = SitemapResolver(fetcher=lambda u: responses[u])
    r = resolver.resolve("https://x.com/sitemap.xml")
    print("URLs:", r.spec.target.start_urls)
    print("Warnings:", r.warnings)
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]

    print("SitemapResolver OK.")