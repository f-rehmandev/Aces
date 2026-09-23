import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from strategy.strategy_memory import get_strategy, record_failure_and_adapt
from playwright_stealth import Stealth
import asyncio
import logging
from playwright.async_api import async_playwright
import config
from strategy.strategy_memory import get_strategy, record_failure_and_adapt, wait_for_rate_limit

logger = logging.getLogger("scraper_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ScraperEngine:
    """
    Handles visiting a URL with a real browser and returning its HTML.
    Concurrency is capped so we never open more than N browser tabs at once.
    """

    def __init__(self, max_concurrent: int = None, headless: bool = True):
        max_concurrent = max_concurrent or config.MAX_CONCURRENT_BROWSERS
        self.max_concurrent = max_concurrent
        self.headless = headless
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_html(self, url: str, timeout: int = None) -> str:
        async with self._semaphore:
            strategy = get_strategy(url)
            effective_timeout = timeout or strategy.get("timeout", config.DEFAULT_TIMEOUT_MS)

            logger.info(f"Fetching: {url} (timeout={effective_timeout}ms)")

            await wait_for_rate_limit(url)

            async with Stealth().use_async(async_playwright()) as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1366, "height": 768},
                )
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=effective_timeout)
                    await page.wait_for_timeout(1500)
                    html = await page.content()
                    logger.info(f"Success: {url} ({len(html)} chars)")
                    return html
                except Exception as e:
                    logger.warning(f"Failed on {url}, adapting strategy: {e}")
                    record_failure_and_adapt(url, strategy)
                    raise
                finally:
                    await browser.close()


    async def fetch_screenshot(self, url: str, timeout: int = None) -> bytes:
        """Returns a PNG screenshot of the rendered page, for vision-based extraction."""
        async with self._semaphore:
            strategy = get_strategy(url)
            effective_timeout = timeout or strategy.get("timeout", config.DEFAULT_TIMEOUT_MS)

            logger.info(f"Screenshotting: {url}")

            async with Stealth().use_async(async_playwright()) as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1366, "height": 768},
                )
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=effective_timeout)
                    await page.wait_for_timeout(1500)
                    screenshot_bytes = await page.screenshot(full_page=False)
                    logger.info(f"Screenshot captured: {url} ({len(screenshot_bytes)} bytes)")
                    return screenshot_bytes
                except Exception as e:
                    logger.warning(f"Screenshot failed on {url}: {e}")
                    record_failure_and_adapt(url, strategy)
                    raise
                finally:
                    await browser.close()


    async def fetch_paginated(self, start_url: str, max_pages: int = 5, next_link_selector: str = "a[rel='next']") -> list[str]:
        """
        Follows 'next page' links up to max_pages, returning HTML from each page.
        Stops early on: no next link found, or two consecutive pages look identical
        (no-progress detection — project knowledge Section 14.3).
        """
        async with self._semaphore:
            strategy = get_strategy(start_url)
            effective_timeout = strategy.get("timeout", config.DEFAULT_TIMEOUT_MS)

            pages_html = []
            async with Stealth().use_async(async_playwright()) as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1366, "height": 768},
                )
                try:
                    current_url = start_url
                    for page_num in range(max_pages):
                        logger.info(f"Fetching page {page_num + 1}/{max_pages}: {current_url}")
                        await wait_for_rate_limit(current_url)
                        await page.goto(current_url, wait_until="domcontentloaded", timeout=effective_timeout)
                        await page.wait_for_timeout(1500)
                        html = await page.content()

                        if pages_html and html == pages_html[-1]:
                            logger.warning("No-progress detected (identical page content) — stopping pagination.")
                            break
                        pages_html.append(html)

                        next_href = None
                        try:
                            next_el = await page.query_selector(next_link_selector)
                            if next_el:
                                next_href = await next_el.get_attribute("href")
                        except Exception:
                            pass

                        if not next_href:
                            logger.info("No next-page link found — stopping pagination.")
                            break

                        current_url = next_href if next_href.startswith("http") else f"{page.url.split('?')[0].rsplit('/', 1)[0]}/{next_href.lstrip('/')}"

                    return pages_html
                except Exception as e:
                    logger.warning(f"Pagination failed: {e}")
                    record_failure_and_adapt(start_url, strategy)
                    return pages_html
                finally:
                    await browser.close()

async def _demo():
    engine = ScraperEngine()
    html = await engine.fetch_html("https://example.com")
    print(f"\nGot {len(html)} characters of HTML.")
    print("First 300 chars:\n")
    print(html[:300])


if __name__ == "__main__":
    asyncio.run(_demo())