import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from strategy.strategy_memory import get_strategy, record_failure_and_adapt
from playwright_stealth import Stealth
import asyncio
import logging
import time
from functools import wraps
from playwright.async_api import async_playwright
import config
from strategy.strategy_memory import get_strategy, record_failure_and_adapt, wait_for_rate_limit

logger = logging.getLogger("scraper_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
from src.navigation.budget import CrawlBudget
from src.navigation.progress import html_unchanged


def _metered(fn):
    """
    Decorator: accumulate browser-seconds on the engine instance (§41.4).

    Every public fetch method on ScraperEngine is wrapped so a run's total
    browser time can be metered. Timing is recorded even when the method
    raises — the browser time was still spent.

    `self.total_browser_seconds` and `self.browser_call_count` are the
    accumulators; `ScraperEngine.reset_usage()` zeros them.
    """
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return await fn(self, *args, **kwargs)
        finally:
            self.total_browser_seconds += time.monotonic() - start
            self.browser_call_count += 1
    return wrapper


class ScraperEngine:
    """
    Handles visiting a URL with a real browser and returning its HTML.
    Concurrency is capped so we never open more than N browser tabs at once.

    Browser time is metered per instance (§41.4) so the pipeline can report
    total browser-seconds consumed by a run. The pipeline calls
    `reset_usage()` at the start of every run.
    """

    def __init__(self, max_concurrent: int = None, headless: bool = True):
        max_concurrent = max_concurrent or config.MAX_CONCURRENT_BROWSERS
        self.max_concurrent = max_concurrent
        self.headless = headless
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # --- usage meter (§41.4) ---
        self.total_browser_seconds: float = 0.0
        self.browser_call_count: int = 0

    def reset_usage(self) -> None:
        """Zero the browser-time meter. Called by PipelineRunner per run."""
        self.total_browser_seconds = 0.0
        self.browser_call_count = 0

    @_metered
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

    @_metered
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

    @_metered
    async def fetch_paginated(
        self,
        start_url: str,
        max_pages: int = 5,
        next_link_selector: str = "a[rel='next'], li.next a, a.next, a[aria-label='Next']",
        budget: CrawlBudget = None,
    ) -> list[str]:
        """
        Follows "next page" links up to a budget, returning HTML from each page.

        Stops on:
          - no next-link found
          - two consecutive pages look identical (§14.3 no-progress)
          - crawl budget exhausted (§14.1)

        If `budget` is provided, it takes precedence and `max_pages` is ignored.
        """
        from urllib.parse import urljoin

        if budget is None:
            budget = CrawlBudget(max_pages=max_pages)

        async with self._semaphore:
            strategy = get_strategy(start_url)
            effective_timeout = strategy.get("timeout", config.DEFAULT_TIMEOUT_MS)
            pages_html: list[str] = []

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
                    while True:
                        ok, reason = budget.can_continue()
                        if not ok:
                            logger.info(f"Pagination stopping: {reason}  {budget.snapshot()}")
                            break

                        logger.info(
                            f"Fetching page {budget.pages_fetched + 1}: {current_url}"
                        )
                        await wait_for_rate_limit(current_url)
                        await page.goto(
                            current_url,
                            wait_until="domcontentloaded",
                            timeout=effective_timeout,
                        )
                        await page.wait_for_timeout(1500)
                        html = await page.content()

                        if pages_html and html_unchanged(pages_html[-1], html):
                            logger.warning("No-progress detected (equivalent page) — stopping pagination.")
                            break

                        pages_html.append(html)
                        budget.consume_page(byte_count=len(html))

                        next_href = None
                        try:
                            next_el = await page.query_selector(next_link_selector)
                            if next_el:
                                next_href = await next_el.get_attribute("href")
                        except Exception as e:
                            logger.debug(f"next-link selector error (non-fatal): {e}")

                        if not next_href:
                            logger.info("No next-page link found — stopping pagination.")
                            break

                        # urljoin resolves /relative, ../relative, and absolute URLs correctly.
                        current_url = urljoin(page.url, next_href)

                    return pages_html

                except Exception as e:
                    logger.warning(f"Pagination failed: {e}")
                    record_failure_and_adapt(start_url, strategy)
                    return pages_html
                finally:
                    await browser.close()

    @_metered
    async def fetch_infinite_scroll(
        self,
        url: str,
        max_scrolls: int = 10,
        scroll_pause_ms: int = 800,
        scroll_pixels: int = 1500,
        budget: CrawlBudget = None,
    ) -> str:
        """
        Loads `url`, scrolls the page repeatedly to trigger lazy loading,
        and returns the FULL final HTML.

        Stops when:
          - max_scrolls reached
          - document.body.scrollHeight stops growing (nothing new loaded)
          - budget exhausted

        NOTE: Some sites replace the DOM with the current viewport rather than
        appending. For those, the returned HTML reflects only the final state.
        This is a known limitation of browser-based infinite scroll.
        """
        if budget is None:
            budget = CrawlBudget(max_pages=max_scrolls)

        async with self._semaphore:
            strategy = get_strategy(url)
            effective_timeout = strategy.get("timeout", config.DEFAULT_TIMEOUT_MS)

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
                    await wait_for_rate_limit(url)
                    logger.info(f"Infinite-scroll: loading {url}")
                    await page.goto(url, wait_until="domcontentloaded", timeout=effective_timeout)
                    await page.wait_for_timeout(scroll_pause_ms)

                    last_height = 0
                    for scroll_num in range(max_scrolls):
                        ok, reason = budget.can_continue()
                        if not ok:
                            logger.info(f"Infinite-scroll stopping: {reason}")
                            break

                        height = await page.evaluate("document.body.scrollHeight")
                        if height == last_height:
                            logger.info(f"Infinite-scroll: no growth after {scroll_num} scroll(s) — stopping.")
                            break
                        last_height = height

                        logger.info(f"Infinite-scroll: scroll {scroll_num + 1}/{max_scrolls} (height={height})")
                        await page.evaluate(f"window.scrollBy(0, {scroll_pixels})")
                        await page.wait_for_timeout(scroll_pause_ms)
                        budget.consume_page()

                    # Give lazy content a final moment to settle
                    await page.wait_for_timeout(500)
                    html = await page.content()
                    logger.info(f"Infinite-scroll: final HTML = {len(html)} chars, {budget.pages_fetched} scroll(s)")
                    return html

                except Exception as e:
                    logger.warning(f"Infinite-scroll failed on {url}: {e}")
                    record_failure_and_adapt(url, strategy)
                    # Best-effort: return whatever HTML we have
                    try:
                        return await page.content()
                    except Exception:
                        return ""
                finally:
                    await browser.close()

    @_metered
    async def fetch_after_click(
        self,
        url: str,
        click_selector: str,
        wait_after_click_ms: int = 1500,
    ) -> str:
        """
        Loads `url`, clicks the element matching `click_selector`, waits,
        and returns the resulting HTML.

        Useful for: "Load all reviews" buttons, expanding accordions,
        switching tabs, applying a filter.

        Errors on: selector not found, click failure, page navigation.
        """
        async with self._semaphore:
            strategy = get_strategy(url)
            effective_timeout = strategy.get("timeout", config.DEFAULT_TIMEOUT_MS)

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
                    await wait_for_rate_limit(url)
                    logger.info(f"Loading {url} for click on {click_selector!r}")
                    await page.goto(url, wait_until="domcontentloaded", timeout=effective_timeout)
                    await page.wait_for_timeout(800)

                    await page.click(click_selector, timeout=effective_timeout)
                    logger.info(f"Clicked {click_selector!r}, waiting {wait_after_click_ms}ms")
                    await page.wait_for_timeout(wait_after_click_ms)

                    html = await page.content()
                    logger.info(f"Post-click HTML: {len(html)} chars")
                    return html

                except Exception as e:
                    logger.warning(f"Click failed on {url} ({click_selector!r}): {e}")
                    record_failure_and_adapt(url, strategy)
                    raise
                finally:
                    await browser.close()


async def _demo():
    engine = ScraperEngine()
    html = await engine.fetch_html("https://example.com")
    print(f"\nGot {len(html)} characters of HTML.")
    print(f"Browser time: {engine.total_browser_seconds:.2f}s "
          f"over {engine.browser_call_count} call(s)")
    print("First 300 chars:\n")
    print(html[:300])


if __name__ == "__main__":
    asyncio.run(_demo())