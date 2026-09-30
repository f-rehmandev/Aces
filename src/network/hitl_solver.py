"""
Human-in-the-Loop (HITL) CAPTCHA solver.

When automated tiers (Byparr, SeleniumBase) fail to bypass a CAPTCHA,
this module launches a headed Playwright browser, navigates to the
target URL, and hands control to the user via page.pause(). The user
solves the CAPTCHA manually in the browser window, then presses
'Resume' in the Playwright Inspector.

After the human finishes, this module extracts the session cookies
(including any anti-bot clearance cookies like cf_clearance) and
returns them as a FetchResult with the final HTML. Downstream tiers
can then proceed with these cookies already in the session.

This is deliberately non-automated: the human is the solver. ACES
provides the plumbing; the human provides the judgment.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, ProviderName, Transport,
)


logger = logging.getLogger("network.hitl_solver")


class HITLSolver:
    """
    Launches a headed browser and pauses for manual CAPTCHA solving.

    Not a provider in the tier chain. Called explicitly by
    NetworkManager when all automated tiers have failed and the
    caller has opted into HITL via route.options['hitl'] = True.
    """

    name = ProviderName.HITL.value

    def __init__(
        self,
        timeout_seconds: int = 300,
        headless: bool = False,
    ):
        self.timeout_seconds = timeout_seconds
        self.headless = headless

    def is_configured(self) -> bool:
        # Always available if the caller is on a machine with a display.
        # In a headless server environment, calling this returns False
        # and NetworkManager skips HITL cleanly.
        import os
        if os.name == "nt":       # Windows — always has a display
            return True
        if os.environ.get("DISPLAY"):
            return True
        return False

    async def fetch(
        self,
        url: str,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        """
        Launch a headed browser, navigate to `url`, pause for the
        human to solve the CAPTCHA, then extract cookies + HTML.
        """
        if not self.is_configured():
            raise ProviderError(
                self.name,
                "HITL requires a display; not available in this environment",
                retryable=False,
            )

        route = route or NetworkRoute()
        timeout = route.timeout_seconds or self.timeout_seconds

        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ProviderError(
                self.name,
                f"playwright not installed: {e}",
                retryable=False,
            ) from e

        # Playwright's sync API works better with page.pause() than
        # the async API, because pause() opens the Inspector which is
        # fundamentally a blocking UI. Run it in a thread.
        return await asyncio.to_thread(
            self._sync_solve, url, route, timeout,
        )

    def _sync_solve(
        self,
        url: str,
        route: NetworkRoute,
        timeout: int,
    ) -> FetchResult:
        """Blocking sync implementation of the HITL flow."""
        import time as _time
        from playwright.sync_api import sync_playwright

        started = _time.monotonic()
        cookies: dict = {}
        html: str = ""
        status = 0

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000},
            )
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)

                self._print_instructions(url, timeout)

                # Baseline: cookies + URL before the human does anything
                baseline_cookies = {c["name"] for c in context.cookies()}
                baseline_url = page.url

                # Watch the browser for signs the human has solved it.
                # No buttons. No prompts. Just wait.
                self._poll_until_solved(
                    page, context, baseline_cookies, baseline_url, timeout,
                )

                # Let the page settle after solve
                page.wait_for_timeout(1500)

                cookies = {
                    c["name"]: c["value"] for c in context.cookies()
                }
                html = page.content()
                status = 200

                logger.info(
                    f"HITL: captured {url}; {len(cookies)} cookie(s), "
                    f"{len(html)} bytes"
                )

            except Exception as e:
                logger.warning(f"HITL failed on {url}: {e}")
                raise ProviderError(
                    self.name,
                    f"human-in-the-loop error: {type(e).__name__}: {e}",
                    retryable=False,
                ) from e
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

        latency_ms = int((_time.monotonic() - started) * 1000)
        return FetchResult(
            url=url,
            html=html,
            status_code=status,
            provider=self.name,
            transport=Transport.BROWSER.value,
            bytes_received=len(html.encode("utf-8", errors="ignore")),
            latency_ms=latency_ms,
            blocked=False,
            metadata={
                "hitl_cookies": cookies,
                "hitl_elapsed_ms": latency_ms,
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _print_instructions(url: str, timeout: int) -> None:
        print()
        print("=" * 66)
        print("  ACES — HUMAN-IN-THE-LOOP")
        print("=" * 66)
        print(f"  URL: {url}")
        print()
        print("  A Chromium window has opened with the target page.")
        print()
        print("  1. Solve the CAPTCHA in that window, whatever kind it is.")
        print()
        print("  2. That's it. ACES is watching the page and will")
        print("     detect the solve automatically (via cookies, URL")
        print("     change, or the page's own success marker).")
        print()
        print("  3. You do NOT need to press anything. Just wait.")
        print()
        print("     You can close the browser window at any time to")
        print("     force-capture the current state.")
        print()
        print(f"  Timeout: {timeout}s")
        print("=" * 66)
        print()

    @staticmethod
    def _poll_until_solved(
        page, context, baseline_cookies, baseline_url, timeout,
    ) -> bool:
        """
        Poll for signs the human has actually solved the CAPTCHA.

        Signals we trust:
            - Browser closed by user (force capture)
            - User pressed Enter in the terminal (force capture)
            - URL changed
            - A KNOWN SOLVE COOKIE appeared (cf_clearance, etc.)
            - The CAPTCHA's response token field got filled

        Signals we explicitly IGNORE:
            - _GRECAPTCHA  (set on checkbox click, not on solve)
            - __cf_bm      (set on page load, not on solve)
        """
        import time as _time
        import sys

        # Cookies that ONLY appear after a successful solve.
        SOLVE_COOKIES = {
            "cf_clearance",         # Cloudflare
            "cf_chl_2",             # Cloudflare legacy
            "cf_chl_prog",
            "ak_bmsc",              # Akamai (sometimes)
        }

        # Fields that hold the CAPTCHA response token — only filled on solve.
        RESPONSE_FIELDS = (
            'textarea[name="g-recaptcha-response"]',   # Google reCAPTCHA
            'input[name="cf-turnstile-response"]',     # Cloudflare Turnstile
            'input[name="h-captcha-response"]',        # hCaptcha
        )

        # Try to set up non-blocking keyboard reading (Windows only).
        _kbhit = None
        try:
            import msvcrt
            _kbhit = msvcrt.kbhit
            _getch = msvcrt.getch
        except ImportError:
            _kbhit = None

        deadline = _time.monotonic() + timeout
        last_tick = _time.monotonic()

        print("  [HITL] Watching for solve. Press ENTER in this terminal")
        print("         to force-capture at any time.")
        print()

        while _time.monotonic() < deadline:
            # --- Keyboard escape: user pressed a key in the terminal ---
            if _kbhit is not None and _kbhit():
                try:
                    _getch()
                except Exception:
                    pass
                print()
                print("  [HITL] Manual capture requested (keypress).")
                return True

            # --- Browser closed by the user = force capture ---
            try:
                if page.is_closed():
                    print()
                    print("  [HITL] Browser closed — capturing final state.")
                    return True
            except Exception:
                print()
                print("  [HITL] Browser closed — capturing final state.")
                return True

            try:
                # --- URL changed (typical after a solve redirect) ---
                if page.url != baseline_url:
                    print()
                    print(f"  [HITL] URL changed → {page.url[:80]}")
                    return True

                # --- A real solve cookie appeared ---
                current = {c["name"] for c in context.cookies()}
                new = current - baseline_cookies
                real_solve_cookies = new & SOLVE_COOKIES
                if real_solve_cookies:
                    print()
                    print(f"  [HITL] Solve cookie(s): "
                          f"{', '.join(sorted(real_solve_cookies))}")
                    return True

                # --- A response token field has a value ---
                for selector in RESPONSE_FIELDS:
                    try:
                        value = page.eval_on_selector(
                            selector,
                            "el => (el && el.value) ? el.value : ''",
                        )
                        if value and len(str(value)) > 100:
                            print()
                            print(f"  [HITL] CAPTCHA response token found "
                                  f"({len(value)} chars).")
                            return True
                    except Exception:
                        continue

            except Exception:
                # Mid-navigation — try again
                pass

            # Status tick every 15 seconds
            now = _time.monotonic()
            if now - last_tick >= 15:
                remaining = int(deadline - now)
                print(f"  [HITL] Still watching… {remaining}s remaining "
                      f"(press ENTER in this window to force-capture)",
                      flush=True)
                last_tick = now

            _time.sleep(0.5)

        print()
        print("  [HITL] Timeout reached without detecting a solve.")
        print("  [HITL] Capturing whatever's on the page now.")
        return False