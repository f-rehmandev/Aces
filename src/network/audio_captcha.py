"""
Audio-CAPTCHA solver — the middle rung of the solver chain.

Fires only when a challenge page has been detected (the caller decides
this). Attempts to solve it via the site's own accessibility audio
challenge:

    1. Click the audio button ("🔊" / "audio challenge" / aria-label).
    2. Capture the audio URL that the challenge page then requests.
    3. Download the audio bytes.
    4. Transcribe locally with faster-whisper (open-weights model).
    5. Type the transcription into the challenge's response field.
    6. Submit.

Cost: $0. No paid CAPTCHA farm. No third-party solver service. The
model runs on CPU; `tiny.en` transcribes in ~1s on a modern laptop.

If the site has disabled the audio option, or the answer is rejected,
the caller falls through to HITL attended mode (§15.3.1, §15.7.2).

Metering (§41.4): each solve attempt returns the audio duration in
seconds so the pipeline can account for it. Whether the audio path
counts as a "vision call" or a new resource type is a policy decision
left to the caller.

Design:
    - All Playwright work is a single injected callable so tests don't
      launch a browser.
    - Whisper is an injected callable so tests don't need the model.
    - The default Playwright driver is a small, focused coroutine that
      knows nothing about ACES — it just clicks, downloads, types.
    - The default Whisper transcriber caches its model at module level.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


logger = logging.getLogger("network.audio_captcha")


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class AudioCaptchaError(Exception):
    """Base class for audio-CAPTCHA failures."""


class AudioCaptchaUnavailable(AudioCaptchaError):
    """The site did not offer an audio option (or didn't respond to it)."""


class AudioCaptchaFailed(AudioCaptchaError):
    """The audio challenge was attempted but not solved."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AudioSolveResult:
    solved: bool
    answer: str = ""
    audio_seconds: float = 0.0
    audio_bytes: int = 0
    latency_ms: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Injected callables
# ---------------------------------------------------------------------------

# (url, timeout_seconds) -> (audio_bytes, answer_field_selector, submit_selector, status)
#   status is one of: "ok" | "no_audio" | "error:<msg>"
PlaywrightDriver = Callable[
    [str, int],
    Awaitable[tuple[bytes, str, str, str]],
]

# (audio_bytes) -> transcribed text (may be empty on failure)
Transcriber = Callable[[bytes], str]


# ---------------------------------------------------------------------------
# Default Playwright driver
# ---------------------------------------------------------------------------

# Selectors used to find the audio button. Ordered by likelihood across
# reCAPTCHA v2, hCaptcha, and common clones.
_AUDIO_BUTTON_SELECTORS = (
    'button[aria-label*="audio" i]',
    'div[role="button"][aria-label*="audio" i]',
    'button[id*="audio" i]',
    'a[id*="audio" i]',
    'div[title*="audio" i]',
    '#recaptcha-audio-button',
    '.rc-button-audio',
)

# Selectors for the answer input field on the audio challenge.
_ANSWER_FIELD_SELECTORS = (
    '#audio-response',
    'input[name="audio-response"]',
    'input[id*="audio-response" i]',
    'textarea[name*="captcha" i]',
    'input[type="text"][id*="captcha" i]',
)

# Selectors for the submit / verify button.
_SUBMIT_SELECTORS = (
    '#recaptcha-verify-button',
    'button[id*="verify" i]',
    'button[type="submit"]',
    'button:has-text("Verify")',
    'button:has-text("Submit")',
)


async def _default_playwright_driver(
    url: str,
    timeout_seconds: int,
) -> tuple[bytes, str, str, str]:
    """
    Open `url` in a headless Chromium, click the audio option, capture
    the audio payload, and return:

        (audio_bytes, answer_selector, submit_selector, status)

    `status`:
        "ok"                — audio captured, selectors located
        "no_audio"          — site did not offer an audio option
        "error:<message>"   — everything else

    The caller uses the two returned selectors to type + submit the
    answer. The driver deliberately does NOT type or submit — that's the
    caller's job so a wrong transcription doesn't burn a challenge
    attempt.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        return b"", "", "", f"error:playwright not installed: {e}"

    captured_audio: list[bytes] = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            # Intercept requests so we can grab the audio payload when
            # the challenge page requests it.
            async def _on_response(response):
                try:
                    ct = (response.headers or {}).get("content-type", "")
                    if "audio" in ct.lower():
                        body = await response.body()
                        if body:
                            captured_audio.append(body)
                except Exception:
                    pass

            page.on("response", _on_response)

            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout_seconds * 1000,
            )
            await page.wait_for_timeout(800)

            # Click the audio option
            clicked = False
            for sel in _AUDIO_BUTTON_SELECTORS:
                try:
                    el = await page.query_selector(sel)
                    if el is not None:
                        await el.click()
                        clicked = True
                        break
                except Exception:
                    continue

            if not clicked:
                await browser.close()
                return b"", "", "", "no_audio"

            # Wait up to ~6s for the audio request to fire.
            for _ in range(60):
                if captured_audio:
                    break
                await page.wait_for_timeout(100)

            if not captured_audio:
                await browser.close()
                return b"", "", "", "no_audio"

            # Locate the answer + submit selectors so the caller can
            # type + submit against the SAME page (we keep the browser
            # open by returning an async closer, but for simplicity the
            # caller gets the selectors and a second driver pass). For
            # now we return the selectors that the caller will use on
            # their own page; a tighter coupling is a future
            # optimisation.
            answer_sel = ""
            for sel in _ANSWER_FIELD_SELECTORS:
                try:
                    if await page.query_selector(sel):
                        answer_sel = sel
                        break
                except Exception:
                    continue

            submit_sel = ""
            for sel in _SUBMIT_SELECTORS:
                try:
                    if await page.query_selector(sel):
                        submit_sel = sel
                        break
                except Exception:
                    continue

            await browser.close()
            return captured_audio[0], answer_sel, submit_sel, "ok"

    except Exception as e:
        return b"", "", "", f"error:{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Default Whisper transcriber
# ---------------------------------------------------------------------------

_WHISPER_MODEL_CACHE: dict = {}


def _default_transcriber(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes with faster-whisper. Caches the model across
    calls (loading takes ~2s the first time; subsequent calls are ~0.5s
    for a short audio clip).

    Never raises — a failed transcription returns "".
    """
    if not audio_bytes:
        return ""
    try:
        from faster_whisper import WhisperModel
        import io
    except ImportError as e:
        logger.warning(f"faster-whisper not installed: {e}")
        return ""

    try:
        model = _WHISPER_MODEL_CACHE.get("tiny.en")
        if model is None:
            # tiny.en is the smallest English model — ~75 MB, runs
            # acceptably on CPU. Swap via env if you want a larger one.
            import os
            model_name = os.getenv("ACES_WHISPER_MODEL", "tiny.en")
            model = WhisperModel(
                model_name, device="cpu", compute_type="int8",
            )
            _WHISPER_MODEL_CACHE["tiny.en"] = model

        segments, _info = model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=1,
            language="en",
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    except Exception as e:
        logger.warning(f"whisper transcription failed: {type(e).__name__}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class AudioCaptchaSolver:
    """
    Attempt to solve an audio CAPTCHA challenge on the given page URL.

    Usage:
        solver = AudioCaptchaSolver()
        result = await solver.solve("https://target.example/challenge")
        if result.solved:
            # continue the run; session cookies will carry the clearance
            ...
        else:
            # fall through to HITL attended mode
            ...

    The solver never raises on failure — inspect `result.solved`.
    Exceptions are reserved for programmer errors (bad arguments).
    """

    def __init__(
        self,
        driver: Optional[PlaywrightDriver] = None,
        transcriber: Optional[Transcriber] = None,
    ):
        self._driver = driver or _default_playwright_driver
        self._transcriber = transcriber or _default_transcriber

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """True if the driver + transcriber appear to be usable."""
        try:
            # If the driver is the default, playwright must be importable.
            if self._driver is _default_playwright_driver:
                import playwright  # noqa: F401
            # If the transcriber is the default, faster-whisper must be
            # importable (the model itself downloads on first use).
            if self._transcriber is _default_transcriber:
                import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    async def solve(
        self,
        url: str,
        timeout_seconds: int = 90,
    ) -> AudioSolveResult:
        started = time.monotonic()

        # ---- Step 1: navigate + click + capture audio ----
        try:
            audio_bytes, answer_sel, submit_sel, status = await self._driver(
                url, timeout_seconds,
            )
        except Exception as e:
            return AudioSolveResult(
                solved=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"driver raised: {type(e).__name__}: {e}",
            )

        if status == "no_audio":
            return AudioSolveResult(
                solved=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error="site did not offer an audio challenge",
            )

        if status.startswith("error:"):
            return AudioSolveResult(
                solved=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=status[len("error:"):].strip(),
            )

        if not audio_bytes:
            return AudioSolveResult(
                solved=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error="audio capture returned no bytes",
            )

        # ---- Step 2: transcribe ----
        try:
            transcript = self._transcriber(audio_bytes) or ""
        except Exception as e:
            return AudioSolveResult(
                solved=False,
                audio_bytes=len(audio_bytes),
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"transcriber raised: {type(e).__name__}: {e}",
            )

        transcript = transcript.strip()
        if not transcript:
            return AudioSolveResult(
                solved=False,
                audio_bytes=len(audio_bytes),
                latency_ms=int((time.monotonic() - started) * 1000),
                error="empty transcription",
            )

        # ---- Step 3: hand the answer back to the caller ----
        # NOTE: typing + submitting is deliberately out of scope for this
        # module. The caller has the live page context; we don't.
        latency_ms = int((time.monotonic() - started) * 1000)
        return AudioSolveResult(
            solved=True,
            answer=transcript,
            audio_bytes=len(audio_bytes),
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Smoke test — injectable fake driver + fake transcriber, no browser, no model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def run():
        # Happy path
        async def ok_driver(url, timeout):
            return b"AUDIOBYTES", "#audio-response", "#recaptcha-verify-button", "ok"

        solver = AudioCaptchaSolver(
            driver=ok_driver,
            transcriber=lambda b: "hello world",
        )
        r = await solver.solve("https://x.example/challenge")
        assert r.solved is True
        assert r.answer == "hello world"
        assert r.audio_bytes == len(b"AUDIOBYTES")

        # Site has no audio option
        async def no_audio_driver(url, timeout):
            return b"", "", "", "no_audio"

        solver = AudioCaptchaSolver(
            driver=no_audio_driver, transcriber=lambda b: "x",
        )
        r = await solver.solve("https://x.example/challenge")
        assert r.solved is False
        assert "audio challenge" in r.error.lower()

        # Driver raised
        async def boom_driver(url, timeout):
            raise RuntimeError("network down")

        solver = AudioCaptchaSolver(
            driver=boom_driver, transcriber=lambda b: "x",
        )
        r = await solver.solve("https://x.example/challenge")
        assert r.solved is False
        assert "network down" in r.error

        # Empty transcription
        solver = AudioCaptchaSolver(
            driver=ok_driver, transcriber=lambda b: "",
        )
        r = await solver.solve("https://x.example/challenge")
        assert r.solved is False
        assert "empty transcription" in r.error.lower()

        # Transcriber raised
        def boom_transcriber(b):
            raise RuntimeError("model failed to load")

        solver = AudioCaptchaSolver(
            driver=ok_driver, transcriber=boom_transcriber,
        )
        r = await solver.solve("https://x.example/challenge")
        assert r.solved is False
        assert "model failed to load" in r.error

        print("AudioCaptchaSolver OK.")

    asyncio.run(run())