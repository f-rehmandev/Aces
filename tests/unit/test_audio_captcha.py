"""
Unit tests for the Whisper audio-CAPTCHA solver.

Covers:
    - Happy path: driver captures audio, transcriber returns text,
      result.solved is True
    - Site has no audio option → solved=False, specific error
    - Driver raises → solved=False, error captured
    - Transcriber returns empty → solved=False, "empty transcription"
    - Transcriber raises → solved=False, error captured
    - No audio bytes captured → solved=False
    - is_available() reports False when faster-whisper isn't installed
"""
import asyncio

import pytest

from src.network.audio_captcha import (
    AudioCaptchaSolver,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_solve():
    async def driver(url, timeout):
        return b"AUDIO", "#audio-response", "#verify", "ok"

    solver = AudioCaptchaSolver(
        driver=driver, transcriber=lambda b: "hello world",
    )
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is True
    assert r.answer == "hello world"
    assert r.audio_bytes == len(b"AUDIO")
    assert r.latency_ms >= 0
    assert r.error == ""


def test_transcript_is_stripped():
    async def driver(url, timeout):
        return b"AUDIO", "#field", "#submit", "ok"

    solver = AudioCaptchaSolver(
        driver=driver, transcriber=lambda b: "  hello  \n",
    )
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.answer == "hello"


# ---------------------------------------------------------------------------
# Failure modes → solved=False, clean error
# ---------------------------------------------------------------------------

def test_no_audio_option():
    async def driver(url, timeout):
        return b"", "", "", "no_audio"

    solver = AudioCaptchaSolver(driver=driver, transcriber=lambda b: "x")
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is False
    assert "audio challenge" in r.error.lower()


def test_driver_error_status():
    async def driver(url, timeout):
        return b"", "", "", "error:browser crashed"

    solver = AudioCaptchaSolver(driver=driver, transcriber=lambda b: "x")
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is False
    assert "browser crashed" in r.error


def test_driver_raises():
    async def driver(url, timeout):
        raise RuntimeError("network down")

    solver = AudioCaptchaSolver(driver=driver, transcriber=lambda b: "x")
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is False
    assert "network down" in r.error
    assert "RuntimeError" in r.error


def test_empty_audio_bytes():
    async def driver(url, timeout):
        return b"", "#f", "#s", "ok"

    solver = AudioCaptchaSolver(driver=driver, transcriber=lambda b: "x")
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is False
    assert "no bytes" in r.error.lower()


def test_empty_transcription():
    async def driver(url, timeout):
        return b"AUDIO", "#f", "#s", "ok"

    solver = AudioCaptchaSolver(driver=driver, transcriber=lambda b: "")
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is False
    assert "empty transcription" in r.error.lower()
    # audio_bytes still reported so callers can meter
    assert r.audio_bytes == len(b"AUDIO")


def test_transcriber_raises():
    async def driver(url, timeout):
        return b"AUDIO", "#f", "#s", "ok"

    def boom(b):
        raise RuntimeError("model failed")

    solver = AudioCaptchaSolver(driver=driver, transcriber=boom)
    r = _run(solver.solve("https://x.example/challenge"))
    assert r.solved is False
    assert "model failed" in r.error


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------

def test_is_available_with_injected_callables():
    """Injected callables make the solver trivially available."""
    solver = AudioCaptchaSolver(
        driver=lambda u, t: None,
        transcriber=lambda b: "",
    )
    # No real check needed — but the method must not raise.
    result = solver.is_available()
    assert isinstance(result, bool)


def test_is_available_reports_false_when_deps_missing(monkeypatch):
    """Force both imports to fail — is_available() must return False."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("playwright", "faster_whisper") or \
                name.startswith(("playwright.", "faster_whisper.")):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    solver = AudioCaptchaSolver()   # uses both defaults
    assert solver.is_available() is False