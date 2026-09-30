"""
Unit tests for token-count reporting in LLMRouter (spec §41.4).

The router must return input/output token counts so upstream metering
can compute per-run cost. This file verifies:

    - Gemini usage extraction from `usage_metadata`
    - OpenAI-compatible extraction from `.usage`
    - Defensive defaults when the SDK or a mock doesn't expose usage
    - `call()` includes token counts on successful calls
    - Cache hits report zero tokens (nothing was spent)
"""
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: mocked router that never touches the network
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_router(monkeypatch):
    """
    Build an LLMRouter with every SDK client mocked. Returns
    (router, fake_openai_client, fake_gemini_model).
    """
    env = {
        "GEMINI_API_KEY": "gemini-test",
        "GROQ_API_KEY": "groq-test",
        "OPENROUTER_API_KEY": "openrouter-test",
        "NARA_API_KEY": "sk-nry-test-key",
    }
    monkeypatch.setattr("os.getenv", lambda k, d=None: env.get(k, d))

    with patch("src.llm.router.genai") as fake_genai, \
         patch("src.llm.router.OpenAI") as fake_openai_cls:
        fake_gemini = MagicMock()
        fake_genai.GenerativeModel.return_value = fake_gemini

        fake_client = MagicMock()
        fake_openai_cls.return_value = fake_client

        from src.llm.router import LLMRouter
        # cache=False keeps these tests deterministic — no cache layer
        # interfering with the call-count assertions.
        router = LLMRouter(cache=False)

    return router, fake_client, fake_gemini


# ---------------------------------------------------------------------------
# Gemini extraction
# ---------------------------------------------------------------------------

def test_gemini_reports_token_counts(mocked_router):
    router, _, fake_gemini = mocked_router

    response = MagicMock()
    response.text = "hello"
    response.usage_metadata.prompt_token_count = 120
    response.usage_metadata.candidates_token_count = 40
    fake_gemini.generate_content.return_value = response

    result = router.call("hi", timeout=5)
    assert result["provider"] == "gemini"
    assert result["text"] == "hello"
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 40
    assert result["total_tokens"] == 160


def test_gemini_missing_usage_metadata_gives_zeros(mocked_router):
    """A mock or older SDK without usage_metadata must not crash."""
    router, _, fake_gemini = mocked_router

    response = MagicMock()
    response.text = "hello"
    # Simulate no usage_metadata attribute
    del response.usage_metadata
    fake_gemini.generate_content.return_value = response

    result = router.call("hi", timeout=5)
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


def test_gemini_usage_metadata_none_gives_zeros(mocked_router):
    router, _, fake_gemini = mocked_router

    response = MagicMock()
    response.text = "hello"
    response.usage_metadata = None
    fake_gemini.generate_content.return_value = response

    result = router.call("hi", timeout=5)
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


# ---------------------------------------------------------------------------
# OpenAI-compatible extraction
# ---------------------------------------------------------------------------

def _make_openai_response(text: str, prompt_t: int, completion_t: int):
    """Build a MagicMock that mimics the OpenAI chat-completions shape."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = prompt_t
    response.usage.completion_tokens = completion_t
    return response


def test_openai_compatible_reports_token_counts_via_fallback(
    mocked_router,
):
    """
    Force gemini to fail so the router falls through to groq, which
    reports usage in the OpenAI shape.
    """
    router, fake_client, fake_gemini = mocked_router

    fake_gemini.generate_content.side_effect = RuntimeError("gemini down")
    fake_client.chat.completions.create.return_value = _make_openai_response(
        "groq says hi", 80, 20,
    )

    result = router.call("hi", timeout=5)
    assert result["provider"] == "groq"
    assert result["input_tokens"] == 80
    assert result["output_tokens"] == 20


def test_openai_compatible_missing_usage_gives_zeros(mocked_router):
    router, fake_client, fake_gemini = mocked_router

    fake_gemini.generate_content.side_effect = RuntimeError("gemini down")
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "text"
    del response.usage
    fake_client.chat.completions.create.return_value = response

    result = router.call("hi", timeout=5)
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


def test_openai_compatible_null_usage_gives_zeros(mocked_router):
    router, fake_client, fake_gemini = mocked_router

    fake_gemini.generate_content.side_effect = RuntimeError("gemini down")
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "text"
    response.usage = None
    fake_client.chat.completions.create.return_value = response

    result = router.call("hi", timeout=5)
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


# ---------------------------------------------------------------------------
# Cache-hit behavior
# ---------------------------------------------------------------------------

def test_cache_hit_reports_zero_tokens():
    """A cache hit must not claim to have spent tokens."""
    from src.llm.cache import LLMCache
    from src.llm.router import LLMRouter

    cache = LLMCache(db_path=":memory:")
    try:
        with patch("src.llm.router.genai"), \
             patch("src.llm.router.OpenAI"):
            router = LLMRouter(cache=cache)

        # First call populates the cache
        router._call_gemini = lambda p: __import__(
            "src.llm.router", fromlist=["_ProviderCall"],
        )._ProviderCall(text="hello", input_tokens=100, output_tokens=30)

        r1 = router.call("same prompt", timeout=5)
        assert r1["input_tokens"] == 100
        assert r1["output_tokens"] == 30

        # Second call: cache hit
        r2 = router.call("same prompt", timeout=5)
        assert r2.get("cached") is True
        assert r2["input_tokens"] == 0
        assert r2["output_tokens"] == 0
        assert r2["total_tokens"] == 0
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Backward compatibility with existing router tests
# ---------------------------------------------------------------------------

def test_old_shape_still_present(mocked_router):
    """
    Existing callers only read `text` and `provider`. Those keys must
    still be there; the token keys are additions, not replacements.
    """
    router, _, fake_gemini = mocked_router
    fake_gemini.generate_content.return_value = MagicMock(text="ok")

    result = router.call("hi", timeout=5)
    assert "text" in result
    assert "provider" in result
    assert result["text"] == "ok"
    assert result["provider"] == "gemini"