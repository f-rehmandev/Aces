"""Unit tests for the nararouter provider in LLMRouter.

All provider backends are mocked at the module level, so these tests
make zero network calls and don't depend on any real API keys.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: build an LLMRouter with every backend mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_router(monkeypatch):
    """
    Returns (router, fake_openai_client, fake_genai_model).

    The same fake OpenAI client is shared by groq, openrouter, and
    nararouter because they all call `OpenAI(...)` — that's fine for
    testing routing order via a call counter.
    """
    env = {
        "GEMINI_API_KEY": "gemini-test",
        "GROQ_API_KEY": "groq-test",
        "OPENROUTER_API_KEY": "openrouter-test",
        "NARA_API_KEY": "sk-nry-test-key",
    }
    monkeypatch.setattr(os, "getenv", lambda k, d=None: env.get(k, d))

    with patch("src.llm.router.genai") as fake_genai, \
         patch("src.llm.router.OpenAI") as fake_openai_cls:
        fake_genai_model = MagicMock()
        fake_genai.GenerativeModel.return_value = fake_genai_model

        fake_client = MagicMock()
        fake_openai_cls.return_value = fake_client

        from src.llm.router import LLMRouter
        router = LLMRouter()

    return router, fake_client, fake_genai_model


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_nararouter_is_in_provider_order(mocked_router):
    router, _, _ = mocked_router
    assert "nararouter" in router._providers


def test_nararouter_is_last_provider(mocked_router):
    router, _, _ = mocked_router
    assert router._providers[-1] == "nararouter"


def test_nararouter_client_built_when_key_present(mocked_router):
    router, _, _ = mocked_router
    assert router._nara_client is not None


def test_nararouter_client_is_none_without_key(monkeypatch):
    env = {
        "GEMINI_API_KEY": "g",
        "GROQ_API_KEY": "g",
        "OPENROUTER_API_KEY": "g",
        # NARA_API_KEY deliberately absent
    }
    monkeypatch.setattr(os, "getenv", lambda k, d=None: env.get(k, d))

    with patch("src.llm.router.genai"), patch("src.llm.router.OpenAI"):
        from src.llm.router import LLMRouter
        router = LLMRouter()

    assert router._nara_client is None


# ---------------------------------------------------------------------------
# _call_nararouter
# ---------------------------------------------------------------------------

def test_call_nararouter_raises_without_client():
    from src.llm.router import LLMRouter
    router = LLMRouter.__new__(LLMRouter)
    router._nara_client = None
    with pytest.raises(RuntimeError) as exc:
        router._call_nararouter("hi")
    assert "NARA_API_KEY" in str(exc.value)


def test_call_nararouter_returns_text(mocked_router):
    router, fake_client, _ = mocked_router
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="hello from nara"))
    ]
    # `_call_*` methods now return a `_ProviderCall` (text + token
    # usage) rather than a bare string, so upstream metering can
    # record cost per call (§41.4).
    result = router._call_nararouter("hi")
    assert result.text == "hello from nara"


def test_call_nararouter_passes_model_and_prompt(mocked_router):
    router, fake_client, _ = mocked_router
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="ok"))
    ]
    router._call_nararouter("what is 2+2?")
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "laguna-s-2.1"
    assert kwargs["messages"] == [
        {"role": "user", "content": "what is 2+2?"}
    ]


# ---------------------------------------------------------------------------
# Routing: nararouter is reached only after earlier providers fail
# ---------------------------------------------------------------------------

def test_call_uses_gemini_when_healthy(mocked_router):
    router, _, fake_genai_model = mocked_router
    fake_genai_model.generate_content.return_value = MagicMock(text="gemini wins")
    result = router.call("hi", timeout=5)
    assert result["provider"] == "gemini"
    assert result["text"] == "gemini wins"


def test_call_falls_back_to_nararouter_when_all_others_fail(mocked_router):
    router, fake_client, fake_genai_model = mocked_router

    # Gemini fails
    fake_genai_model.generate_content.side_effect = RuntimeError("gemini down")

    # groq and openrouter fail; nararouter succeeds (3rd call to the shared client)
    calls = {"n": 0}
    def side_effect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("provider down")
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="nara wins"))]
        )
    fake_client.chat.completions.create.side_effect = side_effect

    result = router.call("hi", timeout=5)
    assert result["provider"] == "nararouter"
    assert result["text"] == "nara wins"


def test_call_raises_when_every_provider_fails(mocked_router):
    router, fake_client, fake_genai_model = mocked_router
    fake_genai_model.generate_content.side_effect = RuntimeError("gemini down")
    fake_client.chat.completions.create.side_effect = RuntimeError("all down")

    with pytest.raises(RuntimeError) as exc:
        router.call("hi", timeout=5)
    assert "All providers failed" in str(exc.value)