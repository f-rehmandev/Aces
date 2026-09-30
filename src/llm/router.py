from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import os
import logging
import sys
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from src.llm.cache import LLMCache, build_default_cache, compute_key


load_dotenv()
logger = logging.getLogger("llm_router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Internal return shape for the provider-specific call methods
# ---------------------------------------------------------------------------

@dataclass
class _ProviderCall:
    """What every `_call_*` method returns: text + token usage."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _extract_gemini_usage(response) -> tuple[int, int]:
    """
    Pull (prompt, candidates) token counts out of a Gemini response.
    Returns (0, 0) if the SDK doesn't report usage (older versions, mocks).
    """
    try:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return 0, 0
        return (
            int(getattr(meta, "prompt_token_count", 0) or 0),
            int(getattr(meta, "candidates_token_count", 0) or 0),
        )
    except Exception:
        return 0, 0


def _extract_openai_usage(response) -> tuple[int, int]:
    """
    Pull (prompt, completion) token counts out of an OpenAI-compatible
    response. Groq, OpenRouter, and nararouter all use this shape.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
    except Exception:
        return 0, 0


class LLMRouter:
    """
    Provider-agnostic LLM caller. Tries providers in order until one succeeds.
    This is the single entry point the rest of the project uses to talk to any LLM.

    Optional response cache (§42): when a cache is supplied (or auto-built
    from env), identical prompts return instantly without an API call.

    Every successful call returns token counts (§41.4) so upstream code
    can meter LLM cost per run.

    Cache control:
        LLMRouter()               -> auto-build from env
        LLMRouter(cache=False)    -> force-disable the cache
        LLMRouter(cache=mycache)  -> use the supplied LLMCache instance
    """

    def __init__(self, cache=None):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)

        self._groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        self._openrouter_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        # nararouter is OpenAI-compatible; only construct the client when
        # a key is present so the router works fine without one.
        _nara_key = os.getenv("NARA_API_KEY")
        self._nara_client = (
            OpenAI(api_key=_nara_key, base_url=config.NARA_BASE_URL)
            if _nara_key else None
        )

        self._providers = config.LLM_PROVIDER_ORDER

        # --- response cache (§42) ---
        # Three modes: False -> disabled, instance -> use it,
        # None (default) -> auto-build from env.
        if cache is False:
            self._cache: LLMCache | None = None
        elif cache is None:
            self._cache = build_default_cache()
        else:
            self._cache = cache

    # ------------------------------------------------------------------
    # Provider-specific call methods
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str) -> _ProviderCall:
        response = self._gemini_model.generate_content(prompt)
        text = (response.text or "").strip()
        input_t, output_t = _extract_gemini_usage(response)
        return _ProviderCall(text=text, input_tokens=input_t, output_tokens=output_t)

    def _call_groq(self, prompt: str) -> _ProviderCall:
        response = self._groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        input_t, output_t = _extract_openai_usage(response)
        return _ProviderCall(text=text, input_tokens=input_t, output_tokens=output_t)

    def _call_openrouter(self, prompt: str) -> _ProviderCall:
        response = self._openrouter_client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        input_t, output_t = _extract_openai_usage(response)
        return _ProviderCall(text=text, input_tokens=input_t, output_tokens=output_t)

    def _call_nararouter(self, prompt: str) -> _ProviderCall:
        if self._nara_client is None:
            raise RuntimeError(
                "NARA_API_KEY is not set — nararouter provider disabled."
            )
        response = self._nara_client.chat.completions.create(
            model=config.NARA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        input_t, output_t = _extract_openai_usage(response)
        return _ProviderCall(text=text, input_tokens=input_t, output_tokens=output_t)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def call(self, prompt: str, timeout: int = None) -> dict:
        """
        Tries each provider in order until one succeeds.
        Each provider gets `timeout` seconds before we give up and move to the next.

        Returns a dict with:
            text           — the model's response text
            provider       — which provider served it ("gemini" | "groq" | ...)
            input_tokens   — prompt tokens (0 when served from cache)
            output_tokens  — completion tokens (0 when served from cache)
            total_tokens   — input + output
            cached         — True only when served from the cache

        Raises RuntimeError if every provider fails.
        """
        cache_key = None
        if self._cache is not None:
            cache_key = compute_key(prompt)
            hit = self._cache.get(cache_key)
            if hit is not None:
                logger.info(
                    f"LLM cache HIT (provider={hit.provider!r}, "
                    f"age={hit.age_seconds:.0f}s)"
                )
                return {
                    "text": hit.response_text,
                    "provider": hit.provider or "cache",
                    "cached": True,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

        dispatch = {
            "gemini": self._call_gemini,
            "groq": self._call_groq,
            "openrouter": self._call_openrouter,
            "nararouter": self._call_nararouter,
        }

        timeout = timeout or config.LLM_TIMEOUT_SECONDS
        errors = {}
        for name in self._providers:
            logger.info(f"Trying provider: {name}")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(dispatch[name], prompt)
                try:
                    result: _ProviderCall = future.result(timeout=timeout)
                    logger.info(
                        f"Success via {name} "
                        f"({result.input_tokens} in / {result.output_tokens} out tokens)"
                    )
                    if self._cache is not None and cache_key is not None:
                        self._cache.put(
                            cache_key, prompt, result.text, provider=name,
                        )
                    return {
                        "text": result.text,
                        "provider": name,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "total_tokens": result.total_tokens,
                    }
                except FutureTimeoutError:
                    logger.warning(f"{name} timed out after {timeout}s")
                    errors[name] = f"timed out after {timeout}s"
                except Exception as e:
                    logger.warning(f"{name} failed: {e}")
                    errors[name] = str(e)

        raise RuntimeError(f"All providers failed: {errors}")


if __name__ == "__main__":
    router = LLMRouter()
    result = router.call("Reply with one short sentence confirming you are working.")
    print(f"\nProvider used: {result['provider']}")
    print(f"Response: {result['text']}")
    print(f"Tokens:   {result['input_tokens']} in / {result['output_tokens']} out")