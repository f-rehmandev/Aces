from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
import logging
import sys
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


load_dotenv()
logger = logging.getLogger("llm_router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LLMRouter:
    """
    Provider-agnostic LLM caller. Tries providers in order until one succeeds.
    This is the single entry point the rest of the project uses to talk to any LLM.
    """

    def __init__(self):
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

        self._providers = config.LLM_PROVIDER_ORDER

    def _call_gemini(self, prompt: str) -> str:
        response = self._gemini_model.generate_content(prompt)
        return response.text.strip()

    def _call_groq(self, prompt: str) -> str:
        response = self._groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _call_openrouter(self, prompt: str) -> str:
        response = self._openrouter_client.chat.completions.create(
        model=config.OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def call(self, prompt: str, timeout: int = None) -> dict:
        """
        Tries each provider in order until one succeeds.
        Each provider gets `timeout` seconds before we give up and move to the next.
        Returns {"text": ..., "provider": ...} on success.
        Raises RuntimeError if every provider fails.
        """
        dispatch = {
            "gemini": self._call_gemini,
            "groq": self._call_groq,
            "openrouter": self._call_openrouter,
        }

        timeout = timeout or config.LLM_TIMEOUT_SECONDS
        errors = {}
        for name in self._providers:
            logger.info(f"Trying provider: {name}")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(dispatch[name], prompt)
                try:
                    text = future.result(timeout=timeout)
                    logger.info(f"Success via {name}")
                    return {"text": text, "provider": name}
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