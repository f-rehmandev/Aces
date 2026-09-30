import json
import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv

from src.extractor.html_cleaner import clean_html
from src.llm.router import LLMRouter
from src import config

load_dotenv()

logger = logging.getLogger("data_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DataExtractor:
    """
    Uses an LLM to discover a schema and extract structured data from raw HTML,
    without any hardcoded CSS selectors.

    Token usage is accumulated across every call this instance makes so
    the pipeline can meter LLM cost per run (§41.4). Callers should call
    `reset_usage()` before starting a new run if the extractor instance
    is reused.
    """

    def __init__(self, router: LLMRouter = None):
        self.router = router or LLMRouter()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        # Vision is metered as a distinct resource type (§41.4 — VISION_CALL
        # is per-invocation, not per-token). `extract_from_image` increments
        # this on every successful response from the multimodal model.
        self.vision_call_count = 0

    # ------------------------------------------------------------------
    # Usage metering (§41.4)
    # ------------------------------------------------------------------
    def _record_usage(self, result: dict) -> None:
        """Accumulate token counts from a single router.call() result."""
        self.total_input_tokens += int(result.get("input_tokens", 0) or 0)
        self.total_output_tokens += int(result.get("output_tokens", 0) or 0)
        self.call_count += 1

    def reset_usage(self) -> None:
        """Zero the running counters. Called by PipelineRunner at run start."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.vision_call_count = 0

    # ------------------------------------------------------------------
    # Single-record extraction
    # ------------------------------------------------------------------
    def extract(self, html: str, instruction: str) -> dict:
        trimmed_html = clean_html(html)[:config.HTML_TRUNCATE_SINGLE]

        prompt = f"""You are a data extraction engine. Given raw HTML, extract the requested data.

Instruction: {instruction}

HTML:
{trimmed_html}

Respond with ONLY valid JSON. No explanation, no markdown code fences, just the raw JSON object."""

        result = self.router.call(prompt)
        self._record_usage(result)
        raw_text = result["text"]

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {result['provider']}: {raw_text}")
            raise ValueError(f"LLM did not return valid JSON: {e}")

        logger.info(f"Extracted via {result['provider']}: {data}")
        return data

    # ------------------------------------------------------------------
    # List extraction
    # ------------------------------------------------------------------
    def extract_list(self, html: str, instruction: str) -> list[dict]:
        trimmed_html = clean_html(html)[:config.HTML_TRUNCATE_LIST]

        prompt = f"""You are a data extraction engine. Given raw HTML containing multiple items, extract the requested data for EVERY item found.

Instruction: {instruction}

HTML:
{trimmed_html}

Respond with ONLY a valid JSON array of objects. No explanation, no markdown code fences, just the raw JSON array."""

        result = self.router.call(prompt)
        self._record_usage(result)
        raw_text = result["text"]

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON list from {result['provider']}: {raw_text}")
            raise ValueError(f"LLM did not return valid JSON array: {e}")

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got: {type(data)}")

        logger.info(f"Extracted {len(data)} items via {result['provider']}")
        return data

    # ------------------------------------------------------------------
    # Vision fallback (bypasses the router — token metering not wired yet)
    # ------------------------------------------------------------------
    def extract_from_image(self, image_bytes: bytes, instruction: str) -> list[dict]:
        """
        Vision-based extraction fallback: sends a screenshot directly to Gemini
        when text-based HTML extraction isn't reliable (heavy JS, canvas rendering,
        content baked into images). Used for self-healing when normal extraction fails.

        Metering note: this path calls genai directly rather than through the
        router, so its *token* usage is not included in `total_input_tokens` /
        `total_output_tokens`. It is metered as a distinct resource type — each
        invocation increments `vision_call_count`, which the pipeline writes
        as a VISION_CALL usage event (§41.4).
        """
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-3.5-flash-lite")

        prompt = f"""You are looking at a screenshot of a webpage. {instruction}
Return ONLY a valid JSON array of objects. No explanation, no markdown fences."""

        image_part = {"mime_type": "image/png", "data": image_bytes}
        response = model.generate_content([prompt, image_part])
        # Count the invocation immediately — the model was billed even if
        # the JSON parse below fails.
        self.vision_call_count += 1
        raw_text = response.text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vision JSON: {raw_text}")
            raise ValueError(f"Vision model did not return valid JSON: {e}")

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array from vision extraction, got: {type(data)}")

        logger.info(f"Vision-extracted {len(data)} items")
        return data


if __name__ == "__main__":
    sample_html = """
    <html><body>
        <div class="product">
            <h2>Wireless Mouse</h2>
            <span class="price">$24.99</span>
            <span class="rating">4.5 stars</span>
        </div>
        <div class="product">
            <h2>Mechanical Keyboard</h2>
            <span class="price">$75.00</span>
            <span class="rating">4.8 stars</span>
        </div>
    </body></html>
    """

    extractor = DataExtractor()

    single_data = extractor.extract(
        html=sample_html,
        instruction="Extract the first product name, price, and rating as JSON with keys: name, price, rating"
    )
    print("\nExtracted single data:")
    print(json.dumps(single_data, indent=2))

    list_data = extractor.extract_list(
        html=sample_html,
        instruction="Extract all products. For each, get the name, price, and rating. Keys should be: name, price, rating"
    )
    print("\nExtracted list data:")
    print(json.dumps(list_data, indent=2))

    print(f"\nUsage: {extractor.call_count} call(s), "
          f"{extractor.total_input_tokens} in / {extractor.total_output_tokens} out")