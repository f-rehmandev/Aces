import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
from extractor.html_cleaner import clean_html
import json
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.llm.router import LLMRouter

logger = logging.getLogger("data_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DataExtractor:
    """
    Uses an LLM to discover a schema and extract structured data from raw HTML,
    without any hardcoded CSS selectors.
    """

    def __init__(self, router: LLMRouter = None):
        self.router = router or LLMRouter()

    def extract(self, html: str, instruction: str) -> dict:
        # Keep the HTML short-ish for free-tier token limits during dev
        trimmed_html = clean_html(html)[:config.HTML_TRUNCATE_SINGLE]

        prompt = f"""You are a data extraction engine. Given raw HTML, extract the requested data.

Instruction: {instruction}

HTML:
{trimmed_html}

Respond with ONLY valid JSON. No explanation, no markdown code fences, just the raw JSON object."""

        result = self.router.call(prompt)
        raw_text = result["text"]

        # Models sometimes wrap JSON in ```json fences anyway — strip if present
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

    def extract_list(self, html: str, instruction: str) -> list[dict]:
        trimmed_html = clean_html(html)[:config.HTML_TRUNCATE_LIST]

        prompt = f"""You are a data extraction engine. Given raw HTML containing multiple items, extract the requested data for EVERY item found.

Instruction: {instruction}

HTML:
{trimmed_html}

Respond with ONLY a valid JSON array of objects. No explanation, no markdown code fences, just the raw JSON array."""

        result = self.router.call(prompt)
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


    def extract_from_image(self, image_bytes: bytes, instruction: str) -> list[dict]:
        """
        Vision-based extraction fallback: sends a screenshot directly to Gemini
        when text-based HTML extraction isn't reliable (heavy JS, canvas rendering,
        content baked into images). Used for self-healing when normal extraction fails.
        """
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-3.5-flash-lite")

        prompt = f"""You are looking at a screenshot of a webpage. {instruction}
Return ONLY a valid JSON array of objects. No explanation, no markdown fences."""

        image_part = {"mime_type": "image/png", "data": image_bytes}
        response = model.generate_content([prompt, image_part])
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
    
    # Testing the single extraction
    single_data = extractor.extract(
        html=sample_html,
        instruction="Extract the first product name, price, and rating as JSON with keys: name, price, rating"
    )
    print("\nExtracted single data:")
    print(json.dumps(single_data, indent=2))

    # Testing the list extraction
    list_data = extractor.extract_list(
        html=sample_html,
        instruction="Extract all products. For each, get the name, price, and rating. Keys should be: name, price, rating"
    )
    print("\nExtracted list data:")
    print(json.dumps(list_data, indent=2))