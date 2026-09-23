"""
Autonomous Schema Discovery — project knowledge Section 12.
Quick mode: one representative page, LLM proposes record boundary + field
candidates with a confidence note, before any real extraction commits to them.
"""

import json
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm.router import LLMRouter
from extractor.html_cleaner import clean_html
import config

logger = logging.getLogger("schema_discovery")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def quick_discover(html: str, entity_name: str, requested_fields: list[str], router: LLMRouter = None) -> dict:
    """
    Quick-mode schema discovery: samples the page once, proposes whether it's
    a listing (repeated records) or a single-record page, and flags which
    requested fields actually look findable here before extraction runs.
    """
    router = router or LLMRouter()
    trimmed_html = clean_html(html)[:config.HTML_TRUNCATE_LIST]

    prompt = f"""You are a schema discovery engine, looking at a web page BEFORE extraction runs.
The user wants to find items of type: '{entity_name}'.
Requested fields: {requested_fields}

HTML:
{trimmed_html}

Respond with ONLY valid JSON:
{{
  "record_boundary": "listing" or "single_record" or "not_relevant",
  "estimated_record_count": <integer, rough guess how many matching items are on this page>,
  "field_availability": {{"<field_name>": "likely_present" or "likely_absent" or "uncertain", ...}},
  "confidence": <float 0-1, how confident you are this page is usable for this request>
}}"""

    result = router.call(prompt)
    raw_text = result["text"].strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        report = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Discovery JSON parse failed: {raw_text}")
        raise ValueError(f"Discovery did not return valid JSON: {e}")

    logger.info(f"Discovery report via {result['provider']}: {report}")
    return report


if __name__ == "__main__":
    sample_html = """
    <html><body>
        <div class="listing">
            <div class="item"><h2>Tony's Pizza</h2><span>123 Main St</span></div>
            <div class="item"><h2>Marco's Pizzeria</h2><span>456 Oak Ave</span></div>
            <div class="item"><h2>Bella Napoli</h2><span>789 Elm St</span></div>
        </div>
    </body></html>
    """
    report = quick_discover(
        sample_html,
        entity_name="pizza shop",
        requested_fields=["business_name", "address", "phone", "has_website"],
    )
    print(json.dumps(report, indent=2))