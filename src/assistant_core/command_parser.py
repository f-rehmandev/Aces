import json
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm.router import LLMRouter
from core.task_spec import TaskSpec

logger = logging.getLogger("command_parser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_command(user_request: str, router: LLMRouter = None) -> TaskSpec:
    """
    Converts plain English into a real TaskSpec (project knowledge Section 11).
    If the request is materially ambiguous, the LLM is asked to note that in
    `ambiguity_note` rather than silently guessing on something that matters.
    """
    router = router or LLMRouter()

    prompt = f"""You are a task-planning assistant for a lead-generation and research tool.
Given a user's plain-English request, extract a structured task specification.

User request: "{user_request}"

Respond with ONLY valid JSON in this exact shape:
{{
  "search_query": "a good web search query to find relevant sources for this request",
  "objective": "extract" or "compare" or "lead_gen" or "monitor",
  "min_records": <integer, how many results the user wants, default 10 if not specified>,
  "fields": ["list", "of", "data", "fields", "to", "extract", "per", "result"],
  "filters": "any filtering criteria, niche, email or special conditions mentioned",
  "location": "a specific city/region/country mentioned in the request, or empty string if none",
  "entity_name": "the single core business/product/job type this request is about, e.g. 'pizza shop', 'wireless mouse', 'freelance job'",
  "ambiguity_note": "if this request is materially under-specified (e.g. missing WHICH sites, WHICH country, no clear entity), briefly say what's ambiguous here. Otherwise empty string."
}}

For requests about finding multiple businesses (leads), craft a natural search_query
likely to surface pages listing many businesses at once — e.g. "best pizza shops [city]"
or "pizza shops near me list" — similar in style to how you'd search for product reviews
or best-of lists. Do NOT use search operators like "site:" — write a normal, natural
search phrase a person would actually type.

Common field examples depending on request type: for businesses/leads use things like
business_name, phone, email, address, has_website. For products use title, price,
availability. For jobs use title, price (rate), description, employer.
Pick whatever fields genuinely make sense for THIS specific request."""

    result = router.call(prompt)
    raw_text = result["text"].strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse command spec: {raw_text}")
        raise ValueError(f"Could not parse command: {e}")

    ambiguity_note = parsed.get("ambiguity_note", "")
    if ambiguity_note:
        logger.warning(f"Ambiguous request: {ambiguity_note}")

    spec = TaskSpec(
        natural_language_prompt=user_request,
        source_hint=parsed.get("search_query", ""),
        objective=parsed.get("objective", "extract"),
        entity_name=parsed.get("entity_name", ""),
        fields=parsed.get("fields", ["title", "price", "description"]),
        filters=parsed.get("filters", ""),
        location=parsed.get("location", ""),
        min_records=parsed.get("min_records", 10),
    )
    spec.ambiguity_note = ambiguity_note  # dynamic attribute, not persisted in schema yet

    logger.info(f"Parsed command via {result['provider']}: {spec.to_dict()}")
    return spec


if __name__ == "__main__":
    test_requests = [
        "generate 15 leads for a website maker, pizza shops niche, need phone, address, email and whether they have a website or not",
        "find best 10 prices of a Logitech wireless mouse",
        "get me leads",  # deliberately vague, should trigger ambiguity_note
    ]
    for req in test_requests:
        print(f"\nRequest: {req}")
        spec = parse_command(req)
        print(spec.to_json())
        if spec.ambiguity_note:
            print(f"⚠ Ambiguity flagged: {spec.ambiguity_note}")