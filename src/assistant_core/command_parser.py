import json
import logging

from src.llm.router import LLMRouter
from src.core.task_spec import (
    TaskSpec, Target, EntitySpec, FieldSpec,
    Constraints, Quality, Navigation, Output, Compliance,
)
from src.intake.intent_helpers import (
    infer_navigation, infer_output_format, infer_compliance,
    make_safe_prompt, infer_language,
)

logger = logging.getLogger("command_parser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_command(user_request: str, router: LLMRouter = None) -> TaskSpec:
    """
    Converts plain English into a real TaskSpec (spec §11).

    Pipeline steps (§11.1):
        - language detection
        - PII redaction before the LLM call
        - LLM parses entities / fields / constraints / search query
        - deterministic inference for navigation, output format, compliance
    """
    router = router or LLMRouter()

    # --- §11.1 step 1: language detection ---
    language = infer_language(user_request)

    # --- §11.1 step 2: PII redaction before the LLM sees the prompt ---
    safe = make_safe_prompt(user_request)
    if safe.redaction.replacements:
        logger.info(
            f"Redacted {len(safe.redaction.replacements)} PII item(s) from prompt"
        )

    prompt = f"""You are a task-planning assistant for a lead-generation and research tool.
Given a user's plain-English request, extract a structured task specification.

User request (PII has been redacted to tokens like <EMAIL_1>, <PHONE_1> — treat them as opaque placeholders):
"{safe.redacted}"

Respond with ONLY valid JSON in this exact shape:
{{
  "search_query": "a good web search query to find relevant sources for this request",
  "objective": "extract" or "compare" or "lead_gen" or "monitor",
  "min_records": <integer, how many results the user wants, default 10 if not specified>,
  "fields": ["list", "of", "data", "fields", "to", "extract", "per", "result"],
  "filters": "any filtering criteria, niche, email or special conditions mentioned",
  "location": "a specific city/region/country mentioned in the request, or empty string if none",
  "entity_name": "the single core business/product/job type this request is about, e.g. 'pizza shop', 'wireless mouse', 'freelance job'",
  "ambiguity_note": "if this request is materially under-specified, briefly say what's ambiguous. Otherwise empty string."
}}

For requests about finding multiple businesses (leads), craft a natural search_query
likely to surface pages listing many businesses at once — e.g. "best pizza shops [city]".

Common field examples: for businesses/leads use things like business_name, phone, email,
address, has_website. For products use title, price, availability. For jobs use title,
price (rate), description, employer. Pick whatever fields genuinely make sense."""

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

    # --- heuristic field typing (§11.2) ---
    def _guess_type(name: str) -> str:
        n = name.lower()
        if "price" in n or "cost" in n:            return "currency"
        if "email" in n:                            return "email"
        if "phone" in n or "tel" in n:              return "phone"
        if "url" in n or "website" in n or "link" in n: return "url"
        if "date" in n:                             return "date"
        return "text"

    raw_fields = parsed.get("fields", ["title", "price", "description"])
    field_specs = [FieldSpec(name=f, type=_guess_type(f)) for f in raw_fields]

    # If the user asked for a price, auto-add `quantity` so the extractor
    # tries to pull package size from the page. Harmless if the page has
    # no quantity — the field will just be null.
    if any("price" in f.lower() or "cost" in f.lower() for f in raw_fields):
        if not any(f.name == "quantity" for f in field_specs):
            field_specs.append(FieldSpec(name="quantity", type="text"))

    # --- §11.1 step 9: compliance pre-screen ---
    comp = infer_compliance(user_request)
    if not comp.allowed:
        logger.warning(f"Request refused at pre-screen: {comp.refusal_reason}")

    # --- §11.1 step 6: navigation inference ---
    nav = infer_navigation(user_request, llm_hints=parsed.get("navigation"))

    # --- §11.1 step 7: output format inference ---
    output_format = infer_output_format(
        user_request, default=parsed.get("output_format", "xlsx")
    )

    spec = TaskSpec(
        natural_language_prompt=user_request,
        target=Target(source_hint=parsed.get("search_query", "")),
        objective=parsed.get("objective", "extract"),
        entities=[EntitySpec(entity_name=parsed.get("entity_name", ""))],
        fields=field_specs,
        constraints=Constraints(
            filters=parsed.get("filters", ""),
            geography=parsed.get("location", ""),
            language=language,
        ),
        navigation=Navigation(
            pagination=nav.pagination,
            detail_pages=nav.detail_pages,
            max_pages=nav.max_pages,
        ),
        output=Output(format=output_format),
        compliance=Compliance(
            user_authorization_declared=comp.user_authorization_declared,
            refusal_reason=comp.refusal_reason,
        ),
        quality=Quality(min_records=parsed.get("min_records", 10)),
    )
    spec.ambiguity_note = ambiguity_note

    logger.info(f"Parsed command via {result['provider']}: {spec.to_dict()}")
    return spec


if __name__ == "__main__":
    test_requests = [
        "generate 15 leads for a website maker, pizza shops niche, need phone, address, email and whether they have a website or not",
        "find best 10 prices of a Logitech wireless mouse, output as csv",
        "get me the first 3 pages of freelance web scraping jobs",
    ]
    for req in test_requests:
        print(f"\nRequest: {req}")
        spec = parse_command(req)
        print(f"  objective: {spec.objective}")
        print(f"  fields:    {spec.field_names}")
        print(f"  language:  {spec.constraints.language}")
        print(f"  output:    {spec.output.format}")
        print(f"  pages:     {spec.navigation.max_pages} (pagination={spec.navigation.pagination})")
        if spec.compliance.refusal_reason:
            print(f"  REFUSED:   {spec.compliance.refusal_reason}")