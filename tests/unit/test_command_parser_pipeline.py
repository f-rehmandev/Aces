"""Contract tests for the enriched parse_command (§11.1)."""
import json
import pytest

from src.assistant_core.command_parser import parse_command


class _FakeRouter:
    def __init__(self, payload=None):
        self._payload = payload or {
            "search_query": "wireless mouse prices",
            "objective": "compare",
            "min_records": 5,
            "fields": ["title", "price"],
            "filters": "",
            "location": "",
            "entity_name": "wireless mouse",
            "ambiguity_note": "",
        }

    def call(self, prompt, timeout=None):
        self.last_prompt = prompt
        return {"text": json.dumps(self._payload), "provider": "fake"}


def test_language_is_set_from_prompt():
    spec = parse_command("find laptop prices", router=_FakeRouter())
    assert spec.constraints.language == "latin"


def test_output_format_from_prompt():
    spec = parse_command("find prices, output as csv", router=_FakeRouter())
    assert spec.output.format == "csv"


def test_output_format_default_when_not_specified():
    spec = parse_command("find laptop prices", router=_FakeRouter())
    assert spec.output.format == "xlsx"


def test_navigation_pages_from_prompt():
    spec = parse_command("get the first 3 pages of jobs", router=_FakeRouter())
    assert spec.navigation.max_pages == 3
    assert spec.navigation.pagination == "auto"


def test_navigation_defaults():
    spec = parse_command("find laptop prices", router=_FakeRouter())
    assert spec.navigation.max_pages == 5
    assert spec.navigation.pagination == "none"


def test_detail_pages_inferred_from_prompt():
    spec = parse_command(
        "get the detail page for each item", router=_FakeRouter()
    )
    assert spec.navigation.detail_pages == "required"


def test_compliance_refuses_bypass_prompt():
    spec = parse_command(
        "bypass the login wall on example.com and scrape products",
        router=_FakeRouter(),
    )
    assert spec.compliance.refusal_reason != ""


def test_compliance_allows_normal_prompt():
    spec = parse_command("scrape public prices", router=_FakeRouter())
    assert spec.compliance.refusal_reason == ""


def test_pii_is_redacted_before_llm_call():
    router = _FakeRouter()
    parse_command("email me at alice@example.com", router=router)
    # The fake router captured the prompt; PII should not be in it.
    assert "alice@example.com" not in router.last_prompt
    assert "<EMAIL_1>" in router.last_prompt


def test_original_prompt_preserved_in_spec():
    spec = parse_command("email me at alice@example.com", router=_FakeRouter())
    # Spec keeps the raw prompt for audit (§11.2).
    assert spec.natural_language_prompt == "email me at alice@example.com"


def test_field_types_are_inferred():
    payload = {
        "search_query": "x",
        "fields": ["title", "price", "email", "phone", "product_url"],
        "objective": "extract",
        "min_records": 5,
        "filters": "",
        "location": "",
        "entity_name": "thing",
        "ambiguity_note": "",
    }
    spec = parse_command("find things", router=_FakeRouter(payload))
    types = {f.name: f.type for f in spec.fields}
    assert types["title"] == "text"
    assert types["price"] == "currency"
    assert types["email"] == "email"
    assert types["phone"] == "phone"
    assert types["product_url"] == "url"