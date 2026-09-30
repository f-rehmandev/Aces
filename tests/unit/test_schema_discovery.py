"""Unit tests for SchemaDiscovery (spec §12)."""
import asyncio
import json
import pytest

from src.discovery.schema_discovery import (
    SchemaDiscovery, DiscoveryReport,
    _coerce_report, _strip_code_fences, _build_prompt,
)


class _FakeRouter:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None
    def call(self, prompt, timeout=None):
        self.last_prompt = prompt
        return {"text": json.dumps(self.payload), "provider": "fake"}


class _TextRouter:
    def __init__(self, text):
        self.text = text
    def call(self, prompt, timeout=None):
        return {"text": self.text, "provider": "fake"}


# --- happy path --------------------------------------------------------

def test_quick_listing_report():
    payload = {
        "record_boundary": "listing",
        "estimated_record_count": 20,
        "field_availability": {"title": "likely_present", "price": "likely_present"},
        "confidence": 0.9,
    }
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title", "price"])
    assert report.record_boundary == "listing"
    assert report.estimated_record_count == 20
    assert report.confidence == 0.9
    assert report.is_usable()
    assert report.warnings == []


def test_single_record_boundary_is_usable():
    payload = {"record_boundary": "single_record", "confidence": 0.8}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.is_usable()


# --- not usable --------------------------------------------------------

def test_not_relevant_is_not_usable():
    payload = {"record_boundary": "not_relevant", "confidence": 0.05}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert not report.is_usable()


def test_low_confidence_is_not_usable():
    payload = {"record_boundary": "listing", "confidence": 0.1}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert not report.is_usable()


# --- coercions / robustness -------------------------------------------

def test_missing_fields_get_uncertain_default():
    payload = {"record_boundary": "listing", "confidence": 0.5}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title", "price"])
    assert report.field_availability["title"] == "uncertain"
    assert report.field_availability["price"] == "uncertain"


def test_invalid_boundary_warned_and_downgraded():
    payload = {"record_boundary": "weird_value", "confidence": 0.9}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.record_boundary == "unknown"
    assert any("record_boundary" in w for w in report.warnings)
    assert not report.is_usable()


def test_invalid_count_warned_and_zeroed():
    payload = {"record_boundary": "listing",
               "estimated_record_count": "many", "confidence": 0.9}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.estimated_record_count == 0
    assert any("record_count" in w for w in report.warnings)


def test_confidence_clamped_high():
    payload = {"record_boundary": "listing", "confidence": 5.0}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.confidence == 1.0


def test_confidence_clamped_low():
    payload = {"record_boundary": "listing", "confidence": -0.5}
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.confidence == 0.0


def test_invalid_availability_value_degrades_to_uncertain():
    payload = {
        "record_boundary": "listing",
        "confidence": 0.5,
        "field_availability": {"title": "NOPE"},
    }
    d = SchemaDiscovery(router=_FakeRouter(payload))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.field_availability["title"] == "uncertain"


# --- malformed LLM output ---------------------------------------------

def test_malformed_json_returns_unusable_report():
    d = SchemaDiscovery(router=_TextRouter("this is not json"))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.record_boundary == "unknown"
    assert report.confidence == 0.0
    assert any("JSON" in w for w in report.warnings)


def test_json_array_instead_of_object_returns_unusable():
    d = SchemaDiscovery(router=_TextRouter("[1, 2, 3]"))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.confidence == 0.0


def test_code_fenced_json_is_unwrapped():
    payload_text = "```json\n" + json.dumps({
        "record_boundary": "listing", "confidence": 0.7,
    }) + "\n```"
    d = SchemaDiscovery(router=_TextRouter(payload_text))
    report = d.discover_from_html("<html></html>", "product", ["title"])
    assert report.record_boundary == "listing"
    assert report.confidence == 0.7


# --- empty HTML --------------------------------------------------------

def test_empty_html_short_circuits_before_llm():
    d = SchemaDiscovery(router=_FakeRouter({"record_boundary": "listing"}))
    report = d.discover_from_html("", "product", ["title"])
    assert report.confidence == 0.0
    assert any("empty" in w.lower() for w in report.warnings)


# --- manual mode -------------------------------------------------------

def test_manual_returns_full_confidence():
    d = SchemaDiscovery(router=_FakeRouter({}))
    report = d.manual(["title", "price"])
    assert report.confidence == 1.0
    assert report.record_boundary == "listing"
    assert report.field_availability == {
        "title": "likely_present", "price": "likely_present",
    }


def test_manual_rejects_invalid_boundary():
    d = SchemaDiscovery(router=_FakeRouter({}))
    report = d.manual(["title"], record_boundary="garbage")
    assert report.record_boundary == "listing"


# --- async fetcher -----------------------------------------------------

def test_discover_url_uses_fetcher():
    async def fake_fetch(url):
        return "<html><body>hi</body></html>"
    d = SchemaDiscovery(
        router=_FakeRouter({"record_boundary": "listing", "confidence": 0.9}),
        fetcher=fake_fetch,
    )
    report = asyncio.run(d.discover_url("https://x.com", "thing", ["title"]))
    assert report.record_boundary == "listing"


def test_discover_url_without_fetcher_raises():
    d = SchemaDiscovery(router=_FakeRouter({}))
    with pytest.raises(RuntimeError):
        asyncio.run(d.discover_url("https://x.com", "thing", ["title"]))


# --- helpers -----------------------------------------------------------

def test_strip_code_fences_plain():
    assert _strip_code_fences("hello") == "hello"


def test_strip_code_fences_json_block():
    assert _strip_code_fences("```json\n{}\n```") == "{}"


def test_strip_code_fences_bare_block():
    assert _strip_code_fences("```\n{}\n```") == "{}"


def test_prompt_contains_untrusted_markers():
    prompt = _build_prompt("product", ["title"], "<div>x</div>")
    assert "<untrusted_content>" in prompt
    assert "</untrusted_content>" in prompt
    assert "product" in prompt
    assert "title" in prompt


def test_coerce_report_handles_non_dict():
    report = _coerce_report("not a dict", ["title"])
    assert report.record_boundary == "unknown"
    assert any("non-object" in w.lower() for w in report.warnings)