"""Unit tests for NaturalLanguageResolver (spec §10)."""
import json
import pytest

from src.intake.nl_resolver import NaturalLanguageResolver
from src.intake.resolver import InputResolution


class _FakeRouter:
    def __init__(self, payload):
        self.payload = payload

    def call(self, prompt, timeout=None):
        return {"text": json.dumps(self.payload), "provider": "fake"}


BASE_PAYLOAD = {
    "search_query": "wireless mouse prices",
    "objective": "compare",
    "min_records": 5,
    "fields": ["title", "price"],
    "filters": "",
    "location": "",
    "entity_name": "wireless mouse",
    "ambiguity_note": "",
}


def test_basic_resolution():
    r = NaturalLanguageResolver(router=_FakeRouter(BASE_PAYLOAD)).resolve(
        "find best prices for wireless mouse"
    )
    assert isinstance(r, InputResolution)
    assert r.spec.objective == "compare"
    # Price queries automatically get a `quantity` field so we can
    # compute per-unit pricing later. That's intentional.
    assert r.spec.field_names == ["title", "price", "quantity"]
    assert r.spec.quality.min_records == 5
    assert r.warnings == []


def test_ambiguity_note_becomes_warning():
    payload = dict(BASE_PAYLOAD, ambiguity_note="which country?")
    r = NaturalLanguageResolver(router=_FakeRouter(payload)).resolve("find shops")
    assert any("ambiguous" in w.lower() for w in r.warnings)


def test_rejects_empty_string():
    with pytest.raises(ValueError):
        NaturalLanguageResolver(router=_FakeRouter(BASE_PAYLOAD)).resolve("")


def test_rejects_non_string():
    with pytest.raises(ValueError):
        NaturalLanguageResolver(router=_FakeRouter(BASE_PAYLOAD)).resolve(None)


def test_source_description_truncates_long_prompts():
    long = "a" * 200
    r = NaturalLanguageResolver(router=_FakeRouter(BASE_PAYLOAD)).resolve(long)
    assert r.source_description.endswith("...")
    assert len(r.source_description) < 120