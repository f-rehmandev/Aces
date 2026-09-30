"""Unit tests for JsonTaskFileResolver (spec §10, §55)."""
import json
from pathlib import Path

import pytest

from src.intake.json_task import JsonTaskFileResolver, MalformedTaskFile


GOOD = {
    "natural_language_prompt": "track laptop prices",
    "objective": "monitor",
    "target": {"start_urls": ["https://x.com/a"], "source_hint": "laptops"},
    "fields": [{"name": "title", "type": "text"}],
    "budget": {"max_usd": 1.50},
}


def test_loads_valid_spec_from_string():
    r = JsonTaskFileResolver().resolve(json.dumps(GOOD))
    assert r.spec.objective == "monitor"
    assert r.spec.target.start_urls == ["https://x.com/a"]
    assert r.spec.target.source_hint == "laptops"
    assert r.spec.budget.max_usd == 1.50
    assert r.warnings == []


def test_loads_from_file(tmp_path: Path):
    p = tmp_path / "task.json"
    p.write_text(json.dumps(GOOD), encoding="utf-8")
    r = JsonTaskFileResolver().resolve(p)
    assert r.spec.objective == "monitor"
    assert "task.json" in r.source_description


def test_invalid_json_raises():
    with pytest.raises(MalformedTaskFile):
        JsonTaskFileResolver().resolve("{not valid json")


def test_non_object_top_level_raises():
    with pytest.raises(MalformedTaskFile):
        JsonTaskFileResolver().resolve('["array", "not", "object"]')


def test_unknown_key_rejected_in_strict_mode():
    bad = json.dumps({"totally_unknown": "value"})
    with pytest.raises(MalformedTaskFile):
        JsonTaskFileResolver(strict=True).resolve(bad)


def test_unknown_key_allowed_with_warning_in_non_strict():
    bad = json.dumps({"totally_unknown": "value", "objective": "extract"})
    r = JsonTaskFileResolver(strict=False).resolve(bad)
    assert r.spec.objective == "extract"
    assert any("ignored" in w for w in r.warnings)


def test_nested_groups_are_reconstructed():
    data = {
        "natural_language_prompt": "x",
        "target": {"start_urls": ["https://x.com/a"], "domains": ["x.com"]},
        "quality": {"min_records": 5},
        "compliance": {"user_authorization_declared": True},
        "budget": {"max_usd": 2.0, "max_pages": 100},
    }
    r = JsonTaskFileResolver().resolve(json.dumps(data))
    assert r.spec.target.domains == ["x.com"]
    assert r.spec.quality.min_records == 5
    assert r.spec.compliance.user_authorization_declared is True
    assert r.spec.budget.max_pages == 100


def test_unsupported_input_type_raises():
    with pytest.raises(TypeError):
        JsonTaskFileResolver().resolve(12345)


def test_source_description_for_inline():
    r = JsonTaskFileResolver().resolve(json.dumps(GOOD))
    assert "inline" in r.source_description.lower()

    

def test_accepts_already_parsed_dict():
    r = JsonTaskFileResolver().resolve({"objective": "monitor"})
    assert r.spec.objective == "monitor"
    assert "dict" in r.source_description.lower()


def test_dict_still_rejects_unknown_keys_in_strict_mode():
    with pytest.raises(MalformedTaskFile):
        JsonTaskFileResolver(strict=True).resolve({"totally_unknown": True})