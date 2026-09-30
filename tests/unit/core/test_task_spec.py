"""Unit tests for TaskSpec (spec §11.2, §62)."""
from src.core.task_spec import (
    TaskSpec, Target, EntitySpec, FieldSpec,
    Constraints, Quality, Budget, Compliance,
)


def test_default_construction_is_valid():
    spec = TaskSpec()
    assert spec.schema_version == "2.0"
    assert spec.objective == "extract"
    assert spec.budget.max_usd == 0.50
    assert spec.compliance.robots_policy == "respect"


def test_backward_compat_properties():
    spec = TaskSpec(
        target=Target(source_hint="search this"),
        entities=[EntitySpec(entity_name="pizza shop")],
        fields=[FieldSpec(name="title"), FieldSpec(name="price", type="currency")],
        quality=Quality(min_records=7),
    )
    assert spec.source_hint == "search this"
    assert spec.entity_name == "pizza shop"
    assert spec.field_names == ["title", "price"]
    assert spec.min_records == 7


def test_round_trip_through_dict():
    original = TaskSpec(
        natural_language_prompt="find cheap laptops",
        objective="compare",
        target=Target(start_urls=["https://example.com"], source_hint="laptops"),
        entities=[EntitySpec(entity_name="laptop", identity_hint="sku")],
        fields=[FieldSpec(name="title"), FieldSpec(name="price", type="currency", required=True)],
        constraints=Constraints(geography="Lahore"),
        quality=Quality(min_records=20),
        budget=Budget(max_usd=2.0, max_pages=100),
        compliance=Compliance(user_authorization_declared=True),
    )
    restored = TaskSpec.from_dict(original.to_dict())

    assert restored.natural_language_prompt == original.natural_language_prompt
    assert restored.target.start_urls == ["https://example.com"]
    assert restored.entities[0].identity_hint == "sku"
    assert restored.fields[1].type == "currency"
    assert restored.fields[1].required is True
    assert restored.constraints.geography == "Lahore"
    assert restored.quality.min_records == 20
    assert restored.budget.max_usd == 2.0
    assert restored.compliance.user_authorization_declared is True