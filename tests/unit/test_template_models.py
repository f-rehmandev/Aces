"""Unit tests for template data model (spec §10A)."""
import pytest

from src.templates.models import (
    InputParameter,
    ParameterType,
    TaskArchetype,
    Template,
    TemplateStatus,
    ValidationRules,
)


# ---------------------------------------------------------------------------
# InputParameter
# ---------------------------------------------------------------------------

def test_parameter_valid_name():
    ok, _ = InputParameter(name="city").validates()
    assert ok


def test_parameter_allows_digits_and_underscores():
    ok, _ = InputParameter(name="max_pages_2").validates()
    assert ok


def test_parameter_rejects_uppercase():
    ok, reason = InputParameter(name="City").validates()
    assert not ok
    assert "snake_case" in reason


def test_parameter_rejects_leading_digit():
    ok, _ = InputParameter(name="1city").validates()
    assert not ok


def test_parameter_rejects_empty():
    ok, _ = InputParameter(name="").validates()
    assert not ok


def test_parameter_rejects_special_chars():
    ok, _ = InputParameter(name="city-name").validates()
    assert not ok


def test_enum_without_choices_invalid():
    ok, reason = InputParameter(
        name="sort", type=ParameterType.ENUM,
    ).validates()
    assert not ok
    assert "choices" in reason


def test_enum_with_choices_valid():
    ok, _ = InputParameter(
        name="sort",
        type=ParameterType.ENUM,
        choices=["asc", "desc"],
    ).validates()
    assert ok


def test_parameter_round_trip():
    p = InputParameter(
        name="count",
        type=ParameterType.INTEGER,
        description="how many",
        required=False,
        default=10,
    )
    d = p.to_dict()
    assert d["type"] == "integer"
    assert d["default"] == 10

    p2 = InputParameter.from_dict(d)
    assert p2.name == "count"
    assert p2.type == ParameterType.INTEGER
    assert p2.required is False


# ---------------------------------------------------------------------------
# ValidationRules
# ---------------------------------------------------------------------------

def test_validation_rules_defaults():
    vr = ValidationRules()
    assert vr.require_min_fields == 1
    assert vr.forbid_login_walls is True
    assert vr.require_authorization_declared is False


def test_validation_rules_round_trip():
    vr = ValidationRules(
        require_min_fields=3,
        require_authorization_declared=True,
        max_records_per_run=500,
    )
    vr2 = ValidationRules.from_dict(vr.to_dict())
    assert vr2.require_min_fields == 3
    assert vr2.require_authorization_declared is True
    assert vr2.max_records_per_run == 500


# ---------------------------------------------------------------------------
# Template: construction & defaults
# ---------------------------------------------------------------------------

def test_new_template_defaults():
    t = Template()
    assert t.status == TemplateStatus.DRAFT
    assert t.version == 1
    assert t.archetype == TaskArchetype.GENERIC
    assert t.input_parameters == []
    assert t.required_fields == []
    assert t.task_spec_defaults == {}
    assert t.security_profile == "standard"


def test_template_status_immutability():
    draft = Template()
    assert not draft.is_immutable()

    published = Template(status=TemplateStatus.PUBLISHED)
    assert published.is_immutable()


def test_template_status_selectability():
    assert not Template(status=TemplateStatus.DRAFT).is_selectable()
    assert not Template(status=TemplateStatus.TESTED).is_selectable()
    assert Template(status=TemplateStatus.APPROVED).is_selectable()
    assert Template(status=TemplateStatus.PUBLISHED).is_selectable()
    assert not Template(status=TemplateStatus.RETIRED).is_selectable()


# ---------------------------------------------------------------------------
# Template: parameter lookup
# ---------------------------------------------------------------------------

def test_get_parameter_found():
    t = Template(input_parameters=[
        InputParameter(name="city"),
        InputParameter(name="count"),
    ])
    assert t.get_parameter("city") is not None
    assert t.get_parameter("count") is not None


def test_get_parameter_missing_returns_none():
    t = Template(input_parameters=[InputParameter(name="city")])
    assert t.get_parameter("nope") is None


def test_get_parameter_on_empty_template():
    assert Template().get_parameter("anything") is None


# ---------------------------------------------------------------------------
# Template: serialization
# ---------------------------------------------------------------------------

def test_to_dict_serializes_enums():
    t = Template(
        name="pizza",
        archetype=TaskArchetype.LEAD_GENERATION,
        status=TemplateStatus.APPROVED,
    )
    d = t.to_dict()
    assert d["archetype"] == "lead_generation"
    assert d["status"] == "approved"


def test_round_trip_preserves_all_scalar_fields():
    t = Template(
        name="pizza-lead-gen",
        description="Find pizza shops",
        category="lead_gen",
        archetype=TaskArchetype.LEAD_GENERATION,
        task_schema_version="2.0",
        security_profile="lead_gen",
        author="system",
        version=3,
    )
    t2 = Template.from_dict(t.to_dict())
    assert t2.template_id == t.template_id
    assert t2.name == "pizza-lead-gen"
    assert t2.archetype == TaskArchetype.LEAD_GENERATION
    assert t2.security_profile == "lead_gen"
    assert t2.author == "system"
    assert t2.version == 3


def test_round_trip_preserves_parameters():
    t = Template(input_parameters=[
        InputParameter(name="city", description="Target city"),
        InputParameter(name="count", type=ParameterType.INTEGER, default=20),
    ])
    t2 = Template.from_dict(t.to_dict())
    assert len(t2.input_parameters) == 2
    assert t2.input_parameters[0].name == "city"
    assert t2.input_parameters[1].type == ParameterType.INTEGER
    assert t2.input_parameters[1].default == 20


def test_round_trip_preserves_validation_rules():
    t = Template(validation_rules=ValidationRules(
        require_min_fields=2,
        max_records_per_run=100,
    ))
    t2 = Template.from_dict(t.to_dict())
    assert t2.validation_rules.require_min_fields == 2
    assert t2.validation_rules.max_records_per_run == 100


def test_round_trip_preserves_archetypes():
    t = Template(supported_archetypes=[
        TaskArchetype.LEAD_GENERATION,
        TaskArchetype.DIRECTORY,
    ])
    t2 = Template.from_dict(t.to_dict())
    assert t2.supported_archetypes == [
        TaskArchetype.LEAD_GENERATION,
        TaskArchetype.DIRECTORY,
    ]


def test_round_trip_preserves_task_spec_defaults():
    t = Template(task_spec_defaults={
        "objective": "lead_gen",
        "fields": [{"name": "business_name"}, {"name": "phone"}],
        "quality": {"min_records": 10},
    })
    t2 = Template.from_dict(t.to_dict())
    assert t2.task_spec_defaults["objective"] == "lead_gen"
    assert t2.task_spec_defaults["quality"]["min_records"] == 10


def test_from_dict_accepts_already_built_objects():
    """If dict already has InputParameter/TaskArchetype objects, don't re-wrap."""
    p = InputParameter(name="city")
    t = Template(
        input_parameters=[p],
        supported_archetypes=[TaskArchetype.GENERIC],
    )
    t2 = Template.from_dict(t.to_dict())
    assert isinstance(t2.input_parameters[0], InputParameter)
    assert isinstance(t2.supported_archetypes[0], TaskArchetype)


def test_from_dict_ignores_unknown_keys():
    d = Template().to_dict()
    d["totally_unknown_key"] = "ignored"
    t = Template.from_dict(d)
    assert t.template_id == d["template_id"]


def test_from_dict_handles_missing_optional_blocks():
    """A minimal dict should reconstruct a template with sensible defaults."""
    t = Template.from_dict({"template_id": "abc", "name": "minimal"})
    assert t.template_id == "abc"
    assert t.name == "minimal"
    assert t.input_parameters == []
    assert t.status == TemplateStatus.DRAFT
    assert t.archetype == TaskArchetype.GENERIC
    assert isinstance(t.validation_rules, ValidationRules)


# ---------------------------------------------------------------------------
# TaskArchetype / TemplateStatus enums
# ---------------------------------------------------------------------------

def test_task_archetypes_present():
    assert TaskArchetype.LEAD_GENERATION.value == "lead_generation"
    assert TaskArchetype.ECOMMERCE_LISTING.value == "ecommerce_listing"
    assert TaskArchetype.JOB_BOARD.value == "job_board"


def test_template_statuses_present():
    assert TemplateStatus.DRAFT.value == "draft"
    assert TemplateStatus.PUBLISHED.value == "published"
    assert TemplateStatus.RETIRED.value == "retired"