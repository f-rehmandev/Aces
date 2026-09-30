"""Unit tests for TemplateBuilder (spec §10A.3)."""
import pytest

from src.templates.builder import (
    InvalidParameterError,
    MissingParameterError,
    TemplateBuildError,
    TemplateBuilder,
    UnpublishedTemplateError,
    _deep_merge,
    _resolve_parameters,
    _substitute,
    build_from_template,
)
from src.templates.models import (
    InputParameter,
    ParameterType,
    TaskArchetype,
    Template,
    TemplateStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _published_template() -> Template:
    return Template(
        name="pizza-lead-gen",
        archetype=TaskArchetype.LEAD_GENERATION,
        status=TemplateStatus.PUBLISHED,
        input_parameters=[
            InputParameter(name="city", required=True),
            InputParameter(name="count", type=ParameterType.INTEGER, default=10),
            InputParameter(name="rating", type=ParameterType.FLOAT, required=False),
            InputParameter(name="strict", type=ParameterType.BOOLEAN, required=False),
            InputParameter(
                name="sort", type=ParameterType.ENUM,
                required=False, choices=["rating", "name"],
            ),
        ],
        task_spec_defaults={
            "objective": "lead_gen",
            "target": {"source_hint": "pizza shops in {city}"},
            "fields": [
                {"name": "business_name", "type": "text"},
                {"name": "phone", "type": "phone"},
            ],
            "constraints": {"geography": "{city}"},
            "quality": {"min_records": "{count}"},
        },
    )


# ---------------------------------------------------------------------------
# _substitute
# ---------------------------------------------------------------------------

def test_substitute_whole_string_preserves_int():
    assert _substitute("{count}", {"count": 10}) == 10


def test_substitute_whole_string_preserves_bool():
    assert _substitute("{flag}", {"flag": True}) is True


def test_substitute_embedded_placeholder_coerces_to_string():
    assert _substitute("hi {name}", {"name": "there"}) == "hi there"


def test_substitute_int_embedded_in_text():
    assert _substitute("page {n}", {"n": 5}) == "page 5"


def test_substitute_missing_placeholder_left_intact():
    assert _substitute("{unknown}", {"other": 1}) == "{unknown}"


def test_substitute_recurses_into_dict():
    result = _substitute(
        {"a": "{x}", "b": {"c": "hi {x}"}},
        {"x": 7},
    )
    assert result == {"a": 7, "b": {"c": "hi 7"}}


def test_substitute_recurses_into_list():
    result = _substitute(["{x}", "pre {x}"], {"x": 3})
    assert result == [3, "pre 3"]


def test_substitute_non_string_unchanged():
    assert _substitute(42, {}) == 42
    assert _substitute(None, {}) is None


# ---------------------------------------------------------------------------
# _resolve_parameters
# ---------------------------------------------------------------------------

def test_resolve_fills_defaults():
    t = _published_template()
    resolved = _resolve_parameters(t, {"city": "Lahore"})
    assert resolved["city"] == "Lahore"
    assert resolved["count"] == 10        # default


def test_resolve_rejects_missing_required():
    t = _published_template()
    with pytest.raises(MissingParameterError):
        _resolve_parameters(t, {})


def test_resolve_coerces_int_from_string():
    t = _published_template()
    resolved = _resolve_parameters(t, {"city": "Lahore", "count": "25"})
    assert resolved["count"] == 25
    assert isinstance(resolved["count"], int)


def test_resolve_rejects_non_int_string():
    t = _published_template()
    with pytest.raises(InvalidParameterError):
        _resolve_parameters(t, {"city": "Lahore", "count": "many"})


def test_resolve_rejects_bool_for_int():
    t = _published_template()
    with pytest.raises(InvalidParameterError):
        _resolve_parameters(t, {"city": "Lahore", "count": True})


def test_resolve_coerces_float():
    t = _published_template()
    resolved = _resolve_parameters(t, {"city": "Lahore", "rating": "4.5"})
    assert resolved["rating"] == 4.5
    assert isinstance(resolved["rating"], float)


def test_resolve_boolean_string():
    t = _published_template()
    resolved = _resolve_parameters(t, {"city": "Lahore", "strict": "true"})
    assert resolved["strict"] is True


def test_resolve_rejects_bad_boolean():
    t = _published_template()
    with pytest.raises(InvalidParameterError):
        _resolve_parameters(t, {"city": "Lahore", "strict": "yep"})


def test_resolve_enum_validation():
    t = _published_template()
    with pytest.raises(InvalidParameterError):
        _resolve_parameters(t, {"city": "Lahore", "sort": "bogus"})


def test_resolve_enum_accepts_valid():
    t = _published_template()
    resolved = _resolve_parameters(t, {"city": "Lahore", "sort": "rating"})
    assert resolved["sort"] == "rating"


def test_resolve_ignores_unknown_keys():
    t = _published_template()
    resolved = _resolve_parameters(
        t, {"city": "Lahore", "totally_unknown": "ignored"},
    )
    assert "totally_unknown" not in resolved


def test_resolve_optional_without_default_absent():
    t = Template(
        input_parameters=[
            InputParameter(name="maybe", required=False),
        ],
    )
    resolved = _resolve_parameters(t, {})
    assert "maybe" not in resolved


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

def test_deep_merge_overrides_scalar():
    assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_deep_merge_recursive_dict():
    assert _deep_merge(
        {"a": {"x": 1, "y": 2}},
        {"a": {"y": 3, "z": 4}},
    ) == {"a": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_adds_new_key():
    assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_deep_merge_list_replaces_not_merges():
    assert _deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    overlay = {"a": {"y": 2}}
    _deep_merge(base, overlay)
    assert base == {"a": {"x": 1}}
    assert overlay == {"a": {"y": 2}}


# ---------------------------------------------------------------------------
# Build — happy path
# ---------------------------------------------------------------------------

def test_build_produces_task_spec():
    spec = build_from_template(
        _published_template(),
        parameters={"city": "Lahore"},
        natural_language_prompt="find pizza shops in Lahore",
    )
    assert spec.objective == "lead_gen"
    assert spec.target.source_hint == "pizza shops in Lahore"
    assert spec.constraints.geography == "Lahore"
    assert spec.quality.min_records == 10
    assert spec.field_names == ["business_name", "phone"]
    assert spec.natural_language_prompt == "find pizza shops in Lahore"


def test_build_respects_explicit_int():
    spec = build_from_template(
        _published_template(),
        parameters={"city": "Karachi", "count": 50},
    )
    assert spec.quality.min_records == 50


def test_build_coerces_string_int():
    spec = build_from_template(
        _published_template(),
        parameters={"city": "Karachi", "count": "50"},
    )
    assert spec.quality.min_records == 50


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def test_overrides_merged_on_top():
    spec = build_from_template(
        _published_template(),
        parameters={"city": "Lahore"},
        overrides={
            "quality": {"min_records": 99},
            "output": {"format": "csv"},
        },
    )
    assert spec.quality.min_records == 99
    assert spec.output.format == "csv"
    # Other quality fields preserved
    assert spec.quality.max_failed_page_pct == 0.3


def test_overrides_do_not_mutate_template_defaults():
    t = _published_template()
    original = t.to_dict()["task_spec_defaults"]
    build_from_template(
        t,
        parameters={"city": "Lahore"},
        overrides={"quality": {"min_records": 999}},
    )
    assert t.to_dict()["task_spec_defaults"] == original


def test_no_prompt_leaves_field_empty():
    spec = build_from_template(
        _published_template(), parameters={"city": "Lahore"},
    )
    assert spec.natural_language_prompt == ""


# ---------------------------------------------------------------------------
# Unpublished templates
# ---------------------------------------------------------------------------

def test_draft_rejected_by_default():
    t = Template(
        name="draft", status=TemplateStatus.DRAFT,
        task_spec_defaults={"objective": "extract"},
    )
    with pytest.raises(UnpublishedTemplateError):
        build_from_template(t)


def test_tested_rejected_by_default():
    t = Template(
        name="tested", status=TemplateStatus.TESTED,
        task_spec_defaults={"objective": "extract"},
    )
    with pytest.raises(UnpublishedTemplateError):
        build_from_template(t)


def test_approved_accepted():
    t = Template(
        name="approved", status=TemplateStatus.APPROVED,
        task_spec_defaults={"objective": "extract"},
    )
    spec = build_from_template(t)
    assert spec.objective == "extract"


def test_draft_allowed_when_builder_opts_in():
    t = Template(
        name="draft", status=TemplateStatus.DRAFT,
        task_spec_defaults={"objective": "extract"},
    )
    builder = TemplateBuilder(require_selectable=False)
    spec = builder.build(t)
    assert spec.objective == "extract"


# ---------------------------------------------------------------------------
# Missing / invalid parameters
# ---------------------------------------------------------------------------

def test_missing_required_raises():
    with pytest.raises(MissingParameterError):
        build_from_template(_published_template(), parameters={})


def test_unknown_type_parameter_raises():
    """A parameter with an unrecognized type should refuse coercion."""
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        input_parameters=[
            # Construct directly to bypass the enum:
            InputParameter(name="weird"),
        ],
        task_spec_defaults={"objective": "extract"},
    )
    # Force an invalid type value
    t.input_parameters[0].type = "definitely-not-real"
    with pytest.raises(InvalidParameterError):
        build_from_template(t, parameters={"weird": "x"})


# ---------------------------------------------------------------------------
# Type-preservation edge cases
# ---------------------------------------------------------------------------

def test_placeholder_in_int_field_stays_int():
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        input_parameters=[
            InputParameter(name="n", type=ParameterType.INTEGER, default=5),
        ],
        task_spec_defaults={"quality": {"min_records": "{n}"}},
    )
    spec = build_from_template(t, parameters={})
    assert spec.quality.min_records == 5
    assert isinstance(spec.quality.min_records, int)


def test_placeholder_in_text_field_stays_string():
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        input_parameters=[InputParameter(name="city")],
        task_spec_defaults={"constraints": {"geography": "{city}"}},
    )
    spec = build_from_template(t, parameters={"city": "Lahore"})
    assert spec.constraints.geography == "Lahore"


# ---------------------------------------------------------------------------
# TaskSpec validation surfaced as TemplateBuildError
# ---------------------------------------------------------------------------

def test_nested_group_with_wrong_type_raises():
    """A field that maps to a nested dataclass must itself be a dict."""
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        task_spec_defaults={"target": "should-be-a-dict-not-a-string"},
    )
    with pytest.raises(TemplateBuildError) as exc:
        build_from_template(t)
    msg = str(exc.value)
    assert "target" in msg
    assert "dict" in msg


def test_nested_list_with_wrong_type_raises():
    """A field that maps to a list of dataclasses must itself be a list."""
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        task_spec_defaults={"fields": "not-a-list"},
    )
    with pytest.raises(TemplateBuildError) as exc:
        build_from_template(t)
    assert "fields" in str(exc.value)


def test_nested_list_with_non_dict_items_raises():
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        task_spec_defaults={"fields": [{"name": "ok"}, "not-a-dict"]},
    )
    with pytest.raises(TemplateBuildError) as exc:
        build_from_template(t)
    msg = str(exc.value)
    assert "fields" in msg and "[1]" in msg


def test_well_formed_spec_dict_passes_validation():
    """Sanity check: the happy path still works."""
    t = Template(
        name="x", status=TemplateStatus.PUBLISHED,
        task_spec_defaults={
            "objective": "extract",
            "target": {"start_urls": ["https://example.com/a"]},
            "fields": [{"name": "title"}],
        },
    )
    spec = build_from_template(t)
    assert spec.target.start_urls == ["https://example.com/a"]
    assert spec.field_names == ["title"]