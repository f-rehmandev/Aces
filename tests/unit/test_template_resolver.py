"""Unit tests for TemplateResolver + dispatcher template wiring (§10A.3)."""
import pytest

from src.intake.dispatcher import InputKind, detect_kind, resolve_input
from src.intake.template_resolver import (
    TemplateResolutionError,
    TemplateResolver,
    resolve_template,
)
from src.templates.models import (
    InputParameter,
    ParameterType,
    TaskArchetype,
    Template,
    TemplateStatus,
)
from src.templates.registry import TemplateRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    reg = TemplateRegistry()
    t = Template(
        name="pizza-lead-gen",
        archetype=TaskArchetype.LEAD_GENERATION,
        status=TemplateStatus.DRAFT,   # walk status later
        input_parameters=[
            InputParameter(name="city", required=True),
            InputParameter(name="count", type=ParameterType.INTEGER, default=20),
        ],
        task_spec_defaults={
            "objective": "lead_gen",
            "target": {"source_hint": "pizza shops in {city}"},
            "constraints": {"geography": "{city}"},
            "quality": {"min_records": "{count}"},
        },
    )
    reg.register(t)
    # Walk the status forward manually — this is a fixture, not a
    # registry-transition test.
    t.status = TemplateStatus.PUBLISHED
    return reg, t


@pytest.fixture
def resolver(registry):
    reg, _ = registry
    return TemplateResolver(reg)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_resolve_by_id(resolver, registry):
    _, t = registry
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
        "prompt": "find pizza shops in Lahore",
    })
    assert r.spec.objective == "lead_gen"
    assert r.spec.constraints.geography == "Lahore"
    assert r.spec.quality.min_records == 20
    assert r.spec.natural_language_prompt == "find pizza shops in Lahore"


def test_resolve_by_name(resolver):
    r = resolver.resolve({
        "template_name": "pizza-lead-gen",
        "parameters": {"city": "Karachi"},
    })
    assert r.spec.constraints.geography == "Karachi"


def test_resolve_by_name_and_version(resolver):
    r = resolver.resolve({
        "template_name": "pizza-lead-gen",
        "template_version": 1,
        "parameters": {"city": "Quetta"},
    })
    assert r.spec.constraints.geography == "Quetta"


def test_resolve_applies_overrides(resolver, registry):
    _, t = registry
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
        "overrides": {"quality": {"min_records": 999}},
    })
    assert r.spec.quality.min_records == 999


def test_source_description_mentions_template(resolver, registry):
    _, t = registry
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
    })
    assert "pizza-lead-gen" in r.source_description
    assert "v1" in r.source_description
    assert "published" in r.source_description


def test_no_prompt_produces_warning(resolver, registry):
    _, t = registry
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
    })
    assert r.warnings
    assert any("prompt" in w.lower() for w in r.warnings)


def test_prompt_suppresses_warning(resolver, registry):
    _, t = registry
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
        "prompt": "anything",
    })
    assert r.warnings == []


def test_missing_optional_parameters_uses_default(resolver, registry):
    _, t = registry
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
    })
    assert r.spec.quality.min_records == 20


# ---------------------------------------------------------------------------
# Rejections — input shape
# ---------------------------------------------------------------------------

def test_non_dict_input_rejected(resolver):
    with pytest.raises(TemplateResolutionError):
        resolver.resolve("not a dict")


def test_missing_identifier_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({"parameters": {"city": "Lahore"}})
    assert "template_id" in str(exc.value) or "template_name" in str(exc.value)


def test_wrong_typed_template_id_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({"template_id": 123})
    assert "template_id" in str(exc.value)


def test_wrong_typed_template_name_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({"template_name": 42})
    assert "template_name" in str(exc.value)


def test_wrong_typed_template_version_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({
            "template_name": "pizza-lead-gen",
            "template_version": "1",
        })
    assert "template_version" in str(exc.value)


def test_wrong_typed_parameters_rejected(resolver, registry):
    _, t = registry
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({
            "template_id": t.template_id,
            "parameters": "not a dict",
        })
    assert "parameters" in str(exc.value)


def test_wrong_typed_overrides_rejected(resolver, registry):
    _, t = registry
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({
            "template_id": t.template_id,
            "parameters": {"city": "Lahore"},
            "overrides": "not a dict",
        })
    assert "overrides" in str(exc.value)


def test_wrong_typed_prompt_rejected(resolver, registry):
    _, t = registry
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({
            "template_id": t.template_id,
            "parameters": {"city": "Lahore"},
            "prompt": ["not", "a", "string"],
        })
    assert "prompt" in str(exc.value)


# ---------------------------------------------------------------------------
# Rejections — lookup
# ---------------------------------------------------------------------------

def test_unknown_id_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({"template_id": "nope"})
    assert "nope" in str(exc.value)


def test_unknown_name_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({"template_name": "ghost"})
    assert "ghost" in str(exc.value)


def test_unknown_name_and_version_rejected(resolver):
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({
            "template_name": "ghost",
            "template_version": 9,
        })
    msg = str(exc.value)
    assert "ghost" in msg and "9" in msg


# ---------------------------------------------------------------------------
# Rejections — build errors surfaced clearly
# ---------------------------------------------------------------------------

def test_missing_required_parameter_surfaces(resolver, registry):
    _, t = registry
    with pytest.raises(TemplateResolutionError) as exc:
        resolver.resolve({"template_id": t.template_id, "parameters": {}})
    assert "city" in str(exc.value)


def test_draft_template_rejected_by_default(registry):
    reg, _ = registry
    draft = Template(
        name="draft", status=TemplateStatus.DRAFT,
        task_spec_defaults={"objective": "extract"},
    )
    reg.register(draft)
    with pytest.raises(TemplateResolutionError) as exc:
        TemplateResolver(reg).resolve({"template_id": draft.template_id})
    msg = str(exc.value).lower()
    assert "approved" in msg or "published" in msg


def test_draft_allowed_when_resolver_opts_in(registry):
    reg, _ = registry
    draft = Template(
        name="draft", status=TemplateStatus.DRAFT,
        task_spec_defaults={"objective": "extract"},
    )
    reg.register(draft)
    r = TemplateResolver(reg, require_selectable=False).resolve(
        {"template_id": draft.template_id},
    )
    assert r.spec.objective == "extract"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def test_convenience_resolve_template(registry):
    reg, t = registry
    r = resolve_template(
        {"template_id": t.template_id, "parameters": {"city": "Lahore"}},
        reg,
    )
    assert r.spec.constraints.geography == "Lahore"


# ---------------------------------------------------------------------------
# Dispatcher wiring
# ---------------------------------------------------------------------------

def test_detect_kind_recognizes_template_id():
    assert detect_kind({"template_id": "abc"}) == InputKind.TEMPLATE


def test_detect_kind_recognizes_template_name():
    assert detect_kind({"template_name": "pizza"}) == InputKind.TEMPLATE


def test_detect_kind_prefers_template_over_json():
    assert detect_kind(
        {"template_id": "x", "objective": "extract"},
    ) == InputKind.TEMPLATE


def test_detect_kind_json_when_no_template_keys():
    assert detect_kind({"objective": "extract"}) == InputKind.JSON_TASK


def test_dispatcher_resolves_template(registry):
    reg, t = registry
    r = resolve_input(
        {"template_id": t.template_id, "parameters": {"city": "Lahore"}},
        template_registry=reg,
        client_id="acme",
    )
    assert r.spec.constraints.geography == "Lahore"
    assert r.spec.client_id == "acme"


def test_dispatcher_requires_registry_for_template():
    with pytest.raises(ValueError) as exc:
        resolve_input({"template_id": "whatever"})
    assert "TemplateRegistry" in str(exc.value)


def test_dispatcher_url_list_still_works():
    r = resolve_input("https://example.com/a")
    assert r.spec.target.start_urls == ["https://example.com/a"]


def test_dispatcher_json_task_still_works():
    r = resolve_input({"objective": "extract"})
    assert r.spec.objective == "extract"