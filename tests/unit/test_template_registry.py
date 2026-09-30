"""Unit tests for TemplateRegistry (spec §10A.2)."""
import pytest

from src.templates.models import (
    InputParameter,
    ParameterType,
    TaskArchetype,
    Template,
    TemplateStatus,
)
from src.templates.registry import (
    DuplicateTemplateError,
    ImmutableTemplateError,
    InvalidTransitionError,
    TemplateNotFoundError,
    TemplateRegistry,
    TemplateRegistryError,
    can_transition,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def reg():
    return TemplateRegistry()


def _make(name="pizza-lead-gen", **kw) -> Template:
    return Template(
        name=name,
        archetype=kw.pop("archetype", TaskArchetype.LEAD_GENERATION),
        input_parameters=kw.pop("input_parameters", [InputParameter(name="city")]),
        **kw,
    )


# ---------------------------------------------------------------------------
# can_transition helper
# ---------------------------------------------------------------------------

def test_can_transition_same_status():
    for s in TemplateStatus:
        assert can_transition(s, s)


def test_can_transition_draft_to_tested():
    assert can_transition(TemplateStatus.DRAFT, TemplateStatus.TESTED)


def test_can_transition_published_cannot_go_back():
    assert not can_transition(TemplateStatus.PUBLISHED, TemplateStatus.DRAFT)
    assert not can_transition(TemplateStatus.PUBLISHED, TemplateStatus.APPROVED)
    assert can_transition(TemplateStatus.PUBLISHED, TemplateStatus.RETIRED)


def test_can_transition_retired_is_terminal():
    for s in TemplateStatus:
        if s == TemplateStatus.RETIRED:
            continue
        assert not can_transition(TemplateStatus.RETIRED, s)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def test_register_adds_to_index(reg):
    t = _make()
    reg.register(t)
    assert len(reg) == 1
    assert reg.get(t.template_id) is t
    assert t.template_id in reg


def test_register_duplicate_id_raises(reg):
    t = _make()
    reg.register(t)
    with pytest.raises(DuplicateTemplateError):
        reg.register(t)


def test_register_duplicate_name_version_raises(reg):
    reg.register(_make(name="x"))
    with pytest.raises(DuplicateTemplateError):
        reg.register(_make(name="x"))


def test_register_different_versions_ok(reg):
    v1 = _make(name="x", version=1)
    v2 = _make(name="x", version=2)
    reg.register(v1)
    reg.register(v2)
    assert len(reg) == 2


def test_register_rejects_invalid_parameter(reg):
    bad = Template(
        name="bad",
        input_parameters=[InputParameter(name="UPPER")],
    )
    with pytest.raises(TemplateRegistryError) as exc:
        reg.register(bad)
    assert "snake_case" in str(exc.value)


# ---------------------------------------------------------------------------
# get / require / get_by_name
# ---------------------------------------------------------------------------

def test_get_missing_returns_none(reg):
    assert reg.get("nope") is None


def test_require_missing_raises(reg):
    with pytest.raises(TemplateNotFoundError):
        reg.require("nope")


def test_require_found(reg):
    t = _make()
    reg.register(t)
    assert reg.require(t.template_id) is t


def test_get_by_name_missing(reg):
    assert reg.get_by_name("nope") is None


def test_get_by_name_exact_version(reg):
    v1 = _make(name="x", version=1)
    v2 = _make(name="x", version=2)
    reg.register(v1)
    reg.register(v2)
    assert reg.get_by_name("x", version=1) is v1
    assert reg.get_by_name("x", version=2) is v2
    assert reg.get_by_name("x", version=99) is None


def test_get_by_name_prefers_published(reg):
    v1 = _make(name="x", version=1)
    v2 = _make(name="x", version=2)
    reg.register(v1)
    reg.register(v2)
    reg.transition_status(v2.template_id, TemplateStatus.TESTED)
    reg.transition_status(v2.template_id, TemplateStatus.APPROVED)
    reg.transition_status(v2.template_id, TemplateStatus.PUBLISHED)
    # v2 published → preferred even though v1 is a lower version
    assert reg.get_by_name("x") is v2


def test_get_by_name_falls_back_to_highest_of_any_status(reg):
    v1 = _make(name="x", version=1)
    v2 = _make(name="x", version=2)
    v3 = _make(name="x", version=3)
    reg.register(v1)
    reg.register(v2)
    reg.register(v3)
    # v3 is highest draft
    assert reg.get_by_name("x") is v3


def test_get_by_name_prefers_approved_over_lower_published(reg):
    v1 = _make(name="x", version=1)
    v2 = _make(name="x", version=2)
    reg.register(v1)
    reg.register(v2)
    # v1 published, v2 approved — published wins the tier
    reg.transition_status(v1.template_id, TemplateStatus.TESTED)
    reg.transition_status(v1.template_id, TemplateStatus.APPROVED)
    reg.transition_status(v1.template_id, TemplateStatus.PUBLISHED)
    reg.transition_status(v2.template_id, TemplateStatus.TESTED)
    reg.transition_status(v2.template_id, TemplateStatus.APPROVED)
    assert reg.get_by_name("x") is v1


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_list_all(reg):
    reg.register(_make(name="a"))
    reg.register(_make(name="b"))
    assert len(reg.list_all()) == 2


def test_list_by_status(reg):
    a = _make(name="a")
    b = _make(name="b")
    reg.register(a)
    reg.register(b)
    reg.transition_status(b.template_id, TemplateStatus.TESTED)
    assert reg.list_by_status(TemplateStatus.DRAFT) == [a]
    assert reg.list_by_status(TemplateStatus.TESTED) == [b]


def test_list_by_archetype(reg):
    a = _make(name="a", archetype=TaskArchetype.LEAD_GENERATION)
    b = _make(name="b", archetype=TaskArchetype.DIRECTORY)
    reg.register(a)
    reg.register(b)
    assert reg.list_by_archetype(TaskArchetype.LEAD_GENERATION) == [a]
    assert reg.list_by_archetype(TaskArchetype.DIRECTORY) == [b]


def test_list_selectable_only_approved_or_published(reg):
    draft = _make(name="a")
    tested = _make(name="b")
    approved = _make(name="c")
    published = _make(name="d")
    reg.register(draft)
    reg.register(tested)
    reg.register(approved)
    reg.register(published)
    reg.transition_status(tested.template_id, TemplateStatus.TESTED)
    reg.transition_status(approved.template_id, TemplateStatus.TESTED)
    reg.transition_status(approved.template_id, TemplateStatus.APPROVED)
    reg.transition_status(published.template_id, TemplateStatus.TESTED)
    reg.transition_status(published.template_id, TemplateStatus.APPROVED)
    reg.transition_status(published.template_id, TemplateStatus.PUBLISHED)

    selectable = reg.list_selectable()
    assert {t.template_id for t in selectable} == {
        approved.template_id, published.template_id,
    }

# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

def test_transition_through_full_lifecycle(reg):
    t = _make()
    reg.register(t)
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    assert t.status == TemplateStatus.TESTED
    reg.transition_status(t.template_id, TemplateStatus.APPROVED)
    assert t.status == TemplateStatus.APPROVED
    reg.transition_status(t.template_id, TemplateStatus.PUBLISHED)
    assert t.status == TemplateStatus.PUBLISHED


def test_transition_updates_updated_at(reg):
    t = _make()
    reg.register(t)
    before = t.updated_at
    import time
    time.sleep(0.01)
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    assert t.updated_at != before


def test_transition_invalid_raises(reg):
    t = _make()
    reg.register(t)
    with pytest.raises(InvalidTransitionError):
        reg.transition_status(t.template_id, TemplateStatus.PUBLISHED)


def test_transition_missing_template_raises(reg):
    with pytest.raises(TemplateNotFoundError):
        reg.transition_status("nope", TemplateStatus.TESTED)


def test_transition_testing_back_to_draft(reg):
    t = _make()
    reg.register(t)
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    reg.transition_status(t.template_id, TemplateStatus.DRAFT)
    assert t.status == TemplateStatus.DRAFT


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_update_published_raises(reg):
    t = _make()
    reg.register(t)
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    reg.transition_status(t.template_id, TemplateStatus.APPROVED)
    reg.transition_status(t.template_id, TemplateStatus.PUBLISHED)
    with pytest.raises(ImmutableTemplateError):
        reg.update(t)


def test_update_draft_ok(reg):
    t = _make()
    reg.register(t)
    t.description = "new description"
    reg.update(t)
    assert reg.get(t.template_id).description == "new description"


def test_update_status_change_raises(reg):
    t = _make()
    reg.register(t)
    t.status = TemplateStatus.TESTED
    with pytest.raises(TemplateRegistryError):
        reg.update(t)


def test_update_name_change_raises(reg):
    t = _make()
    reg.register(t)
    t.name = "different-name"
    with pytest.raises(TemplateRegistryError):
        reg.update(t)


def test_update_missing_raises(reg):
    with pytest.raises(TemplateNotFoundError):
        reg.update(_make())


# ---------------------------------------------------------------------------
# new_version
# ---------------------------------------------------------------------------

def test_new_version_creates_draft_clone(reg):
    t = _make()
    reg.register(t)
    clone = reg.new_version(t.template_id)
    assert clone.template_id != t.template_id
    assert clone.version == 2
    assert clone.status == TemplateStatus.DRAFT
    assert clone.name == t.name
    assert clone.parent_template_id == t.template_id


def test_new_version_preserves_content(reg):
    t = _make()
    t.description = "original"
    t.input_parameters = [InputParameter(name="city")]
    reg.register(t)
    clone = reg.new_version(t.template_id)
    assert clone.description == "original"
    assert len(clone.input_parameters) == 1


def test_new_version_from_published_creates_draft(reg):
    t = _make()
    reg.register(t)
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    reg.transition_status(t.template_id, TemplateStatus.APPROVED)
    reg.transition_status(t.template_id, TemplateStatus.PUBLISHED)

    clone = reg.new_version(t.template_id)
    assert clone.status == TemplateStatus.DRAFT
    # Original untouched
    assert t.status == TemplateStatus.PUBLISHED


def test_new_version_twice_raises(reg):
    t = _make()
    reg.register(t)
    reg.new_version(t.template_id)   # creates v2
    with pytest.raises(DuplicateTemplateError):
        reg.new_version(t.template_id)   # v2 already exists


def test_new_version_missing_raises(reg):
    with pytest.raises(TemplateNotFoundError):
        reg.new_version("nope")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_draft(reg):
    t = _make()
    reg.register(t)
    assert reg.delete(t.template_id) is True
    assert reg.get(t.template_id) is None
    assert len(reg) == 0


def test_delete_missing_returns_false(reg):
    assert reg.delete("nope") is False


def test_delete_published_raises(reg):
    t = _make()
    reg.register(t)
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    reg.transition_status(t.template_id, TemplateStatus.APPROVED)
    reg.transition_status(t.template_id, TemplateStatus.PUBLISHED)
    with pytest.raises(ImmutableTemplateError):
        reg.delete(t.template_id)


# ---------------------------------------------------------------------------
# Empty registry behavior
# ---------------------------------------------------------------------------

def test_empty_registry(reg):
    assert len(reg) == 0
    assert reg.get("x") is None
    assert reg.get_by_name("x") is None
    assert reg.list_all() == []
    assert reg.list_selectable() == []