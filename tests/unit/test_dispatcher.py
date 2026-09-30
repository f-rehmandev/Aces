"""Unit tests for the input dispatcher (spec §10)."""
import json
import pytest

from src.intake.dispatcher import resolve_input, detect_kind, InputKind
from src.intake.resolver import InputResolution


# --- detection ---------------------------------------------------------

def test_detect_dict_is_json_task():
    assert detect_kind({"a": 1}) == InputKind.JSON_TASK


def test_detect_list_is_url_list():
    assert detect_kind(["https://x.com/a"]) == InputKind.URL_LIST


def test_detect_json_string():
    assert detect_kind('{"objective": "monitor"}') == InputKind.JSON_TASK


def test_detect_url_string_is_url_list():
    assert detect_kind("https://x.com/a") == InputKind.URL_LIST


def test_detect_sitemap_xml_url():
    assert detect_kind("https://x.com/sitemap.xml") == InputKind.SITEMAP


def test_detect_file_path_by_suffix():
    assert detect_kind("urls.csv") == InputKind.URL_LIST
    assert detect_kind("task.json") == InputKind.JSON_TASK
    assert detect_kind("sitemap.xml") == InputKind.SITEMAP


def test_detect_natural_language():
    assert detect_kind("find best prices for a wireless mouse") == InputKind.NATURAL_LANGUAGE


def test_detect_empty_string_raises():
    with pytest.raises(ValueError):
        detect_kind("")


def test_detect_unsupported_type_raises():
    with pytest.raises(TypeError):
        detect_kind(12345)


# --- dispatch ----------------------------------------------------------

def test_resolve_url_list():
    r = resolve_input(["https://x.com/a", "https://x.com/b"])
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]


def test_resolve_json_dict():
    r = resolve_input({"objective": "monitor", "natural_language_prompt": "x"})
    assert r.spec.objective == "monitor"


def test_resolve_inline_json_string():
    r = resolve_input(json.dumps({"objective": "extract"}))
    assert r.spec.objective == "extract"


def test_resolve_single_url():
    r = resolve_input("https://x.com/a")
    assert r.spec.target.start_urls == ["https://x.com/a"]


def test_explicit_kind_overrides_detection():
    # "https://x.com/a" would auto-detect as URL_LIST, but we force JSON
    with pytest.raises(Exception):
        resolve_input("not json", kind=InputKind.JSON_TASK)


def test_client_id_propagates():
    r = resolve_input("https://x.com/a", client_id="acme")
    assert r.spec.client_id == "acme"


def test_saved_task_resolves_from_registry():
    from src.api.registry import ServiceRegistry
    from src.core.task_spec import TaskSpec, Target

    registry = ServiceRegistry()

    spec = TaskSpec(
        natural_language_prompt="find laptop prices",
        target=Target(
            start_urls=["https://example.com/laptops"]
        ),
    )

    registry.save_task("acme", spec)

    result = resolve_input(
        spec.task_id,
        kind=InputKind.SAVED_TASK,
        client_id="acme",
        task_registry=registry,
    )

    assert result.spec is spec
    assert result.spec.client_id == "acme"
    assert result.source_description == f"saved task {spec.task_id}"


def test_saved_task_accepts_dict_form():
    from src.api.registry import ServiceRegistry
    from src.core.task_spec import TaskSpec

    registry = ServiceRegistry()
    spec = TaskSpec(natural_language_prompt="test")
    registry.save_task("acme", spec)

    result = resolve_input(
        {"task_id": spec.task_id},
        kind=InputKind.SAVED_TASK,
        client_id="acme",
        task_registry=registry,
    )

    assert result.spec is spec


def test_saved_task_is_tenant_scoped():
    from src.api.registry import ServiceRegistry
    from src.core.task_spec import TaskSpec

    registry = ServiceRegistry()
    spec = TaskSpec(natural_language_prompt="private task")
    registry.save_task("acme", spec)

    with pytest.raises(ValueError, match="task not found"):
        resolve_input(
            spec.task_id,
            kind=InputKind.SAVED_TASK,
            client_id="other",
            task_registry=registry,
        )


def test_scheduled_task_resolves_enabled_recurring_task():
    from src.api.registry import ServiceRegistry
    from src.core.task_spec import TaskSpec, Schedule

    registry = ServiceRegistry()

    spec = TaskSpec(natural_language_prompt="daily prices")
    spec.schedule = Schedule(
        cadence="daily",
        at_time="09:00",
        timezone="UTC",
        enabled=True,
    )

    registry.save_task("acme", spec)

    result = resolve_input(
        spec.task_id,
        kind=InputKind.SCHEDULED_TASK,
        client_id="acme",
        task_registry=registry,
    )

    assert result.spec is spec
    assert result.source_description == (
        f"scheduled task {spec.task_id} (daily)"
    )


def test_scheduled_task_rejects_disabled_task():
    from src.api.registry import ServiceRegistry
    from src.core.task_spec import TaskSpec, Schedule

    registry = ServiceRegistry()

    spec = TaskSpec()
    spec.schedule = Schedule(
        cadence="daily",
        enabled=False,
    )

    registry.save_task("acme", spec)

    with pytest.raises(ValueError, match="disabled"):
        resolve_input(
            spec.task_id,
            kind=InputKind.SCHEDULED_TASK,
            client_id="acme",
            task_registry=registry,
        )


def test_scheduled_task_rejects_once_schedule():
    from src.api.registry import ServiceRegistry
    from src.core.task_spec import TaskSpec, Schedule

    registry = ServiceRegistry()

    spec = TaskSpec()
    spec.schedule = Schedule(
        cadence="once",
        enabled=True,
    )

    registry.save_task("acme", spec)

    with pytest.raises(ValueError, match="recurring"):
        resolve_input(
            spec.task_id,
            kind=InputKind.SCHEDULED_TASK,
            client_id="acme",
            task_registry=registry,
        )


def test_saved_task_requires_registry():
    with pytest.raises(ValueError, match="ServiceRegistry"):
        resolve_input(
            "abc",
            kind=InputKind.SAVED_TASK,
        )


def test_api_request_and_webhook_remain_deferred():
    with pytest.raises(NotImplementedError):
        resolve_input(
            "anything",
            kind=InputKind.API_REQUEST,
        )

    with pytest.raises(NotImplementedError):
        resolve_input(
            "anything",
            kind=InputKind.WEBHOOK,
        )