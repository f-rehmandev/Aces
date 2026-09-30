"""
Input dispatcher — spec §10.

One entry point (`resolve_input`) that:
    - accepts any of the ten input shapes the spec lists
    - auto-detects kind when the caller doesn't specify one
    - routes to the right resolver
    - returns a single InputResolution

Deferred kinds (raise NotImplementedError with a clear message until the
supporting infrastructure exists):
    - API_REQUEST    (machine-authenticated external invocation)
    - WEBHOOK        (event-triggered invocation)
    
SAVED_TASK and SCHEDULED_TASK are backed by the existing tenant-scoped
ServiceRegistry. Scheduling state lives on TaskSpec.schedule.
"""

from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from src.intake.resolver import InputResolution, UrlListResolver
from src.intake.sitemap_resolver import SitemapResolver
from src.intake.json_task import JsonTaskFileResolver
from src.intake.nl_resolver import NaturalLanguageResolver


class InputKind(str, Enum):
    NATURAL_LANGUAGE = "natural_language"
    URL_LIST = "url_list"
    SITEMAP = "sitemap"
    JSON_TASK = "json_task"
    TEMPLATE = "template"
    SAVED_TASK = "saved_task"
    SCHEDULED_TASK = "scheduled_task"
    API_REQUEST = "api_request"
    WEBHOOK = "webhook"


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

_PATH_SUFFIXES = {
    ".json": InputKind.JSON_TASK,
    ".xml": InputKind.SITEMAP,
    ".csv": InputKind.URL_LIST,
    ".txt": InputKind.URL_LIST,
    ".list": InputKind.URL_LIST,
    ".urls": InputKind.URL_LIST,
}


def detect_kind(raw) -> InputKind:
    """Best-effort guess of the input kind. Callers can override."""
    if isinstance(raw, dict):
        # Templates are dicts too — check for the identifier keys
        # BEFORE falling through to JSON_TASK.
        if raw.get("template_id") or raw.get("template_name"):
            return InputKind.TEMPLATE
        return InputKind.JSON_TASK

    if isinstance(raw, (list, tuple)):
        return InputKind.URL_LIST

    if isinstance(raw, Path):
        return _PATH_SUFFIXES.get(raw.suffix.lower(), InputKind.URL_LIST)

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("empty input")

        # JSON blob
        if s.startswith(("{", "[")):
            return InputKind.JSON_TASK

        # Explicit URLs — always treated as URL list, even a single URL
        if s.startswith(("http://", "https://")):
            if s.endswith(".xml") and "\n" not in s:
                return InputKind.SITEMAP
            return InputKind.URL_LIST

        # Path-looking strings
        suffix = Path(s).suffix.lower()
        if suffix in _PATH_SUFFIXES:
            return _PATH_SUFFIXES[suffix]

        # Default: natural language
        return InputKind.NATURAL_LANGUAGE

    raise TypeError(f"cannot detect kind for {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def resolve_input(
    raw: Union[str, Path, list, dict],
    kind: Optional[InputKind] = None,
    client_id: str = "default",
    nl_router=None,
    template_registry=None,
    task_registry=None,
) -> InputResolution:
    """
    Route `raw` to the correct resolver.

    `kind` is optional — if omitted, `detect_kind` guesses. Pass an explicit
    kind when auto-detection would be ambiguous (e.g. a natural-language
    string that happens to end in ".xml").

    `client_id` is attached to the resulting TaskSpec so downstream stages
    stay tenant-scoped.

    `template_registry` is required only for TEMPLATE inputs — a
    `src.templates.registry.TemplateRegistry`. Any other input kind
    ignores it.
    """
    if kind is None:
        kind = detect_kind(raw)

    if kind == InputKind.NATURAL_LANGUAGE:
        result = NaturalLanguageResolver(router=nl_router).resolve(raw)

    elif kind == InputKind.URL_LIST:
        result = UrlListResolver().resolve(raw)

    elif kind == InputKind.SITEMAP:
        result = SitemapResolver().resolve(str(raw))

    elif kind == InputKind.JSON_TASK:
        result = JsonTaskFileResolver().resolve(raw)

    elif kind == InputKind.TEMPLATE:
        if template_registry is None:
            raise ValueError(
                "template input requires a TemplateRegistry; pass "
                "template_registry= to resolve_input()"
            )
        from src.intake.template_resolver import TemplateResolver
        result = TemplateResolver(template_registry).resolve(raw)

    elif kind in (InputKind.SAVED_TASK, InputKind.SCHEDULED_TASK):
        if task_registry is None:
            raise ValueError(
                f"{kind.value} input requires a ServiceRegistry; "
                "pass task_registry= to resolve_input()"
            )

        # Explicit input forms:
        #   "task-id"
        #   {"task_id": "task-id"}
        if isinstance(raw, dict):
            task_id = str(
                raw.get("task_id")
                or raw.get("id")
                or ""
            ).strip()
        else:
            task_id = str(raw).strip()

        if not task_id:
            raise ValueError(
                f"{kind.value} input requires a task_id"
            )

        spec = task_registry.get_task(client_id, task_id)

        if spec is None:
            raise ValueError(
                f"{kind.value} task not found: {task_id}"
            )

        if kind == InputKind.SCHEDULED_TASK:
            schedule = spec.schedule

            if not schedule.enabled:
                raise ValueError(
                    f"scheduled task is disabled: {task_id}"
                )

            recurring_cadences = {
                "hourly",
                "daily",
                "weekly",
                "interval",
                "cron",
                "event",
            }

            if schedule.cadence not in recurring_cadences:
                raise ValueError(
                    f"task {task_id} does not have a recurring schedule"
                )

            source_description = (
                f"scheduled task {task_id} "
                f"({schedule.cadence})"
            )
        else:
            source_description = f"saved task {task_id}"

        result = InputResolution(
            spec=spec,
            source_description=source_description,
        )

    elif kind in (InputKind.API_REQUEST, InputKind.WEBHOOK):
        raise NotImplementedError(
            f"{kind.value} resolver requires the API/webhook layer — Part X"
        )

    else:
        raise ValueError(f"unknown input kind: {kind}")

    # Tenant scoping (§43) — apply once at the dispatch boundary.
    result.spec.client_id = client_id
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # List of URLs
    r = resolve_input(["https://x.com/a", "https://x.com/b"])
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]

    # Inline JSON
    r = resolve_input('{"objective": "monitor"}')
    assert r.spec.objective == "monitor"

    # Single URL string
    r = resolve_input("https://x.com/a")
    assert r.spec.target.start_urls == ["https://x.com/a"]

    # detect_kind checks
    assert detect_kind({"a": 1}) == InputKind.JSON_TASK
    assert detect_kind(["https://x.com/a"]) == InputKind.URL_LIST
    assert detect_kind("https://x.com/sitemap.xml") == InputKind.SITEMAP
    assert detect_kind("find best prices") == InputKind.NATURAL_LANGUAGE
    assert detect_kind("/tmp/urls.csv") == InputKind.URL_LIST

    # client_id propagation
    r = resolve_input("https://x.com/a", client_id="acme")
    assert r.spec.client_id == "acme"

    # Deferred kinds raise cleanly
    try:
        resolve_input("ignored", kind=InputKind.SAVED_TASK)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as e:
        print(f"deferred OK: {e}")

    print("Dispatcher OK.")