"""
Concrete API handlers — spec §45.

Endpoints exposed (§45, first pass):

    POST   /v1/tasks                     create a task
    GET    /v1/tasks                     list tasks
    GET    /v1/tasks/{id}                fetch a task
    DELETE /v1/tasks/{id}                delete a task
    POST   /v1/tasks/{id}/validate       dry-validate a task's TaskSpec

    GET    /v1/tasks/{id}/history        run history (dataset versions)
    GET    /v1/tasks/{id}/changes        latest diff summary
    GET    /v1/tasks/{id}/quality        latest quality report

    GET    /v1/strategies                list strategies (stub; §19)

Handlers delegate to `ServiceRegistry`. The registry owns tenant scoping;
handlers translate "not found" into 404s and never touch storage directly.
"""

from __future__ import annotations
import json
from typing import Optional

from src.api.registry import ServiceRegistry
from src.api.schemas import (
    Request, Response, not_found, bad_request, error_response,
)
from src.core.task_spec import TaskSpec
from src.history.change import ChangeClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_body(request: Request) -> Optional[Response]:
    if request.body is None:
        return bad_request("request body required")
    return None


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------

def create_task(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        err = _require_body(request)
        if err:
            return err

        try:
            spec = TaskSpec.from_dict(request.body)
        except TypeError as e:
            return bad_request(f"invalid TaskSpec: {e}", details={"input_keys": sorted(request.body.keys())})

        spec.client_id = request.client_id
        registry.save_task(request.client_id, spec)

        return Response(
            status=201,
            body={"task_id": spec.task_id, "task": spec.to_dict()},
        )
    return handler


def list_tasks(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        tasks = registry.list_tasks(request.client_id)
        return Response(status=200, body={
            "tasks": [{"task_id": t.task_id, "prompt": t.natural_language_prompt}
                      for t in tasks],
        })
    return handler


def get_task(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        spec = registry.get_task(request.client_id, task_id)
        if spec is None:
            return not_found(f"task {task_id} not found")
        return Response(status=200, body={"task": spec.to_dict()})
    return handler


def delete_task(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        deleted = registry.delete_task(request.client_id, task_id)
        if not deleted:
            return not_found(f"task {task_id} not found")
        return Response(status=204, body=None)
    return handler


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_task(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        spec = registry.get_task(request.client_id, task_id)
        if spec is None:
            return not_found(f"task {task_id} not found")

        problems: list[str] = []
        if not spec.target.start_urls and not spec.target.source_hint:
            problems.append("no start URLs and no source hint — cannot discover pages")
        if not spec.fields:
            problems.append("no fields declared — nothing to extract")
        if spec.compliance.refusal_reason:
            problems.append(f"compliance: {spec.compliance.refusal_reason}")

        return Response(status=200, body={
            "valid": not problems,
            "problems": problems,
        })
    return handler


# ---------------------------------------------------------------------------
# Read-only views backed by the dataset registry
# ---------------------------------------------------------------------------

def get_history(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        if registry.get_task(request.client_id, task_id) is None:
            return not_found(f"task {task_id} not found")

        ds = registry.get_dataset(request.client_id, task_id)
        if ds is None or len(ds) == 0:
            return Response(status=200, body={"versions": []})

        versions = []
        for v in ds.all_versions():
            versions.append({
                "version": v.version,
                "created_at": v.created_at,
                "records": len(v.records),
                "quality_passed": v.quality_passed,
                "quality_score": v.quality_score,
                "confidence_mean": v.confidence_mean,
                "superseded": v.superseded,
                "note": v.note,
            })
        return Response(status=200, body={"versions": versions})
    return handler


def get_changes(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        if registry.get_task(request.client_id, task_id) is None:
            return not_found(f"task {task_id} not found")

        ds = registry.get_dataset(request.client_id, task_id)
        if ds is None or len(ds) < 2:
            return Response(status=200, body={
                "summary": {"new": 0, "removed": 0, "modified": 0,
                            "unchanged": 0, "total": 0},
                "note": "fewer than 2 dataset versions; nothing to diff",
            })

        versions = ds.all_versions()
        previous = versions[-2].records
        current = versions[-1].records
        cs = ChangeClassifier().classify(previous, current)
        return Response(status=200, body=cs.to_dict())
    return handler


def get_quality(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        if registry.get_task(request.client_id, task_id) is None:
            return not_found(f"task {task_id} not found")

        ds = registry.get_dataset(request.client_id, task_id)
        if ds is None or len(ds) == 0:
            return not_found(f"no dataset for task {task_id} yet")

        latest = ds.latest()
        return Response(status=200, body={
            "version": latest.version,
            "created_at": latest.created_at,
            "quality_passed": latest.quality_passed,
            "quality_score": latest.quality_score,
            "confidence_mean": latest.confidence_mean,
            "records": len(latest.records),
        })
    return handler

# ---------------------------------------------------------------------------
# Schedule (read-only view of a task's schedule state)
# ---------------------------------------------------------------------------

def get_task_schedule(registry: ServiceRegistry):
    """
    Return the schedule fields of a task.

    `next_fire_at` and `last_fired_at` are populated by SchedulerLoop.
    Before the loop has ticked once, `next_fire_at` is None — that's
    how you know the schedule hasn't been observed yet.
    """
    def handler(request: Request) -> Response:
        task_id = request.path_params.get("id", "")
        spec = registry.get_task(request.client_id, task_id)
        if spec is None:
            return not_found(f"task {task_id} not found")

        s = spec.schedule
        return Response(status=200, body={
            "task_id": task_id,
            "cadence": s.cadence,
            "timezone": s.timezone,
            "at_time": s.at_time,
            "weekdays": list(s.weekdays),
            "interval_seconds": s.interval_seconds,
            "cron_expression": s.cron_expression,
            "run_at": s.run_at,
            "enabled": s.enabled,
            "next_fire_at": s.next_fire_at,
            "last_fired_at": s.last_fired_at,
        })
    return handler



# ---------------------------------------------------------------------------
# Jobs (read-only views over the registry's job store)
# ---------------------------------------------------------------------------

def get_job(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        job_id = request.path_params.get("id", "")
        job = registry.get_job(request.client_id, job_id)
        if job is None:
            return not_found(f"job {job_id} not found")
        # Strip the (potentially large) result payload — this is the
        # lightweight status view. `get_job_result` returns the payload.
        view = {k: v for k, v in job.items() if k != "result"}
        return Response(status=200, body=view)
    return handler


def get_job_result(registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        job_id = request.path_params.get("id", "")
        job = registry.get_job(request.client_id, job_id)
        if job is None:
            return not_found(f"job {job_id} not found")
        state = job.get("state", "")
        if state != "completed":
            return bad_request(
                f"job {job_id} is in state {state!r}, "
                f"result not available"
            )
        return Response(
            status=200,
            body={"job_id": job_id, "result": job.get("result")},
        )
    return handler



# ---------------------------------------------------------------------------
# Stubs (real implementations come with §19 evolution engine)
# ---------------------------------------------------------------------------

def list_strategies(_registry: ServiceRegistry):
    def handler(request: Request) -> Response:
        # Real strategies will be pulled from src.strategy.strategy_memory.
        return Response(status=200, body={"strategies": []})
    return handler


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def build_router(registry: ServiceRegistry):
    """Register all §45 routes on a fresh Router and return it."""
    from src.api.router import Router
    r = Router()

    r.add("POST",   "/v1/tasks",              create_task(registry))
    r.add("GET",    "/v1/tasks",              list_tasks(registry))
    r.add("GET",    "/v1/tasks/{id}",         get_task(registry))
    r.add("DELETE", "/v1/tasks/{id}",         delete_task(registry))
    r.add("POST",   "/v1/tasks/{id}/validate", validate_task(registry))
    r.add("GET",    "/v1/tasks/{id}/history", get_history(registry))
    r.add("GET",    "/v1/tasks/{id}/changes", get_changes(registry))
    r.add("GET",    "/v1/tasks/{id}/quality", get_quality(registry))

    r.add("GET",    "/v1/jobs/{id}",          get_job(registry))
    r.add("GET",    "/v1/jobs/{id}/result",   get_job_result(registry))
    r.add("GET",    "/v1/tasks/{id}/schedule", get_task_schedule(registry))

    r.add("GET",    "/v1/strategies",         list_strategies(registry))

    return r


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.api.keys import ApiKeyStore
    from src.api.router import auth_middleware

    store = ApiKeyStore()
    _, raw = store.create("acme")

    reg = ServiceRegistry()
    r = build_router(reg)
    r.add_middleware(auth_middleware(store))

    def call(method, path, body=None, key=raw):
        return r.dispatch(Request(
            method=method, path=path, body=body,
            headers={"Authorization": f"Bearer {key}"},
        ))

    # Create a task
    resp = call("POST", "/v1/tasks", body={
        "natural_language_prompt": "find laptop prices",
        "target": {"start_urls": ["https://example.com/a"]},
        "fields": [{"name": "title"}, {"name": "price", "type": "currency"}],
    })
    assert resp.status == 201, resp.body
    task_id = resp.body["task_id"]

    # List
    resp = call("GET", "/v1/tasks")
    assert resp.status == 200
    assert len(resp.body["tasks"]) == 1

    # Get
    resp = call("GET", f"/v1/tasks/{task_id}")
    assert resp.status == 200

    # Validate
    resp = call("POST", f"/v1/tasks/{task_id}/validate")
    assert resp.status == 200
    assert resp.body["valid"] is True

    # History (empty)
    resp = call("GET", f"/v1/tasks/{task_id}/history")
    assert resp.status == 200
    assert resp.body["versions"] == []

    # Add two dataset versions
    reg.append_dataset_version("acme", task_id,
                               records=[{"title": "A", "price": "$10"}])
    reg.append_dataset_version("acme", task_id,
                               records=[{"title": "A", "price": "$12"},
                                        {"title": "B", "price": "$20"}])

    # History now has 2 entries
    resp = call("GET", f"/v1/tasks/{task_id}/history")
    assert resp.status == 200
    assert len(resp.body["versions"]) == 2

    # Changes
    resp = call("GET", f"/v1/tasks/{task_id}/changes")
    assert resp.status == 200
    assert resp.body["summary"]["new"] == 1
    assert resp.body["summary"]["modified"] == 1

    # Quality
    resp = call("GET", f"/v1/tasks/{task_id}/quality")
    assert resp.status == 200
    assert resp.body["version"] == 2

    # Unknown task -> 404
    resp = call("GET", "/v1/tasks/nope")
    assert resp.status == 404

    # Cross-tenant isolation: another client cannot see acme's task
    _, other_key = store.create("other")
    resp = call("GET", f"/v1/tasks/{task_id}", key=other_key)
    assert resp.status == 404

    # Delete
    resp = call("DELETE", f"/v1/tasks/{task_id}")
    assert resp.status == 204
    resp = call("GET", f"/v1/tasks/{task_id}")
    assert resp.status == 404

    print("API handlers OK.")