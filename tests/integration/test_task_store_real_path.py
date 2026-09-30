"""
Integration test: real `TaskStore` through the real Supabase client
shape, with only the transport swapped for an in-memory fake.

This exercises the exact fluent-API chains that run in production:

    TaskStore.save_task   -> client.table("tasks").select(...).eq(...).limit(...)
    TaskStore.save_task   -> client.table("tasks").insert(...)
    TaskStore.save_task   -> client.table("task_versions").insert(...)
    TaskStore.get_task    -> client.table("task_versions").select(...).eq(...)
    TaskStore.list_tasks  -> client.table("tasks").select(...).eq(...).order(...)
    TaskStore.delete_task -> client.table("tasks").delete().eq(...)

The level is deliberately one above the unit tests in
`tests/unit/test_task_store.py`: those use a bespoke fake client
tuned to the specific query chains the test expected. This file uses
the generic fake from `tests.integration._fakes` — which knows
nothing about tasks — so the real query chains are the ones under
test.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from src.core.task_spec import FieldSpec, Target, TaskSpec
from tests.integration._fakes import FakeSupabaseClient


@contextmanager
def _patched_get_client():
    """
    Swap `src.storage.db.get_client` for the shared in-memory fake
    for the duration of the `with` block. Identical pattern to
    test_batch_store_real_path.py — kept inline rather than a fixture
    so the patch scope is visibly bounded.
    """
    import src.storage.db as db_mod
    original = db_mod.get_client
    fake = FakeSupabaseClient()
    db_mod.get_client = lambda: fake
    try:
        yield fake
    finally:
        db_mod.get_client = original


def _make_task(
    task_id: str = "t-store-1",
    prompt: str = "find laptop prices",
) -> TaskSpec:
    spec = TaskSpec(
        natural_language_prompt=prompt,
        target=Target(start_urls=["https://example.com/laptops"]),
        fields=[FieldSpec(name="title"), FieldSpec(name="price", type="currency")],
    )
    spec.task_id = task_id
    return spec


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_save_and_get_round_trip():
    from src.storage.task_store import TaskStore

    with _patched_get_client() as client:
        store = TaskStore()
        spec = _make_task()

        version = store.save_task("acme", spec)
        assert version == 1

        # The tasks row exists with current_version=1.
        tasks_rows = client._tables["tasks"]
        assert len(tasks_rows) == 1
        assert tasks_rows[0]["task_id"] == spec.task_id
        assert tasks_rows[0]["current_version"] == 1
        assert tasks_rows[0]["client_id"] == "acme"

        # The task_versions row exists with the serialized spec.
        versions_rows = client._tables["task_versions"]
        assert len(versions_rows) == 1
        assert versions_rows[0]["task_id"] == spec.task_id
        assert versions_rows[0]["version"] == 1
        assert versions_rows[0]["client_id"] == "acme"
        assert isinstance(versions_rows[0]["spec"], dict)

        # Load it back.
        loaded = store.get_task("acme", spec.task_id)
        assert loaded is not None
        assert loaded.task_id == spec.task_id
        assert loaded.natural_language_prompt == "find laptop prices"
        assert loaded.target.start_urls == ["https://example.com/laptops"]
        assert loaded.client_id == "acme"
        assert [f.name for f in loaded.fields] == ["title", "price"]


def test_identical_save_does_not_create_a_new_version():
    from src.storage.task_store import TaskStore

    with _patched_get_client() as client:
        store = TaskStore()
        spec = _make_task()

        v1 = store.save_task("acme", spec)
        v2 = store.save_task("acme", spec)

        assert v1 == 1
        assert v2 == 1
        assert len(client._tables["task_versions"]) == 1


def test_changed_save_creates_new_version_and_bumps_current():
    from src.storage.task_store import TaskStore

    with _patched_get_client() as client:
        store = TaskStore()

        first = _make_task()
        store.save_task("acme", first)

        # Same task_id, different prompt => new version.
        second = _make_task(prompt="find gaming laptops")
        v2 = store.save_task("acme", second)

        assert v2 == 2
        assert len(client._tables["task_versions"]) == 2

        tasks_row = client._tables["tasks"][0]
        assert tasks_row["current_version"] == 2

        # get_task without an explicit version returns the latest.
        latest = store.get_task("acme", first.task_id)
        assert latest.natural_language_prompt == "find gaming laptops"

        # get_task with version=1 returns the older one.
        old = store.get_task("acme", first.task_id, version=1)
        assert old.natural_language_prompt == "find laptop prices"


# ---------------------------------------------------------------------------
# Tenant scoping
# ---------------------------------------------------------------------------

def test_get_task_enforces_client_id():
    from src.storage.task_store import TaskStore

    with _patched_get_client():
        store = TaskStore()
        spec = _make_task()
        store.save_task("acme", spec)

        assert store.get_task("acme", spec.task_id) is not None
        assert store.get_task("other", spec.task_id) is None


def test_list_tasks_scoped_by_client():
    from src.storage.task_store import TaskStore

    with _patched_get_client():
        store = TaskStore()
        store.save_task("acme", _make_task(task_id="a-1", prompt="A"))
        store.save_task("acme", _make_task(task_id="a-2", prompt="B"))
        store.save_task("other", _make_task(task_id="o-1", prompt="C"))

        acme = store.list_tasks("acme")
        other = store.list_tasks("other")

        assert {t.task_id for t in acme} == {"a-1", "a-2"}
        assert {t.task_id for t in other} == {"o-1"}


def test_delete_task_scoped_by_client():
    from src.storage.task_store import TaskStore

    with _patched_get_client():
        store = TaskStore()
        spec = _make_task()
        store.save_task("acme", spec)

        # Wrong tenant cannot delete.
        assert store.delete_task("other", spec.task_id) is False
        assert store.get_task("acme", spec.task_id) is not None

        # Right tenant can.
        assert store.delete_task("acme", spec.task_id) is True
        assert store.get_task("acme", spec.task_id) is None


def test_delete_removes_task_row_but_task_versions_stay():
    """
    Document current behaviour: `delete_task` deletes the tasks row
    only. In production the FK cascade handles the versions. In our
    fake, we assert what we can observe without a cascade.
    """
    from src.storage.task_store import TaskStore

    with _patched_get_client() as client:
        store = TaskStore()
        spec = _make_task()
        store.save_task("acme", spec)

        assert len(client._tables["tasks"]) == 1
        assert len(client._tables["task_versions"]) == 1

        store.delete_task("acme", spec.task_id)

        assert len(client._tables["tasks"]) == 0
        # get_task must return None because the outer tasks row is gone.
        assert store.get_task("acme", spec.task_id) is None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

def test_save_task_requires_client_id():
    from src.storage.task_store import TaskStore

    with _patched_get_client():
        store = TaskStore()
        with pytest.raises(ValueError):
            store.save_task("", _make_task())


def test_get_task_requires_client_id():
    from src.storage.task_store import TaskStore

    with _patched_get_client():
        store = TaskStore()
        with pytest.raises(ValueError):
            store.get_task("", "anything")


def test_get_task_missing_returns_none():
    from src.storage.task_store import TaskStore

    with _patched_get_client():
        store = TaskStore()
        assert store.get_task("acme", "never-saved") is None