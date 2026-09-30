"""Unit tests for persistent TaskStore."""
from __future__ import annotations

from copy import deepcopy

import pytest

from src.core.task_spec import TaskSpec, Target
from src.storage.task_store import TaskStore, TaskStoreError


class FakeResponse:
    def __init__(self, data=None):
        self.data = data or []


class FakeQuery:
    def __init__(self, table):
        self.table = table
        self.operation = "select"
        self.payload = None
        self.filters = []
        self.order_field = None
        self.order_desc = False
        self.limit_value = None

    def select(self, *_fields):
        self.operation = "select"
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = deepcopy(payload)
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = deepcopy(payload)
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def eq(self, field, value):
        self.filters.append((field, value))
        return self

    def order(self, field, desc=False):
        self.order_field = field
        self.order_desc = desc
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        return self.table.execute(self)


class FakeTable:
    def __init__(self, name, db):
        self.name = name
        self.db = db

    def _call_query(self):
        return FakeQuery(self)

    def execute(self, query: FakeQuery):
        rows = self.db.setdefault(self.name, [])

        def matches(row):
            return all(row.get(k) == v for k, v in query.filters)

        if query.operation == "select":
            result = [deepcopy(row) for row in rows if matches(row)]

            if query.order_field:
                result.sort(
                    key=lambda row: row.get(query.order_field) or "",
                    reverse=query.order_desc,
                )

            if query.limit_value is not None:
                result = result[: query.limit_value]

            return FakeResponse(result)

        if query.operation == "insert":
            payload = deepcopy(query.payload)

            if isinstance(payload, list):
                rows.extend(payload)
            else:
                rows.append(payload)

            return FakeResponse([deepcopy(payload)])

        if query.operation == "update":
            changed = []

            for row in rows:
                if matches(row):
                    row.update(deepcopy(query.payload))
                    changed.append(deepcopy(row))

            return FakeResponse(changed)

        if query.operation == "delete":
            kept = []
            deleted = []

            for row in rows:
                if matches(row):
                    deleted.append(deepcopy(row))
                else:
                    kept.append(row)

            rows[:] = kept
            return FakeResponse(deleted)

        raise AssertionError(query.operation)


class FakeClient:
    def __init__(self):
        self.db = {}

    def table(self, name):
        return FakeTableProxy(FakeTable(name, self.db))


class FakeTableProxy:
    def __init__(self, table):
        self.table = table

    def select(self, *fields):
        return self.table._call_query().select(*fields)

    def insert(self, payload):
        return self.table._call_query().insert(payload)

    def update(self, payload):
        return self.table._call_query().update(payload)

    def delete(self):
        return self.table._call_query().delete()


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(
        "src.storage.task_store.get_client",
        lambda: client,
    )
    return client


def test_save_and_get_new_task(fake_client):
    store = TaskStore()

    spec = TaskSpec(
        natural_language_prompt="find laptops",
        target=Target(
            start_urls=["https://example.com/laptops"]
        ),
    )

    version = store.save_task("acme", spec)

    assert version == 1

    restored = store.get_task("acme", spec.task_id)

    assert restored is not None
    assert restored.task_id == spec.task_id
    assert restored.natural_language_prompt == "find laptops"
    assert restored.client_id == "acme"


def test_identical_save_does_not_create_new_version(fake_client):
    store = TaskStore()

    spec = TaskSpec(
        natural_language_prompt="find laptops",
    )

    assert store.save_task("acme", spec) == 1
    assert store.save_task("acme", spec) == 1

    assert len(fake_client.db["task_versions"]) == 1


def test_changed_save_creates_new_version(fake_client):
    store = TaskStore()

    spec = TaskSpec(
        natural_language_prompt="find laptops",
    )

    assert store.save_task("acme", spec) == 1

    changed = TaskSpec.from_dict(spec.to_dict())
    changed.natural_language_prompt = "find gaming laptops"

    assert store.save_task("acme", changed) == 2

    latest = store.get_task("acme", spec.task_id)
    old = store.get_task("acme", spec.task_id, version=1)

    assert latest.natural_language_prompt == "find gaming laptops"
    assert old.natural_language_prompt == "find laptops"


def test_list_tasks_is_tenant_scoped(fake_client):
    store = TaskStore()

    a = TaskSpec(natural_language_prompt="A")
    b = TaskSpec(natural_language_prompt="B")

    store.save_task("acme", a)
    store.save_task("other", b)

    acme = store.list_tasks("acme")
    other = store.list_tasks("other")

    assert [s.task_id for s in acme] == [a.task_id]
    assert [s.task_id for s in other] == [b.task_id]


def test_list_client_ids(fake_client):
    store = TaskStore()

    store.save_task("acme", TaskSpec())
    store.save_task("beta", TaskSpec())

    assert store.list_client_ids() == ["acme", "beta"]


def test_delete_task(fake_client):
    store = TaskStore()

    spec = TaskSpec()
    store.save_task("acme", spec)

    assert store.delete_task("acme", spec.task_id) is True
    assert store.get_task("acme", spec.task_id) is None
    assert store.delete_task("acme", spec.task_id) is False


def test_cross_tenant_task_is_invisible(fake_client):
    store = TaskStore()

    spec = TaskSpec()
    store.save_task("acme", spec)

    assert store.get_task("other", spec.task_id) is None
    assert store.delete_task("other", spec.task_id) is False


def test_requires_client_id():
    store = TaskStore()

    with pytest.raises(ValueError):
        store.get_task("", "x")


def test_store_error_wraps_backend_failure(monkeypatch):
    class BrokenClient:
        def table(self, _name):
            raise RuntimeError("database offline")

    monkeypatch.setattr(
        "src.storage.task_store.get_client",
        lambda: BrokenClient(),
    )

    store = TaskStore()

    with pytest.raises(TaskStoreError, match="database offline"):
        store.list_client_ids()