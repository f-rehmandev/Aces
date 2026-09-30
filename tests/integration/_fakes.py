"""
In-memory fakes that mirror the shape of the supabase-py client.

These are used by integration tests that want to exercise the REAL
storage-layer code (`src.storage.batch_store.BatchStore`,
`src.storage.task_store.TaskStore`, etc.) without a live database.

The fake chain supports the subset of the fluent API our storage
layer actually uses:

    client.table(name)
        .select(cols)
        .insert(row_or_list)
        .upsert(row_or_list, on_conflict=...)
        .update(patch)
        .delete()
        .eq(col, val)             # multiple calls AND together
        .order(col, desc=bool)
        .limit(n)
        .execute()                # -> response with .data
"""
from __future__ import annotations

import copy
from typing import Any


class FakeResponse:
    def __init__(self, data: Any = None):
        self.data = data if data is not None else []


class _Query:
    def __init__(self, table: "_Table"):
        self._table = table
        self._op: str = "select"
        self._cols: str = "*"
        self._payload: Any = None
        self._on_conflict: str | None = None
        self._filters: list[tuple[str, Any]] = []
        self._order_col: str | None = None
        self._order_desc: bool = False
        self._limit: int | None = None

    # --- builders --------------------------------------------------
    def select(self, cols: str = "*"):
        self._op = "select"
        self._cols = cols
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = copy.deepcopy(payload)
        return self

    def upsert(self, payload, on_conflict: str | None = None):
        self._op = "upsert"
        self._payload = copy.deepcopy(payload)
        self._on_conflict = on_conflict
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = copy.deepcopy(payload)
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col: str, val: Any):
        self._filters.append((col, val))
        return self

    def order(self, col: str, desc: bool = False):
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    def _sort_key(self, row: dict):
        """
        Robust sort key for the `order()` clause.

        The naive `row.get(col) or ""` idiom silently coerces every
        falsy value — including the legitimate integer 0 — into an
        empty string. On a real Postgres column (integer chunk_index)
        this never happens because the database sorts server-side.
        Our fake has to sort in Python, where `["", 1]` blows up with
        TypeError.

        This version preserves the raw value for anything non-None, so
        homogeneous columns (which real Postgres guarantees, and which
        every test in this file relies on) sort correctly. If a column
        ever holds mixed types, we tag by type-name first so the
        comparison never crashes.
        """
        v = row.get(self._order_col)
        if v is None:
            # None sorts last. Bucket 1 vs bucket 0.
            return (1, "", 0)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return (0, "", v)
        return (0, str(v), 0)

    # --- terminal --------------------------------------------------
    def execute(self):
        rows = self._table._rows

        def _matches(row):
            return all(row.get(k) == v for k, v in self._filters)

        if self._op == "select":
            result = [copy.deepcopy(r) for r in rows if _matches(r)]
            if self._order_col:
                result.sort(
                    key=self._sort_key,
                    reverse=self._order_desc,
                )
            if self._limit is not None:
                result = result[: self._limit]
            return FakeResponse(result)

        if self._op == "insert":
            payload = self._payload
            if isinstance(payload, list):
                rows.extend(copy.deepcopy(payload))
                return FakeResponse(copy.deepcopy(payload))
            rows.append(copy.deepcopy(payload))
            return FakeResponse([copy.deepcopy(payload)])

        if self._op == "upsert":
            payload = self._payload
            # Normalize to a list of rows
            items = payload if isinstance(payload, list) else [payload]
            conflict_cols = (
                [c.strip() for c in self._on_conflict.split(",")]
                if self._on_conflict else []
            )
            written: list[dict] = []
            for item in items:
                merged = None
                if conflict_cols:
                    for i, existing in enumerate(rows):
                        if all(
                            existing.get(c) == item.get(c)
                            for c in conflict_cols
                        ):
                            merged = {**existing, **copy.deepcopy(item)}
                            rows[i] = merged
                            break
                if merged is None:
                    rows.append(copy.deepcopy(item))
                    merged = copy.deepcopy(item)
                written.append(merged)
            return FakeResponse(written)

        if self._op == "update":
            changed = []
            for row in rows:
                if _matches(row):
                    row.update(copy.deepcopy(self._payload))
                    changed.append(copy.deepcopy(row))
            return FakeResponse(changed)

        if self._op == "delete":
            kept, deleted = [], []
            for row in rows:
                if _matches(row):
                    deleted.append(copy.deepcopy(row))
                else:
                    kept.append(row)
            rows[:] = kept
            return FakeResponse(deleted)

        raise AssertionError(f"unknown op: {self._op}")


class _Table:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    # _Query-based entry points
    def select(self, cols: str = "*"):
        return _Query(self).select(cols)

    def insert(self, payload):
        return _Query(self).insert(payload)

    def upsert(self, payload, on_conflict: str | None = None):
        return _Query(self).upsert(payload, on_conflict=on_conflict)

    def update(self, payload):
        return _Query(self).update(payload)

    def delete(self):
        return _Query(self).delete()


class FakeSupabaseClient:
    """
    Drop-in replacement for `src.storage.db.get_client()`.

    Tables are created lazily on first access, so callers don't need to
    pre-declare which tables a test will touch. Each table's rows are
    stored on the client and persist across `.table()` calls — the same
    way a real database would.
    """
    def __init__(self):
        self._tables: dict[str, list[dict]] = {}

    def table(self, name: str) -> _Table:
        rows = self._tables.setdefault(name, [])
        return _Table(rows)