"""Unit tests for usage store (spec §41.4)."""
import asyncio

import pytest

from src.usage.store import (
    InMemoryUsageStore,
    _in_window,
    _row_to_event,
    _summarize,
)
from src.usage.types import (
    ResourceType,
    UsageEvent,
    UsageSummary,
    unit_for,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_event_unit_autopopulated():
    e = UsageEvent(resource_type=ResourceType.TOKEN, quantity=5)
    assert e.unit == "token"


def test_event_total_computed_from_unit_cost():
    e = UsageEvent(
        resource_type=ResourceType.TOKEN,
        quantity=1000, unit_cost_snapshot=0.000075,
    )
    assert abs(e.total_cost_usd - 0.075) < 1e-9


def test_event_explicit_total_wins():
    e = UsageEvent(
        resource_type=ResourceType.PAGE,
        quantity=5, unit_cost_snapshot=0.01,
        total_cost_usd=99.99,
    )
    assert e.total_cost_usd == 99.99


def test_event_zero_cost():
    e = UsageEvent(resource_type=ResourceType.PAGE, quantity=1)
    assert e.total_cost_usd == 0.0


def test_event_negative_quantity_rejected():
    with pytest.raises(ValueError):
        UsageEvent(quantity=-1)


def test_event_negative_unit_cost_rejected():
    with pytest.raises(ValueError):
        UsageEvent(unit_cost_snapshot=-0.01)


def test_event_negative_total_rejected():
    with pytest.raises(ValueError):
        UsageEvent(total_cost_usd=-1.0)


def test_event_roundtrip():
    e = UsageEvent(
        client_id="acme", job_id="j-1",
        resource_type=ResourceType.TOKEN,
        quantity=1000, unit_cost_snapshot=0.000075,
        provider="gemini",
    )
    d = e.to_dict()
    assert d["resource_type"] == "token"
    e2 = UsageEvent.from_dict(d)
    assert e2.quantity == 1000
    assert e2.provider == "gemini"


def test_event_from_dict_bad_resource_type():
    e = UsageEvent.from_dict({"resource_type": "nonsense"})
    assert e.resource_type == ResourceType.PAGE


def test_unit_for_helper():
    assert unit_for(ResourceType.TOKEN) == "token"
    assert unit_for("provider_credit") == "credit"
    assert unit_for("bogus") == "unit"


# ---------------------------------------------------------------------------
# In-memory store: writes
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_empty_summary():
    s = InMemoryUsageStore()
    summary = _run(s.summarize("acme"))
    assert summary.event_count == 0
    assert summary.total_cost_usd == 0.0


def test_record_single():
    s = InMemoryUsageStore()
    _run(s.record(UsageEvent(
        client_id="acme", job_id="j-1",
        resource_type=ResourceType.PAGE,
        quantity=5, unit_cost_snapshot=0.01,
    )))
    assert len(s) == 1


def test_record_batch():
    s = InMemoryUsageStore()
    n = _run(s.record_batch([
        UsageEvent(client_id="acme", resource_type=ResourceType.PAGE, quantity=1),
        UsageEvent(client_id="acme", resource_type=ResourceType.TOKEN, quantity=100),
    ]))
    assert n == 2
    assert len(s) == 2


def test_record_batch_idempotent_on_event_id():
    s = InMemoryUsageStore()
    e = UsageEvent(resource_type=ResourceType.PAGE, quantity=1)
    _run(s.record(e))
    n = _run(s.record_batch([e]))
    assert n == 0
    assert len(s) == 1


def test_record_ignores_duplicate_id():
    s = InMemoryUsageStore()
    e = UsageEvent(resource_type=ResourceType.PAGE, quantity=1)
    _run(s.record(e))
    _run(s.record(e))
    assert len(s) == 1


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def test_summary_by_resource():
    s = InMemoryUsageStore()
    _run(s.record_batch([
        UsageEvent(client_id="acme", resource_type=ResourceType.PAGE, quantity=5,
                   unit_cost_snapshot=0.01),
        UsageEvent(client_id="acme", resource_type=ResourceType.PAGE, quantity=3,
                   unit_cost_snapshot=0.01),
        UsageEvent(client_id="acme", resource_type=ResourceType.TOKEN, quantity=1000,
                   unit_cost_snapshot=0.000075),
    ]))
    summary = _run(s.summarize("acme"))
    assert summary.event_count == 3
    assert summary.by_resource["page"]["event_count"] == 2
    assert summary.by_resource["page"]["quantity"] == 8.0
    assert summary.by_resource["token"]["quantity"] == 1000.0
    # 8 * 0.01 + 1000 * 0.000075 = 0.08 + 0.075 = 0.155
    assert abs(summary.total_cost_usd - 0.155) < 1e-9


def test_summary_client_isolation():
    s = InMemoryUsageStore()
    _run(s.record(UsageEvent(client_id="acme", resource_type=ResourceType.PAGE)))
    _run(s.record(UsageEvent(client_id="other", resource_type=ResourceType.PAGE)))
    assert _run(s.summarize("acme")).event_count == 1
    assert _run(s.summarize("other")).event_count == 1
    assert _run(s.summarize("nobody")).event_count == 0


def test_summary_per_job():
    s = InMemoryUsageStore()
    _run(s.record_batch([
        UsageEvent(client_id="acme", job_id="j-1", resource_type=ResourceType.PAGE),
        UsageEvent(client_id="acme", job_id="j-1", resource_type=ResourceType.TOKEN),
        UsageEvent(client_id="acme", job_id="j-2", resource_type=ResourceType.PAGE),
    ]))
    s_j1 = _run(s.summarize("acme", job_id="j-1"))
    assert s_j1.event_count == 2
    s_j2 = _run(s.summarize("acme", job_id="j-2"))
    assert s_j2.event_count == 1


def test_summary_window_filter():
    s = InMemoryUsageStore()
    _run(s.record(UsageEvent(
        client_id="acme",
        resource_type=ResourceType.PAGE,
        occurred_at="2026-01-15T00:00:00+00:00",
    )))
    _run(s.record(UsageEvent(
        client_id="acme",
        resource_type=ResourceType.PAGE,
        occurred_at="2026-06-15T00:00:00+00:00",
    )))
    # Only the June event falls in this window
    summary = _run(s.summarize(
        "acme",
        since="2026-05-01T00:00:00+00:00",
        until="2026-07-01T00:00:00+00:00",
    ))
    assert summary.event_count == 1


# ---------------------------------------------------------------------------
# Event queries
# ---------------------------------------------------------------------------

def test_events_for_job():
    s = InMemoryUsageStore()
    _run(s.record_batch([
        UsageEvent(client_id="acme", job_id="j-1", resource_type=ResourceType.PAGE),
        UsageEvent(client_id="acme", job_id="j-1", resource_type=ResourceType.TOKEN),
        UsageEvent(client_id="acme", job_id="j-2", resource_type=ResourceType.PAGE),
    ]))
    evs = _run(s.events_for_job("j-1"))
    assert len(evs) == 2
    evs = _run(s.events_for_job("nonexistent"))
    assert evs == []


def test_events_for_client():
    s = InMemoryUsageStore()
    _run(s.record(UsageEvent(client_id="acme", resource_type=ResourceType.PAGE)))
    _run(s.record(UsageEvent(client_id="other", resource_type=ResourceType.PAGE)))
    assert len(_run(s.events_for_client("acme"))) == 1
    assert len(_run(s.events_for_client("other"))) == 1


def test_events_for_client_respects_limit():
    s = InMemoryUsageStore()
    _run(s.record_batch([
        UsageEvent(client_id="acme", resource_type=ResourceType.PAGE)
        for _ in range(10)
    ]))
    evs = _run(s.events_for_client("acme", limit=3))
    assert len(evs) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_in_window():
    assert _in_window("2026-06-15T00:00:00+00:00", "", "")
    assert _in_window(
        "2026-06-15T00:00:00+00:00",
        "2026-06-01T00:00:00+00:00",
        "2026-07-01T00:00:00+00:00",
    )
    assert not _in_window(
        "2026-01-15T00:00:00+00:00",
        "2026-06-01T00:00:00+00:00",
        "2026-07-01T00:00:00+00:00",
    )
    assert not _in_window(
        "2026-08-15T00:00:00+00:00",
        "2026-06-01T00:00:00+00:00",
        "2026-07-01T00:00:00+00:00",
    )


def test_summarize_helper_empty():
    s = _summarize("acme", [], "", "")
    assert s.event_count == 0
    assert s.total_cost_usd == 0.0
    assert s.by_resource == {}


def test_summarize_helper_multi_resource():
    e1 = UsageEvent(
        client_id="acme", resource_type=ResourceType.PAGE,
        quantity=5, unit_cost_snapshot=0.01,
    )
    e2 = UsageEvent(
        client_id="acme", resource_type=ResourceType.TOKEN,
        quantity=1000, unit_cost_snapshot=0.000075,
    )
    s = _summarize("acme", [e1, e2], "", "")
    assert s.event_count == 2
    assert "page" in s.by_resource
    assert "token" in s.by_resource


def test_row_to_event_coerces_numerics():
    row = {
        "event_id": "x",
        "client_id": "acme",
        "resource_type": "token",
        "quantity": "1000",
        "unit_cost_snapshot": "0.000075",
        "total_cost_usd": "0.075",
        "metadata": None,
        "occurred_at": "2026-06-15T00:00:00+00:00",
    }
    e = _row_to_event(row)
    assert e.quantity == 1000.0
    assert abs(e.unit_cost_snapshot - 0.000075) < 1e-12
    assert e.metadata == {}


def test_row_to_event_handles_bad_numbers():
    row = {
        "event_id": "x", "client_id": "acme",
        "resource_type": "page",
        "quantity": "not-a-number",
    }
    e = _row_to_event(row)
    assert e.quantity == 0.0