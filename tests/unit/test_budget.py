# ---------------------------------------------------------------------------
# Wire <-> runtime Budget bridge (Task 13)
# ---------------------------------------------------------------------------

def test_from_wire_copies_limit_fields():
    from src.core.task_spec import Budget as WireBudget
    from src.jobs.budget import Budget as RuntimeBudget

    wire = WireBudget(
        max_llm_tokens=12_345,
        max_usd=2.50,
        max_scraperapi_credits=42,
        max_pages=99,
    )
    runtime = RuntimeBudget.from_wire(wire)

    assert runtime.max_llm_tokens == 12_345
    assert runtime.max_usd == 2.50
    assert runtime.max_scraperapi_credits == 42
    assert runtime.max_pages == 99

    # Runtime-only fields default to zero/1.
    assert runtime.llm_tokens_used == 0
    assert runtime.usd_used == 0.0
    assert runtime.pages_used == 0


def test_from_wire_falls_back_when_field_missing():
    """
    A forward-compatible wire object that lacks a field must not crash;
    the runtime Budget's own default is used instead.
    """
    from src.jobs.budget import Budget as RuntimeBudget

    class MinimalWire:
        max_llm_tokens = 500

    runtime = RuntimeBudget.from_wire(MinimalWire())
    assert runtime.max_llm_tokens == 500
    assert runtime.max_usd == 0.50       # default
    assert runtime.max_scraperapi_credits == 0
    assert runtime.max_pages == 5_000