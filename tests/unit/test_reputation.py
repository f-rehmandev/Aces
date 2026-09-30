"""Unit tests for source reputation (spec §24)."""
from datetime import datetime, timedelta, timezone

from src.trust.reputation import (
    ReputationStore, SourceReputation,
    MIN_TRUST, MAX_TRUST, INITIAL_TRUST, DECAY_DAYS,
)


# --- neutral start ----------------------------------------------------

def test_new_domain_starts_neutral():
    store = ReputationStore()
    assert store.trust("brand-new.example") == INITIAL_TRUST


def test_get_creates_record_with_domain():
    store = ReputationStore()
    rec = store.get("x.com")
    assert rec.domain == "x.com"
    assert rec.trust_score == INITIAL_TRUST
    assert rec.last_updated


def test_get_lowercases_domain():
    store = ReputationStore()
    rec = store.get("Example.COM")
    assert rec.domain == "example.com"


# --- agreement / disagreement ----------------------------------------

def test_agreement_raises_trust():
    store = ReputationStore()
    for _ in range(5):
        store.record_agreement("good.example")
    assert store.trust("good.example") > INITIAL_TRUST


def test_disagreement_lowers_trust():
    store = ReputationStore()
    for _ in range(5):
        store.record_disagreement("bad.example")
    assert store.trust("bad.example") < INITIAL_TRUST


def test_mixed_agreement_is_between():
    store = ReputationStore()
    for _ in range(3):
        store.record_agreement("mixed.example")
    for _ in range(3):
        store.record_disagreement("mixed.example")
    score = store.trust("mixed.example")
    assert MIN_TRUST <= score <= MAX_TRUST


def test_staleness_lowers_trust():
    store = ReputationStore()
    for _ in range(5):
        store.record_stale("stale.example")
    assert store.trust("stale.example") < INITIAL_TRUST


def test_schema_violation_lowers_trust():
    store = ReputationStore()
    for _ in range(5):
        store.record_schema_violation("broken.example")
    assert store.trust("broken.example") < INITIAL_TRUST


# --- clamping --------------------------------------------------------

def test_trust_never_exceeds_max():
    store = ReputationStore()
    for _ in range(500):
        store.record_agreement("perfect.example")
    assert store.trust("perfect.example") <= MAX_TRUST


def test_trust_never_below_min():
    store = ReputationStore()
    for _ in range(500):
        store.record_disagreement("awful.example")
    assert store.trust("awful.example") >= MIN_TRUST


# --- manual authority -------------------------------------------------

def test_manual_authority_boosts_trust():
    store = ReputationStore()
    before = store.trust("brand.example")
    store.set_manual_authority("brand.example", True)
    assert store.trust("brand.example") > before


def test_manual_authority_can_be_revoked():
    store = ReputationStore()
    store.set_manual_authority("x.example", True)
    with_auth = store.trust("x.example")
    store.set_manual_authority("x.example", False)
    assert store.trust("x.example") < with_auth


# --- decay ------------------------------------------------------------

def test_decay_resets_stale_source():
    store = ReputationStore()
    rec = store.get("old.example")
    rec.trust_score = 0.95
    rec.observation_count = 50
    rec.agreement_count = 50
    old = datetime.now(timezone.utc) - timedelta(days=DECAY_DAYS + 10)
    rec.last_updated = old.isoformat(timespec="seconds")

    store.apply_decay()
    assert store.trust("old.example") == INITIAL_TRUST
    rec2 = store.get("old.example")
    assert rec2.observation_count == 0
    assert rec2.agreement_count == 0


def test_decay_skips_recent_source():
    store = ReputationStore()
    for _ in range(5):
        store.record_agreement("recent.example")
    score_before = store.trust("recent.example")
    store.apply_decay()
    assert store.trust("recent.example") == score_before


# --- serialization ----------------------------------------------------

def test_to_list_and_from_list_roundtrip():
    store = ReputationStore()
    for _ in range(3):
        store.record_agreement("good.example")
    rows = store.to_list()
    store2 = ReputationStore.from_list(rows)
    assert store2.trust("good.example") == store.trust("good.example")


def test_from_list_handles_empty():
    store = ReputationStore.from_list([])
    assert store.to_list() == []


def test_from_list_handles_none():
    store = ReputationStore.from_list(None)
    assert store.to_list() == []


def test_all_returns_every_domain():
    store = ReputationStore()
    store.record_agreement("a.example")
    store.record_agreement("b.example")
    domains = {r.domain for r in store.all()}
    assert domains == {"a.example", "b.example"}


# --- observation counter ----------------------------------------------

def test_observation_count_increments():
    store = ReputationStore()
    for _ in range(7):
        store.record_agreement("counter.example")
    assert store.get("counter.example").observation_count == 7


def test_trust_score_is_clamped_on_record():
    store = ReputationStore()
    rec = store.get("x.example")
    rec.trust_score = 5.0
    assert store.trust("x.example") == 5.0   # we don't clamp on read;
                                              # only _compute clamps
    # but any update re-clamps
    store.record_agreement("x.example")
    assert store.trust("x.example") <= MAX_TRUST