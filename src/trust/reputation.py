"""
Source reputation scoring — spec §24.

ACES learns which sources consistently produce good, agreed-with data.
Each domain has a trust score in [0.05, 0.99], updated after every
triangulated run.

Storage is in-memory here; a Supabase-backed store can subclass to
persist the same fields.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Domain record
# ---------------------------------------------------------------------------

@dataclass
class SourceReputation:
    domain: str
    trust_score: float = 0.5
    observation_count: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    staleness_flags: int = 0
    schema_violations: int = 0
    manual_authority: bool = False
    last_updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Constants (spec §24.2)
# ---------------------------------------------------------------------------

MIN_TRUST = 0.05
MAX_TRUST = 0.99
INITIAL_TRUST = 0.5
DECAY_DAYS = 90
LAPLACE_ALPHA = 1.0   # smoothing for brand-new sources


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ReputationStore:
    """
    In-memory reputation store. Persistence can be layered on top by
    loading `to_list()` from Supabase at startup and pushing updates after
    each run.
    """

    def __init__(self):
        self._by_domain: dict[str, SourceReputation] = {}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get(self, domain: str) -> SourceReputation:
        """Return the record for `domain`, creating a neutral one if new."""
        d = (domain or "").lower()
        if d not in self._by_domain:
            self._by_domain[d] = SourceReputation(
                domain=d,
                trust_score=INITIAL_TRUST,
                last_updated=_iso(_utc_now()),
            )
        return self._by_domain[d]

    def trust(self, domain: str) -> float:
        return self.get(domain).trust_score

    def all(self) -> list[SourceReputation]:
        return list(self._by_domain.values())

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self._by_domain.values()]

    @classmethod
    def from_list(cls, rows: list[dict]) -> "ReputationStore":
        store = cls()
        for row in rows or []:
            rec = SourceReputation(
                domain=str(row.get("domain", "")).lower(),
                trust_score=float(row.get("trust_score", INITIAL_TRUST)),
                observation_count=int(row.get("observation_count", 0)),
                agreement_count=int(row.get("agreement_count", 0)),
                disagreement_count=int(row.get("disagreement_count", 0)),
                staleness_flags=int(row.get("staleness_flags", 0)),
                schema_violations=int(row.get("schema_violations", 0)),
                manual_authority=bool(row.get("manual_authority", False)),
                last_updated=str(row.get("last_updated", "")),
            )
            if rec.domain:
                store._by_domain[rec.domain] = rec
        return store

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def record_agreement(self, domain: str) -> None:
        self._update(domain, agreed=True, stale=False, schema_violation=False)

    def record_disagreement(self, domain: str) -> None:
        self._update(domain, agreed=False, stale=False, schema_violation=False)

    def record_stale(self, domain: str) -> None:
        self._update(domain, agreed=None, stale=True, schema_violation=False)

    def record_schema_violation(self, domain: str) -> None:
        self._update(domain, agreed=None, stale=False, schema_violation=True)

    def set_manual_authority(self, domain: str, is_authority: bool = True) -> None:
        """Human override — a source is treated as authoritative."""
        rec = self.get(domain)
        rec.manual_authority = bool(is_authority)
        rec.trust_score = self._compute(rec)
        rec.last_updated = _iso(_utc_now())

    def apply_decay(self, now: Optional[datetime] = None) -> None:
        """
        §24.5 — trust decays toward 0.5 if a source has not been observed
        in DECAY_DAYS. Call this periodically (e.g. at run start).
        """
        now = now or _utc_now()
        for rec in self._by_domain.values():
            if not rec.last_updated:
                continue
            try:
                last = datetime.fromisoformat(rec.last_updated)
            except ValueError:
                continue
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if (now - last).days >= DECAY_DAYS:
                rec.trust_score = INITIAL_TRUST
                rec.observation_count = 0
                rec.agreement_count = 0
                rec.disagreement_count = 0
                rec.staleness_flags = 0
                rec.schema_violations = 0
                rec.last_updated = _iso(now)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _update(
        self,
        domain: str,
        agreed: Optional[bool],
        stale: bool,
        schema_violation: bool,
    ) -> None:
        rec = self.get(domain)
        rec.observation_count += 1
        if agreed is True:
            rec.agreement_count += 1
        elif agreed is False:
            rec.disagreement_count += 1
        if stale:
            rec.staleness_flags += 1
        if schema_violation:
            rec.schema_violations += 1
        rec.trust_score = self._compute(rec)
        rec.last_updated = _iso(_utc_now())

    @staticmethod
    def _compute(rec: SourceReputation) -> float:
        """
        §24.2 formula, with Laplace smoothing so a brand-new source
        isn't instantly punished.

        trust = clamp(
            0.5
            + 0.4 * (agreement_rate - 0.5)
            - 0.3 * staleness_rate
            - 0.4 * violation_rate
            + 0.2 * manual_authority
            , MIN_TRUST, MAX_TRUST
        )
        """
        n = rec.observation_count
        if n == 0:
            base = INITIAL_TRUST
        else:
            # Laplace-smoothed rates
            agreed = rec.agreement_count
            disagree = rec.disagreement_count
            agree_rate = (agreed + LAPLACE_ALPHA) / (agreed + disagree + 2 * LAPLACE_ALPHA)

            stale_rate = rec.staleness_flags / n
            violation_rate = rec.schema_violations / n

            base = (
                INITIAL_TRUST
                + 0.4 * (agree_rate - 0.5)
                - 0.3 * stale_rate
                - 0.4 * violation_rate
            )

        if rec.manual_authority:
            base += 0.2

        return round(max(MIN_TRUST, min(MAX_TRUST, base)), 3)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = ReputationStore()

    # New source starts at neutral
    assert store.trust("example.com") == INITIAL_TRUST

    # Agreeing repeatedly raises trust
    for _ in range(5):
        store.record_agreement("good.example")
    assert store.trust("good.example") > INITIAL_TRUST

    # Disagreeing lowers trust
    for _ in range(5):
        store.record_disagreement("bad.example")
    assert store.trust("bad.example") < INITIAL_TRUST

    # Staleness lowers trust
    for _ in range(5):
        store.record_stale("stale.example")
    assert store.trust("stale.example") < INITIAL_TRUST

    # Schema violations lower trust
    for _ in range(5):
        store.record_schema_violation("broken.example")
    assert store.trust("broken.example") < INITIAL_TRUST

    # Manual authority boosts trust
    store.set_manual_authority("trusted.example", True)
    assert store.trust("trusted.example") > INITIAL_TRUST

    # Clamped to range
    for _ in range(200):
        store.record_agreement("clamped.example")
    assert store.trust("clamped.example") <= MAX_TRUST

    for _ in range(200):
        store.record_disagreement("clampedlow.example")
    assert store.trust("clampedlow.example") >= MIN_TRUST

    # Decay
    store2 = ReputationStore()
    rec = store2.get("old.example")
    rec.trust_score = 0.95
    rec.observation_count = 10
    old = _utc_now() - timedelta(days=100)
    rec.last_updated = _iso(old)
    store2.apply_decay()
    assert store2.trust("old.example") == INITIAL_TRUST

    # Serialization round-trip
    rows = store.to_list()
    store3 = ReputationStore.from_list(rows)
    assert store3.trust("good.example") == store.trust("good.example")

    print("Reputation store OK.")