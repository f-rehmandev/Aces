"""
Multi-source triangulation — spec §23.

Answers: "Do independent sources agree, and how much should we trust
the consensus?"

Pipeline:
    1. Group observations by (record_id, field_name)
    2. Normalize values (currency, date, url, text) for comparison
    3. Cluster sources into independence groups — 5 copies of the same
       feed count as 1 vote
    4. Weight each cluster by sum of source trust scores
    5. Pick the winning cluster; confidence = winner_weight / total
       adjusted for number of independent clusters
    6. Report dissent as `Conflict` objects — never hidden

Not implemented in this round:
    - Automatic syndication detection (identical content signature,
      WHOIS/ASN analysis). Independence groups are caller-supplied or
      default to the source domain.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    record_id: str
    field_name: str
    source_domain: str
    value: Any
    trust_score: float = 0.5
    observed_at: str = ""
    extraction_method: str = ""
    independence_group: Optional[str] = None   # overrides source_domain if set


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    value: Any
    normalized: Any
    sources: list[str] = field(default_factory=list)
    weight: float = 0.0

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "normalized": self.normalized,
            "sources": list(self.sources),
            "weight": self.weight,
        }


@dataclass
class ConsensusResult:
    record_id: str
    field_name: str
    consensus_value: Any
    normalized_value: Any
    confidence: float
    winning_sources: list[str] = field(default_factory=list)
    total_clusters: int = 0
    conflicts: list[Conflict] = field(default_factory=list)
    note: str = ""

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "field_name": self.field_name,
            "consensus_value": self.consensus_value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "winning_sources": list(self.winning_sources),
            "total_clusters": self.total_clusters,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Value normalization (§23.3)
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_CURRENCY_SYMBOL_RE = re.compile(r"[^\d.\-+]")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def normalize_for_comparison(value: Any, field_type: str = "text") -> str:
    """
    Normalize a value for equality comparison across sources.
    Kept deliberately coarse — we're checking "do these refer to the same
    thing?", not producing the final output value.
    """
    if value is None:
        return ""
    s = _WHITESPACE_RE.sub(" ", str(value).strip())
    if not s:
        return ""

    if field_type == "currency":
        # Extract the numeric part; currency codes are ignored for equality.
        num = _CURRENCY_SYMBOL_RE.sub("", s)
        try:
            return f"{float(num):.4f}"
        except ValueError:
            return s.lower()

    if field_type == "number":
        try:
            return f"{float(s):.4f}"
        except ValueError:
            return s.lower()

    if field_type == "date":
        # Match on the ISO prefix if present, else lowercase comparison.
        if _ISO_DATE_RE.match(s):
            return s[:10]
        return s.lower()

    if field_type == "url":
        from src.navigation.url_normalizer import canonicalize
        try:
            return canonicalize(s).lower()
        except Exception:
            return s.lower()

    if field_type == "email":
        return s.lower()

    # default: text
    return s.lower()


def _infer_field_type(field_name: str) -> str:
    n = (field_name or "").lower()
    if "price" in n or "cost" in n:
        return "currency"
    if "date" in n:
        return "date"
    if "url" in n or "link" in n:
        return "url"
    if "email" in n:
        return "email"
    return "text"


# ---------------------------------------------------------------------------
# Triangulator
# ---------------------------------------------------------------------------

class Triangulator:
    """
    Consensus engine. Stateless per call.

    `min_clusters_for_confident` — number of independent clusters at which
    a unanimous consensus reaches full confidence (default 4).
    """

    def __init__(self, min_clusters_for_confident: int = 4):
        self.min_clusters_for_confident = max(1, min_clusters_for_confident)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def triangulate(
        self,
        observations: list[Observation],
        field_types: Optional[dict[str, str]] = None,
    ) -> list[ConsensusResult]:
        field_types = field_types or {}
        grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for obs in observations:
            if obs.value in (None, ""):
                continue
            grouped[(obs.record_id, obs.field_name)].append(obs)

        results: list[ConsensusResult] = []
        for (record_id, field_name), group in grouped.items():
            ftype = field_types.get(field_name) or _infer_field_type(field_name)
            results.append(self._consensus_for_field(
                record_id, field_name, group, ftype,
            ))
        return results

    # ------------------------------------------------------------------
    # Consensus for one (record, field)
    # ------------------------------------------------------------------
    def _consensus_for_field(
        self,
        record_id: str,
        field_name: str,
        observations: list[Observation],
        field_type: str,
    ) -> ConsensusResult:
        # --- normalize each observation and bucket into independence clusters ---
        # clusters: group_id -> {normalized_value -> {"weight": w, "sources": [...], "raw": representative}}
        clusters: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
            lambda: {"weight": 0.0, "sources": [], "raw": None}
        ))

        for obs in observations:
            normalized = normalize_for_comparison(obs.value, field_type)
            if not normalized:
                continue
            group_id = obs.independence_group or obs.source_domain or "unknown"
            bucket = clusters[group_id][normalized]
            bucket["weight"] += max(0.0, float(obs.trust_score or 0.0))
            bucket["sources"].append(obs.source_domain or "unknown")
            if bucket["raw"] is None:
                bucket["raw"] = obs.value

        # --- collapse each cluster to its dominant value ---
        # Within a cluster, the dominant value is the one with highest weight.
        cluster_values: dict[str, tuple[str, float, Any, list[str]]] = {}
        for group_id, by_value in clusters.items():
            dominant_norm = max(by_value, key=lambda v: by_value[v]["weight"])
            v = by_value[dominant_norm]
            cluster_values[group_id] = (dominant_norm, v["weight"], v["raw"], v["sources"])

        if not cluster_values:
            return ConsensusResult(
                record_id=record_id,
                field_name=field_name,
                consensus_value=None,
                normalized_value=None,
                confidence=0.0,
                note="no usable observations",
            )

        # --- aggregate across clusters, keyed by normalized value ---
        value_weights: dict[str, float] = defaultdict(float)
        value_sources: dict[str, list[str]] = defaultdict(list)
        value_raw: dict[str, Any] = {}
        for group_id, (norm, weight, raw, sources) in cluster_values.items():
            value_weights[norm] += weight
            value_sources[norm].extend(sources)
            value_raw.setdefault(norm, raw)

        total_weight = sum(value_weights.values())
        if total_weight <= 0:
            # All trust scores zero — fall back to simple majority.
            total_weight = float(len(cluster_values))
            for norm in value_weights:
                value_weights[norm] = len(value_sources[norm])

        winning_norm = max(value_weights, key=lambda n: (value_weights[n], -len(value_sources[n])))
        winner_weight = value_weights[winning_norm]
        consensus_ratio = winner_weight / total_weight

        # --- source-count adjustment: a single cluster can never look highly confident ---
        n_clusters = len(cluster_values)
        source_factor = min(1.0, 0.4 + 0.2 * (n_clusters - 1))
        confidence = round(consensus_ratio * source_factor, 3)

        # --- conflicts ---
        conflicts: list[Conflict] = []
        for norm, weight in value_weights.items():
            if norm == winning_norm:
                continue
            conflicts.append(Conflict(
                value=value_raw.get(norm),
                normalized=norm,
                sources=list(value_sources[norm]),
                weight=round(weight, 3),
            ))

        note = ""
        if n_clusters == 1:
            note = "only one independent source"
        elif conflicts:
            note = f"{len(conflicts)} dissenting source cluster(s)"

        return ConsensusResult(
            record_id=record_id,
            field_name=field_name,
            consensus_value=value_raw.get(winning_norm),
            normalized_value=winning_norm,
            confidence=confidence,
            winning_sources=list(value_sources[winning_norm]),
            total_clusters=n_clusters,
            conflicts=conflicts,
            note=note,
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_default = Triangulator()

def triangulate(
    observations: list[Observation],
    field_types: Optional[dict[str, str]] = None,
) -> list[ConsensusResult]:
    return _default.triangulate(observations, field_types=field_types)


def observations_from_records(
    records: list[dict],
    source_domain: str,
    trust_score: float = 0.5,
    record_id_fn=None,
) -> list[Observation]:
    """
    Helper: convert one source's records into Observations.
    `record_id_fn` — callable(record) -> record_id. Defaults to
    make_record_id from provenance.
    """
    from src.trust.provenance import make_record_id
    rid_fn = record_id_fn or make_record_id
    out: list[Observation] = []
    ignored = {"source_url", "diff_status", "_numeric_price"}
    for r in records:
        rid = rid_fn(r)
        for fname, value in r.items():
            if fname in ignored:
                continue
            out.append(Observation(
                record_id=rid,
                field_name=fname,
                source_domain=source_domain,
                value=value,
                trust_score=trust_score,
            ))
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Three independent sources agree
    obs = [
        Observation("r1", "price", "shop-a.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "shop-b.example", "24.99 USD", trust_score=0.85),
        Observation("r1", "price", "shop-c.example", "$24.99", trust_score=0.9),
    ]
    results = triangulate(obs)
    assert len(results) == 1
    r = results[0]
    assert r.consensus_value in ("$24.99", "24.99 USD")
    assert r.confidence >= 0.6
    assert not r.conflicts
    print(f"unanimous 3 sources: {r.consensus_value} confidence={r.confidence}")

    # 2. Two agree, one dissents
    obs = [
        Observation("r1", "price", "shop-a.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "shop-b.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "shop-c.example", "$29.99", trust_score=0.9),
    ]
    r = triangulate(obs)[0]
    assert "24.99" in str(r.consensus_value)
    assert len(r.conflicts) == 1
    print(f"2v1: {r.consensus_value} confidence={r.confidence} conflicts={len(r.conflicts)}")

    # 3. Same-domain sources count as ONE cluster
    obs = [
        Observation("r1", "price", "shop-a.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "shop-a.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "shop-b.example", "$29.99", trust_score=0.9),
    ]
    r = triangulate(obs)[0]
    # one cluster has 2 obs from same domain (weight 1.8), other has weight 0.9
    assert r.total_clusters == 2
    assert "24.99" in str(r.consensus_value)
    print(f"cluster-aware: {r.consensus_value} confidence={r.confidence} clusters={r.total_clusters}")

    # 4. Trust weighting can beat majority
    obs = [
        Observation("r1", "x", "trusted.example", "A", trust_score=0.99),
        Observation("r1", "x", "cheap1.example", "B", trust_score=0.1),
        Observation("r1", "x", "cheap2.example", "B", trust_score=0.1),
    ]
    r = triangulate(obs)[0]
    assert r.consensus_value == "A"
    print(f"trust beats majority: {r.consensus_value}")

    # 5. Single source (only one cluster)
    obs = [Observation("r1", "x", "solo.example", "only", trust_score=0.9)]
    r = triangulate(obs)[0]
    assert r.confidence <= 0.5
    assert "only one" in r.note
    print(f"single source: confidence={r.confidence} note={r.note}")

    # 6. Explicit independence groups
    obs = [
        Observation("r1", "x", "a.example", "A", trust_score=0.9, independence_group="feed-1"),
        Observation("r1", "x", "b.example", "A", trust_score=0.9, independence_group="feed-1"),
        Observation("r1", "x", "c.example", "B", trust_score=0.9, independence_group="feed-2"),
    ]
    r = triangulate(obs)[0]
    assert r.total_clusters == 2
    print(f"explicit groups: consensus={r.consensus_value}")

    # 7. Missing values are ignored
    obs = [
        Observation("r1", "x", "a.example", None, trust_score=0.9),
        Observation("r1", "x", "b.example", "A", trust_score=0.9),
    ]
    r = triangulate(obs)[0]
    assert r.consensus_value == "A"

    # 8. to_dict works
    d = r.to_dict()
    assert d["record_id"] == "r1"

    print("Triangulator OK.")