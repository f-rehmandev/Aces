"""
Confidence scoring — spec §25.

Confidence is not truth. It's an explicit, explainable score representing
the evidence ACES has for a given value.

Inputs (§25.1):
    - source consensus strength        (0..1)
    - number of independent sources    (int)
    - mean source trust score          (0..1)  [reputation, §24 — Round 3]
    - freshness of observation         (0..1)  [1 = fresh, 0 = stale]
    - selector confidence              (0..1)  [from schema discovery §12]
    - field validation passed          (bool)
    - historical stability             (0..1)  [share of prior runs with same value]
    - anomaly score                    (0..1, higher = more anomalous, negative contributor)

Design:
    - Weighted sum, normalised to [0, 1].
    - Weights are configurable; defaults match §25.1's "high/medium/low" bands.
    - `explain()` produces the human-readable sentence required by §25.3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Weights — §25.1 "high / medium / low" bands made concrete.
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "consensus":          0.30,   # high
    "source_count":       0.15,   # high  (normalised: 1 source = 0.2, 3+ = 1.0)
    "mean_trust":         0.15,   # medium
    "freshness":          0.10,   # medium
    "selector":           0.10,   # medium
    "validation":         0.10,   # medium (boolean → 1.0 or 0.0)
    "historical_stability": 0.05, # low
    "anomaly_penalty":    0.05,   # negative contributor (subtracted)
}


@dataclass
class ConfidenceInputs:
    consensus: Optional[float] = None          # 0..1
    source_count: int = 0                      # number of independent sources
    mean_trust: Optional[float] = None         # 0..1
    freshness: Optional[float] = None          # 0..1
    selector: Optional[float] = None           # 0..1
    validation_passed: Optional[bool] = None
    historical_stability: Optional[float] = None  # 0..1
    anomaly_score: Optional[float] = None      # 0..1, higher = worse


@dataclass
class ConfidenceScore:
    value: float                                # 0..1
    inputs: ConfidenceInputs
    contributors: dict[str, float] = field(default_factory=dict)  # per-signal contribution
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "explanation": self.explanation,
            "contributors": dict(self.contributors),
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_source_count(n: int) -> float:
    """
    Map source count to a 0..1 signal.
        0 sources  -> 0.0
        1 source   -> 0.2
        2 sources  -> 0.6
        3+ sources -> 1.0
    """
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.2
    if n == 2:
        return 0.6
    return 1.0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    def __init__(self, weights: Optional[dict] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def score(self, inputs: ConfidenceInputs) -> ConfidenceScore:
        # Collect available signals; missing signals are dropped and the
        # remaining weights are renormalised so absence doesn't imply low.
        available: dict[str, float] = {}

        if inputs.consensus is not None:
            available["consensus"] = _clamp01(inputs.consensus)

        if inputs.source_count > 0:
            available["source_count"] = _normalize_source_count(inputs.source_count)

        if inputs.mean_trust is not None:
            available["mean_trust"] = _clamp01(inputs.mean_trust)

        if inputs.freshness is not None:
            available["freshness"] = _clamp01(inputs.freshness)

        if inputs.selector is not None:
            available["selector"] = _clamp01(inputs.selector)

        if inputs.validation_passed is not None:
            available["validation"] = 1.0 if inputs.validation_passed else 0.0

        if inputs.historical_stability is not None:
            available["historical_stability"] = _clamp01(inputs.historical_stability)

        # Anomaly is a penalty, only applies if provided.
        anomaly_penalty = 0.0
        if inputs.anomaly_score is not None:
            anomaly_penalty = _clamp01(inputs.anomaly_score) * self.weights["anomaly_penalty"]

        if not available:
            return ConfidenceScore(
                value=0.0,
                inputs=inputs,
                contributors={},
                explanation="No signals available to score confidence.",
            )

        # Renormalise weights over the signals we actually have.
        positive_keys = [k for k in available if k in self.weights]
        weight_sum = sum(self.weights[k] for k in positive_keys)
        if weight_sum == 0:
            weight_sum = 1.0

        raw = sum(
            self.weights[k] * available[k]
            for k in positive_keys
        ) / weight_sum

        value = _clamp01(raw - anomaly_penalty)

        contributors = {k: round(available[k], 3) for k in positive_keys}
        if anomaly_penalty:
            contributors["anomaly_penalty"] = -round(anomaly_penalty, 3)

        explanation = self._explain(value, inputs, contributors)

        return ConfidenceScore(
            value=round(value, 3),
            inputs=inputs,
            contributors=contributors,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Explanation (§25.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _explain(value: float, inputs: ConfidenceInputs, contributors: dict) -> str:
        if value >= 0.85:
            tier = "High"
        elif value >= 0.6:
            tier = "Moderate"
        elif value >= 0.35:
            tier = "Low"
        else:
            tier = "Very low"

        reasons: list[str] = []

        if inputs.consensus is not None and inputs.consensus >= 0.8:
            reasons.append(
                f"{int(inputs.source_count) or 'multiple'} independent source(s) agree"
            )
        elif inputs.consensus is not None and inputs.consensus < 0.5:
            reasons.append("sources disagree")

        if inputs.source_count and inputs.source_count == 1:
            reasons.append("only one source supplied this value")

        if inputs.mean_trust is not None:
            if inputs.mean_trust >= 0.75:
                reasons.append(f"high mean source trust ({inputs.mean_trust:.2f})")
            elif inputs.mean_trust < 0.4:
                reasons.append(f"low mean source trust ({inputs.mean_trust:.2f})")

        if inputs.freshness is not None:
            if inputs.freshness >= 0.9:
                reasons.append("value is fresh")
            elif inputs.freshness < 0.4:
                reasons.append("value is stale")

        if inputs.validation_passed is False:
            reasons.append("value failed validation")

        if inputs.historical_stability is not None and inputs.historical_stability >= 0.9:
            reasons.append("stable across recent runs")

        if inputs.anomaly_score is not None and inputs.anomaly_score > 0.5:
            reasons.append(f"anomaly signal ({inputs.anomaly_score:.2f})")

        if not reasons:
            reasons.append("mixed signals")

        return f"{tier} confidence ({value:.2f}) because " + ", and ".join(reasons) + "."


# ---------------------------------------------------------------------------
# Record-level roll-up (§25.2)
# ---------------------------------------------------------------------------

def record_confidence(
    field_scores: dict[str, ConfidenceScore],
    mode: str = "min",
) -> float:
    """
    Roll up per-field confidence to a record-level score.

    mode:
        "min"     — weakest-link (default per §25.2; conservative)
        "mean"    — weighted mean
    """
    if not field_scores:
        return 0.0
    values = [s.value for s in field_scores.values()]
    if mode == "mean":
        return round(sum(values) / len(values), 3)
    return round(min(values), 3)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_scorer = ConfidenceScorer()

def score(inputs: ConfidenceInputs) -> ConfidenceScore:
    return _scorer.score(inputs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Strong evidence
    s = _scorer.score(ConfidenceInputs(
        consensus=0.95,
        source_count=3,
        mean_trust=0.9,
        freshness=0.95,
        selector=0.95,
        validation_passed=True,
        historical_stability=0.98,
    ))
    assert s.value >= 0.85, s.value
    assert "High confidence" in s.explanation
    print("high:", s.value, "|", s.explanation)

    # Weak evidence
    s = _scorer.score(ConfidenceInputs(
        consensus=0.3,
        source_count=1,
        mean_trust=0.4,
        freshness=0.5,
    ))
    assert s.value < 0.5
    assert "only one source" in s.explanation or "Low" in s.explanation or "Very low" in s.explanation
    print("low:", s.value, "|", s.explanation)

    # Minimal inputs
    s = _scorer.score(ConfidenceInputs(validation_passed=True))
    assert 0.0 < s.value <= 1.0

    # Anomaly penalty
    no_anom = _scorer.score(ConfidenceInputs(consensus=0.9, source_count=3))
    with_anom = _scorer.score(ConfidenceInputs(consensus=0.9, source_count=3, anomaly_score=0.9))
    assert with_anom.value < no_anom.value

    # No signals at all
    s = _scorer.score(ConfidenceInputs())
    assert s.value == 0.0
    assert "No signals" in s.explanation

    # Record roll-up: min
    fields = {
        "title": ConfidenceScore(value=0.95, inputs=ConfidenceInputs()),
        "price": ConfidenceScore(value=0.6, inputs=ConfidenceInputs()),
    }
    assert record_confidence(fields, mode="min") == 0.6
    assert record_confidence(fields, mode="mean") == 0.775

    print("Confidence scorer OK.")