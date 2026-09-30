"""
Versioned datasets — spec §28.

Every run produces a new version. ACES never overwrites — it appends and
diffs against the previous trusted version.

`VersionedDataset` is the in-memory/JSON representation. Persistence to
Supabase can be layered on top via `to_dict` / `from_dict`.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DatasetVersion:
    version: int
    created_at: str
    records: list[dict]
    quality_score: float = 0.0
    quality_passed: bool = True
    confidence_mean: float = 0.0
    note: str = ""
    superseded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class VersionedDataset:
    """
    Append-only store of dataset versions for a single task key.

    `task_key` — typically the natural-language prompt or search query.
    Version numbers start at 1 and increase monotonically.
    """

    def __init__(self, task_key: str):
        self.task_key = task_key
        self._versions: list[DatasetVersion] = []

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def latest(self) -> Optional[DatasetVersion]:
        return self._versions[-1] if self._versions else None

    def latest_valid(self) -> Optional[DatasetVersion]:
        """The most recent version that passed quality, if any."""
        for v in reversed(self._versions):
            if v.quality_passed and not v.superseded:
                return v
        return None

    def get(self, version: int) -> Optional[DatasetVersion]:
        for v in self._versions:
            if v.version == version:
                return v
        return None

    def all_versions(self) -> list[DatasetVersion]:
        return list(self._versions)

    def __len__(self) -> int:
        return len(self._versions)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def append(
        self,
        records: list[dict],
        quality_score: float = 0.0,
        quality_passed: bool = True,
        confidence_mean: float = 0.0,
        note: str = "",
    ) -> DatasetVersion:
        next_version = (self._versions[-1].version + 1) if self._versions else 1
        v = DatasetVersion(
            version=next_version,
            created_at=_utc_now_iso(),
            records=list(records),
            quality_score=quality_score,
            quality_passed=quality_passed,
            confidence_mean=confidence_mean,
            note=note,
        )
        self._versions.append(v)
        return v

    def supersede(self, version: int) -> None:
        for v in self._versions:
            if v.version == version:
                v.superseded = True
                return
        raise KeyError(f"version {version} not found")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "task_key": self.task_key,
            "versions": [v.to_dict() for v in self._versions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VersionedDataset":
        store = cls(task_key=data.get("task_key", ""))
        for raw in data.get("versions", []):
            store._versions.append(DatasetVersion(
                version=raw["version"],
                created_at=raw["created_at"],
                records=list(raw.get("records", [])),
                quality_score=raw.get("quality_score", 0.0),
                quality_passed=raw.get("quality_passed", True),
                confidence_mean=raw.get("confidence_mean", 0.0),
                note=raw.get("note", ""),
                superseded=raw.get("superseded", False),
            ))
        return store


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = VersionedDataset("track laptop prices")
    assert len(ds) == 0
    assert ds.latest() is None

    v1 = ds.append([{"title": "A", "price": "$1"}], quality_passed=True)
    assert v1.version == 1
    assert ds.latest().version == 1

    v2 = ds.append([{"title": "B", "price": "$2"}], quality_passed=True)
    assert v2.version == 2

    # Superseded version
    ds.supersede(1)
    assert ds.get(1).superseded

    # latest_valid skips superseded
    assert ds.latest_valid().version == 2

    # Failed quality marked
    v3 = ds.append([{"title": "C"}], quality_passed=False, quality_score=0.3)
    assert ds.latest().version == 3
    assert ds.latest_valid().version == 2   # v3 failed

    # Serialization round-trip
    d = ds.to_dict()
    ds2 = VersionedDataset.from_dict(d)
    assert len(ds2) == 3
    assert ds2.latest().version == 3
    assert ds2.get(1).superseded

    print("VersionedDataset OK.")