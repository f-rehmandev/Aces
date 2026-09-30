"""
Tenant-scoped service registry.

Keeps tasks, job specs, and versioned datasets per client.

A persistent TaskStore can be supplied for durable task/version storage.
The in-memory cache remains the fast path for API/runtime access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from src.core.task_spec import TaskSpec
from src.history.dataset import VersionedDataset

if TYPE_CHECKING:
    from src.storage.task_store import TaskStore


@dataclass
class ClientBucket:
    """Everything owned by one client."""
    tasks: dict[str, TaskSpec] = field(default_factory=dict)

    # task_id -> VersionedDataset
    datasets: dict[str, VersionedDataset] = field(default_factory=dict)

    # job_id -> job record (state, task_id, result, ...)
    jobs: dict[str, dict] = field(default_factory=dict)


class ServiceRegistry:
    """
    Tenant-scoped registry.

    Every method requires client_id.
    Cross-tenant reads return None/empty rather than raising.
    """

    def __init__(self, task_store: "TaskStore | None" = None):
        self._buckets: dict[str, ClientBucket] = {}
        self._task_store = task_store

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------

    def _bucket(self, client_id: str) -> ClientBucket:
        if client_id not in self._buckets:
            self._buckets[client_id] = ClientBucket()
        return self._buckets[client_id]

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    def save_task(self, client_id: str, spec: TaskSpec) -> None:
        if self._task_store is not None:
            self._task_store.save_task(client_id, spec)

        self._bucket(client_id).tasks[spec.task_id] = spec

    def get_task(
        self,
        client_id: str,
        task_id: str,
        version: Optional[int] = None,
    ) -> Optional[TaskSpec]:
        if version is not None:
            if self._task_store is not None:
                loaded = self._task_store.get_task(
                    client_id,
                    task_id,
                    version=version,
                )
                return loaded

            # In-memory registry only has the current TaskSpec.
            # Version 1 is the only historical version available.
            if version != 1:
                return None

            return self._bucket(client_id).tasks.get(task_id)

        cached = self._bucket(client_id).tasks.get(task_id)
        if cached is not None:
            return cached

        if self._task_store is not None:
            loaded = self._task_store.get_task(
                client_id,
                task_id,
            )
            if loaded is not None:
                self._bucket(client_id).tasks[task_id] = loaded
            return loaded

        return None

    def list_tasks(self, client_id: str) -> list[TaskSpec]:
        bucket = self._bucket(client_id)

        if self._task_store is not None:
            persisted = self._task_store.list_tasks(client_id)

            for spec in persisted:
                bucket.tasks[spec.task_id] = spec

        return list(bucket.tasks.values())

    def delete_task(self, client_id: str, task_id: str) -> bool:
        bucket = self._bucket(client_id)

        deleted_persisted = False
        if self._task_store is not None:
            deleted_persisted = self._task_store.delete_task(
                client_id,
                task_id,
            )

        deleted_cached = task_id in bucket.tasks
        if deleted_cached:
            del bucket.tasks[task_id]

        return deleted_persisted or deleted_cached

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def get_dataset(
        self,
        client_id: str,
        task_id: str,
    ) -> Optional[VersionedDataset]:
        return self._bucket(client_id).datasets.get(task_id)

    def ensure_dataset(
        self,
        client_id: str,
        task_id: str,
    ) -> VersionedDataset:
        bucket = self._bucket(client_id)

        if task_id not in bucket.datasets:
            bucket.datasets[task_id] = VersionedDataset(task_key=task_id)

        return bucket.datasets[task_id]

    def append_dataset_version(
        self,
        client_id: str,
        task_id: str,
        records: list[dict],
        quality_score: float = 0.0,
        quality_passed: bool = True,
        confidence_mean: float = 0.0,
        note: str = "",
    ):
        ds = self.ensure_dataset(client_id, task_id)

        return ds.append(
            records=records,
            quality_score=quality_score,
            quality_passed=quality_passed,
            confidence_mean=confidence_mean,
            note=note,
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def save_job(
        self,
        client_id: str,
        job_id: str,
        job: dict,
    ) -> None:
        self._bucket(client_id).jobs[job_id] = job

    def get_job(
        self,
        client_id: str,
        job_id: str,
    ) -> Optional[dict]:
        return self._bucket(client_id).jobs.get(job_id)

    def list_jobs(self, client_id: str) -> list[dict]:
        return list(self._bucket(client_id).jobs.values())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def known_clients(self) -> list[str]:
        clients = set(self._buckets.keys())

        if self._task_store is not None:
            clients.update(self._task_store.list_client_ids())

        return sorted(clients)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.core.task_spec import Target

    reg = ServiceRegistry()

    spec = TaskSpec(
        natural_language_prompt="find laptop prices",
        target=Target(
            start_urls=["https://example.com/a"],
        ),
    )

    reg.save_task("acme", spec)

    assert reg.get_task("acme", spec.task_id) is spec
    assert reg.get_task("other", spec.task_id) is None

    assert len(reg.list_tasks("acme")) == 1
    assert len(reg.list_tasks("other")) == 0

    v1 = reg.append_dataset_version(
        "acme",
        spec.task_id,
        records=[{"title": "A", "price": "$10"}],
        quality_score=1.0,
        quality_passed=True,
        confidence_mean=0.9,
    )

    assert v1.version == 1

    v2 = reg.append_dataset_version(
        "acme",
        spec.task_id,
        records=[{"title": "A", "price": "$12"}],
        quality_score=0.8,
        quality_passed=True,
    )

    assert v2.version == 2

    ds = reg.get_dataset("acme", spec.task_id)
    assert ds is not None
    assert len(ds) == 2
    assert ds.latest().records == [
        {"title": "A", "price": "$12"}
    ]

    assert reg.delete_task("acme", spec.task_id) is True
    assert reg.delete_task("acme", spec.task_id) is False

    print("Service registry OK.")