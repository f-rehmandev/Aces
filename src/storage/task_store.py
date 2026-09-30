"""
Persistent TaskSpec storage — ACES task/version repository.

The database stores:
    tasks          -> current task metadata + current version
    task_versions  -> immutable TaskSpec snapshots

This repository uses the backend Supabase service-role client because
the application server is responsible for tenant scoping before storage.
Database RLS remains the authoritative second layer.
"""

from __future__ import annotations

from typing import Optional

from src.core.task_spec import TaskSpec


class TaskStoreError(RuntimeError):
    """Raised when persistent task storage fails."""


class TaskStore:
    """Supabase-backed task/version repository."""

    @staticmethod
    def _client():
        """
        Resolve the Supabase client lazily, at call time.

        The import is deliberately inside this method rather than at
        module top: `from src.storage.db import get_client` binds the
        name into this module's namespace at import time, which makes
        the `tests.integration._fakes.FakeSupabaseClient` pattern
        ineffective — patching `src.storage.db.get_client` would not
        affect this module's binding.

        `BatchStore` (src/storage/batch_store.py) already uses this
        lazy style inside its `build_supabase_store()` factory. This
        helper makes `TaskStore` consistent with it so every
        storage-layer module can be fake-patched the same way.
        """
        from src.storage.db import get_client
        return get_client()  

    def save_task(self, client_id: str, spec: TaskSpec) -> int:
        if not client_id:
            raise ValueError("client_id is required")

        payload = spec.to_dict()
        client = self._client()

        try:
            existing = (
                client.table("tasks")
                .select("task_id,current_version")
                .eq("task_id", spec.task_id)
                .eq("client_id", client_id)
                .limit(1)
                .execute()
            ).data

            if not existing:
                (
                    client.table("tasks")
                    .insert(
                        {
                            "task_id": spec.task_id,
                            "client_id": client_id,
                            "current_version": 1,
                            "status": "active",
                        }
                    )
                    .execute()
                )

                (
                    client.table("task_versions")
                    .insert(
                        {
                            "task_id": spec.task_id,
                            "version": 1,
                            "client_id": client_id,
                            "spec": payload,
                        }
                    )
                    .execute()
                )

                return 1

            current_version = int(existing[0]["current_version"])

            current_rows = (
                client.table("task_versions")
                .select("spec")
                .eq("task_id", spec.task_id)
                .eq("version", current_version)
                .eq("client_id", client_id)
                .limit(1)
                .execute()
            ).data

            if current_rows:
                current_spec = current_rows[0].get("spec") or {}
                if current_spec == payload:
                    return current_version

            next_version = current_version + 1

            (
                client.table("task_versions")
                .insert(
                    {
                        "task_id": spec.task_id,
                        "version": next_version,
                        "client_id": client_id,
                        "spec": payload,
                    }
                )
                .execute()
            )

            (
                client.table("tasks")
                .update(
                    {
                        "current_version": next_version,
                    }
                )
                .eq("task_id", spec.task_id)
                .eq("client_id", client_id)
                .execute()
            )

            return next_version

        except Exception as exc:
            raise TaskStoreError(
                f"failed to save task {spec.task_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def get_task(
        self,
        client_id: str,
        task_id: str,
        version: Optional[int] = None,
    ) -> Optional[TaskSpec]:
        if not client_id:
            raise ValueError("client_id is required")

        client = self._client()

        try:
            task_rows = (
                client.table("tasks")
                .select("task_id,current_version")
                .eq("task_id", task_id)
                .eq("client_id", client_id)
                .limit(1)
                .execute()
            ).data

            if not task_rows:
                return None

            selected_version = (
                int(version)
                if version is not None
                else int(task_rows[0]["current_version"])
            )

            rows = (
                client.table("task_versions")
                .select("spec")
                .eq("task_id", task_id)
                .eq("version", selected_version)
                .eq("client_id", client_id)
                .limit(1)
                .execute()
            ).data

            if not rows:
                return None

            spec = TaskSpec.from_dict(rows[0]["spec"])
            spec.client_id = client_id
            return spec

        except Exception as exc:
            raise TaskStoreError(
                f"failed to load task {task_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def list_tasks(self, client_id: str) -> list[TaskSpec]:
        if not client_id:
            raise ValueError("client_id is required")

        client = self._client()

        try:
            rows = (
                client.table("tasks")
                .select("task_id,current_version")
                .eq("client_id", client_id)
                .order("updated_at", desc=True)
                .execute()
            ).data

            result: list[TaskSpec] = []

            for row in rows:
                spec = self.get_task(
                    client_id=client_id,
                    task_id=row["task_id"],
                    version=int(row["current_version"]),
                )
                if spec is not None:
                    result.append(spec)

            return result

        except TaskStoreError:
            raise
        except Exception as exc:
            raise TaskStoreError(
                f"failed to list tasks for client {client_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def list_client_ids(self) -> list[str]:
        client = self._client()

        try:
            rows = (
                client.table("tasks")
                .select("client_id")
                .execute()
            ).data

            return sorted(
                {
                    str(row["client_id"])
                    for row in rows
                    if row.get("client_id")
                }
            )

        except Exception as exc:
            raise TaskStoreError(
                f"failed to list task clients: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def delete_task(self, client_id: str, task_id: str) -> bool:
        if not client_id:
            raise ValueError("client_id is required")

        client = self._client()

        try:
            existing = (
                client.table("tasks")
                .select("task_id")
                .eq("task_id", task_id)
                .eq("client_id", client_id)
                .limit(1)
                .execute()
            ).data

            if not existing:
                return False

            (
                client.table("tasks")
                .delete()
                .eq("task_id", task_id)
                .eq("client_id", client_id)
                .execute()
            )

            return True

        except Exception as exc:
            raise TaskStoreError(
                f"failed to delete task {task_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc