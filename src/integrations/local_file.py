"""
LocalFileConnector — spec §47A.

Writes a dataset to the local filesystem in a chosen format.

    LocalFileConnector(
        base_dir="/path/to/output",
        filename_template="leads_{date}.{ext}",
    )

Filename resolution order:
    1. `ref.filename` if provided by the caller
    2. `filename_template` (with {date} / {time} / {ext} substituted)
    3. auto-generated: "{delivery_id}.{ext}"

The extension is derived from `ref.format`:
    json    -> .json
    jsonl   -> .jsonl
    csv     -> .csv
    xlsx    -> .xlsx
    parquet -> .parquet

Safety:
    - Writes to a temp file first, then atomically renames (Windows-safe).
    - Refuses to overwrite an existing file unless `overwrite=True`.
    - Never writes outside `base_dir` — path traversal is blocked.

Capabilities:
    DELIVERS_FILES
    SUPPORTS_IDEMPOTENCY   (a second publish with the same metadata
                            idempotency_key becomes a no-op)
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.integrations.base import BaseConnector
from src.integrations.types import (
    ConnectorCapability,
    ConnectorError,
    ConnectorTestResult,
    ConnectorType,
    DatasetReference,
    DeliveryResult,
)


logger = logging.getLogger("integrations.local_file")


_FORMAT_TO_EXT = {
    "json": "json",
    "jsonl": "jsonl",
    "csv": "csv",
    "xlsx": "xlsx",
    "parquet": "parquet",
}


class LocalFileConnector(BaseConnector):
    connector_type = ConnectorType.LOCAL_FILE

    def __init__(
        self,
        base_dir: str | Path,
        filename_template: str = "",
        overwrite: bool = False,
        connector_id: Optional[str] = None,
    ):
        if not base_dir:
            raise ConnectorError("base_dir is required")
        self.base_dir = Path(base_dir).resolve()
        self.filename_template = filename_template or ""
        self.overwrite = bool(overwrite)

        # Track already-delivered idempotency keys within this process.
        self._delivered_keys: dict[str, str] = {}   # key -> file path

        super().__init__(
            connector_id=connector_id,
            capabilities={
                ConnectorCapability.DELIVERS_FILES,
                ConnectorCapability.SUPPORTS_IDEMPOTENCY,
            },
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    async def _do_test(self) -> ConnectorTestResult:
        # Ensure the directory exists, or can be created
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ConnectorTestResult(
                ok=False,
                message=f"cannot create base_dir {self.base_dir}: {e}",
            )

        if not self.base_dir.is_dir():
            return ConnectorTestResult(
                ok=False,
                message=f"{self.base_dir} is not a directory",
            )

        # Write a probe file to confirm writability
        probe = self.base_dir / ".aces_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as e:
            return ConnectorTestResult(
                ok=False,
                message=f"base_dir is not writable: {e}",
            )

        free_mb = shutil.disk_usage(self.base_dir).free // (1024 * 1024)
        return ConnectorTestResult(
            ok=True,
            message=(
                f"{self.base_dir} is writable; {free_mb} MB free"
            ),
        )

    async def _do_publish(
        self,
        ref: DatasetReference,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        metadata = metadata or {}

        # ---- Idempotency ----
        idem_key = metadata.get("idempotency_key") or ""
        if idem_key and idem_key in self._delivered_keys:
            existing = self._delivered_keys[idem_key]
            logger.info(
                f"local_file: idempotency hit for {idem_key!r} "
                f"-> {existing}"
            )
            return DeliveryResult(
                ok=True,
                destination=existing,
                metadata={"idempotent_replay": True},
            )

        # ---- Format ----
        fmt = (ref.format or "json").lower()
        if fmt not in _FORMAT_TO_EXT:
            raise ConnectorError(
                f"unsupported format {fmt!r}; "
                f"expected one of {sorted(_FORMAT_TO_EXT)}"
            )
        ext = _FORMAT_TO_EXT[fmt]

        # ---- Filename ----
        target = self._resolve_target(ref, ext)
        self._assert_within_base_dir(target)

        if target.exists() and not self.overwrite:
            # Attach a timestamp to avoid clobbering
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            target = target.with_name(f"{target.stem}.{ts}{target.suffix}")

        target.parent.mkdir(parents=True, exist_ok=True)

        # ---- Records ----
        records = list(ref.records or [])
        if not records and ref.url:
            raise ConnectorError(
                "local_file connector requires inline records; "
                "signed-URL fetching is not supported"
            )
        if not records:
            return DeliveryResult(
                ok=False,
                destination=str(target),
                error="no records to write",
            )

        # ---- Atomic write ----
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            from src.output.formats import write_records
            result = write_records(records, tmp, format=fmt)

            # write_records may return a warning only (e.g. parquet missing)
            if result.warnings:
                raise ConnectorError(
                    f"writer warnings: {'; '.join(result.warnings)}"
                )

            os.replace(tmp, target)
        except ConnectorError:
            _safe_unlink(tmp)
            raise
        except Exception as e:
            _safe_unlink(tmp)
            raise ConnectorError(
                f"write failed: {type(e).__name__}: {e}"
            ) from e

        size = target.stat().st_size

        if idem_key:
            self._delivered_keys[idem_key] = str(target)

        return DeliveryResult(
            ok=True,
            destination=str(target),
            bytes_sent=size,
            records_sent=len(records),
            metadata={
                "format": fmt,
                "filename": target.name,
                "base_dir": str(self.base_dir),
            },
        )

    async def _do_send(
        self,
        payload: dict,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        raise ConnectorError(
            "local_file connector does not support send(); "
            "use publish() to write datasets"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_target(self, ref: DatasetReference, ext: str) -> Path:
        if ref.filename:
            return self.base_dir / ref.filename

        if self.filename_template:
            now = datetime.now(timezone.utc)
            filled = (
                self.filename_template
                .replace("{date}", now.strftime("%Y-%m-%d"))
                .replace("{time}", now.strftime("%H%M%S"))
                .replace("{ext}", ext)
            )
            return self.base_dir / filled

        # Fallback: unique name based on current timestamp
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        return self.base_dir / f"delivery_{ts}.{ext}"

    def _assert_within_base_dir(self, target: Path) -> None:
        """
        Refuse any path that escapes base_dir. Blocks '../../etc/passwd'
        in a caller-supplied filename.
        """
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as e:
            raise ConnectorError(f"cannot resolve target path: {e}") from e

        try:
            resolved.relative_to(self.base_dir)
        except ValueError:
            raise ConnectorError(
                f"target {resolved} escapes base_dir {self.base_dir}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import json
    import tempfile

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir:
            c = LocalFileConnector(base_dir=tmpdir)

            # ---- test_connection ----
            tr = await c.test_connection()
            assert tr.ok is True, tr.message
            assert "writable" in tr.message

            # ---- publish JSON ----
            ref = DatasetReference(
                format="json",
                records=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
                filename="out.json",
            )
            dr = await c.publish(ref)
            assert dr.ok is True, dr.error
            assert dr.records_sent == 2
            assert dr.bytes_sent > 0
            assert dr.destination.endswith("out.json")

            # Verify content
            data = json.loads(Path(dr.destination).read_text(encoding="utf-8"))
            assert len(data) == 2
            assert data[0]["a"] == 1

            # ---- publish CSV ----
            ref2 = DatasetReference(
                format="csv",
                records=[{"name": "Alice", "age": 30}],
                filename="people.csv",
            )
            dr2 = await c.publish(ref2)
            assert dr2.ok is True
            text = Path(dr2.destination).read_text(encoding="utf-8")
            assert "name,age" in text
            assert "Alice,30" in text

            # ---- publish JSONL ----
            ref3 = DatasetReference(
                format="jsonl",
                records=[{"x": 1}, {"x": 2}],
                filename="lines.jsonl",
            )
            dr3 = await c.publish(ref3)
            lines = Path(dr3.destination).read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2

            # ---- Overwrite protection ----
            # Publishing out.json again without overwrite should create
            # a new file with a timestamp suffix.
            ref4 = DatasetReference(
                format="json",
                records=[{"a": 99}],
                filename="out.json",
            )
            dr4 = await c.publish(ref4)
            assert dr4.ok is True
            assert dr4.destination != dr.destination
            assert "out." in Path(dr4.destination).name

            # ---- Overwrite=True ----
            c2 = LocalFileConnector(base_dir=tmpdir, overwrite=True)
            ref5 = DatasetReference(
                format="json",
                records=[{"a": 100}],
                filename="out.json",
            )
            dr5 = await c2.publish(ref5)
            assert dr5.destination == dr.destination   # same path

            # ---- Filename template ----
            c3 = LocalFileConnector(
                base_dir=tmpdir,
                filename_template="report_{date}.{ext}",
            )
            ref6 = DatasetReference(
                format="csv",
                records=[{"z": 1}],
            )
            dr6 = await c3.publish(ref6)
            assert dr6.ok is True
            name = Path(dr6.destination).name
            assert name.startswith("report_") and name.endswith(".csv")

            # ---- Idempotency ----
            c4 = LocalFileConnector(base_dir=tmpdir)
            ref7 = DatasetReference(
                format="json",
                records=[{"once": True}],
                filename="idem.json",
            )
            r1 = await c4.publish(ref7, metadata={"idempotency_key": "K-1"})
            r2 = await c4.publish(ref7, metadata={"idempotency_key": "K-1"})
            assert r1.ok and r2.ok
            assert r2.metadata.get("idempotent_replay") is True
            assert r1.destination == r2.destination

            # ---- Path traversal is blocked ----
            evil = DatasetReference(
                format="json",
                records=[{"x": 1}],
                filename="../../etc/passwd",
            )
            dr_evil = await c.publish(evil)
            assert dr_evil.ok is False
            assert "escapes base_dir" in dr_evil.error

            # ---- Empty records -> not ok ----
            empty = DatasetReference(format="json", records=[])
            dr_empty = await c.publish(empty)
            assert dr_empty.ok is False
            assert "no records" in dr_empty.error

            # ---- Unsupported format ----
            bad_fmt = DatasetReference(
                format="weird", records=[{"x": 1}], filename="x.weird",
            )
            dr_bad = await c.publish(bad_fmt)
            assert dr_bad.ok is False
            assert "unsupported format" in dr_bad.error

            # ---- send() is not supported ----
            dr_send = await c.send({"text": "hi"})
            assert dr_send.ok is False
            assert "does not support send" in dr_send.error

            print("LocalFileConnector OK.")

    asyncio.run(run())