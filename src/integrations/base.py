"""
BaseConnector — common plumbing for concrete connectors.

Concrete connectors (S3, Sheets, Slack, webhook, ...) inherit from
this and implement three methods:

    async def _do_test(self) -> ConnectorTestResult
    async def _do_publish(self, ref) -> DeliveryResult
    async def _do_send(self, payload) -> DeliveryResult

Everything else — latency measurement, exception-to-result
conversion, capability checks — is handled here. Subclasses never
need to worry about catching exceptions or formatting results.

Rule: `test_connection()`, `publish()`, and `send()` MUST NOT raise.
They always return a ConnectorTestResult or DeliveryResult with
ok=False on failure.
"""
from __future__ import annotations

import abc
import logging
import time
import uuid
from typing import Optional

from src.integrations.types import (
    ConnectorCapability,
    ConnectorError,
    ConnectorTestResult,
    ConnectorType,
    DatasetReference,
    DeliveryResult,
)


logger = logging.getLogger("integrations.base")


class BaseConnector(abc.ABC):
    """
    Abstract base for every connector. Subclasses must implement
    `_do_test`, `_do_publish`, and `_do_send`.

    Not every connector uses all three:
        - A local-file connector may only care about `_do_publish`.
        - A Slack connector may only care about `_do_send`.
        - An S3 connector may care about both.

    The base class dispatches; subclasses that don't support an
    operation should raise `ConnectorError` from the corresponding
    `_do_*` method — this is caught and converted into a
    DeliveryResult(ok=False, error="...") with the standard shape.
    """

    # Subclasses MUST override these
    connector_type: ConnectorType = ConnectorType.LOCAL_FILE

    def __init__(
        self,
        connector_id: Optional[str] = None,
        capabilities: Optional[set] = None,
    ):
        self.connector_id = connector_id or str(uuid.uuid4())
        self.capabilities = set(capabilities or set())

    # ------------------------------------------------------------------
    # Public API (do not override)
    # ------------------------------------------------------------------
    async def test_connection(self) -> ConnectorTestResult:
        started = time.monotonic()
        try:
            result = await self._do_test()
        except ConnectorError as e:
            return ConnectorTestResult(
                ok=False,
                message=f"configuration error: {e}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as e:
            return ConnectorTestResult(
                ok=False,
                message=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        # Fill in latency if the subclass didn't
        if result.latency_ms == 0:
            result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

    async def publish(
        self,
        ref: DatasetReference,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        started = time.monotonic()
        if ConnectorCapability.DELIVERS_FILES not in self.capabilities \
                and ConnectorCapability.DELIVERS_ROWS not in self.capabilities:
            return DeliveryResult(
                ok=False,
                connector_type=self.connector_type.value,
                connector_id=self.connector_id,
                error=(
                    f"connector {self.connector_type.value} does not "
                    f"support publish() (no delivery capability)"
                ),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        try:
            result = await self._do_publish(ref, metadata=metadata)
        except ConnectorError as e:
            return DeliveryResult(
                ok=False,
                connector_type=self.connector_type.value,
                connector_id=self.connector_id,
                error=f"configuration error: {e}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as e:
            logger.warning(
                f"connector {self.connector_id} publish failed: "
                f"{type(e).__name__}: {e}"
            )
            return DeliveryResult(
                ok=False,
                connector_type=self.connector_type.value,
                connector_id=self.connector_id,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        # Fill in standard fields if the subclass didn't
        if not result.connector_type:
            result.connector_type = self.connector_type.value
        if not result.connector_id:
            result.connector_id = self.connector_id
        if result.latency_ms == 0:
            result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

    async def send(
        self,
        payload: dict,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        started = time.monotonic()
        if ConnectorCapability.DELIVERS_NOTIFICATIONS not in self.capabilities:
            return DeliveryResult(
                ok=False,
                connector_type=self.connector_type.value,
                connector_id=self.connector_id,
                error=(
                    f"connector {self.connector_type.value} does not "
                    f"support send() (no notification capability)"
                ),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        try:
            result = await self._do_send(payload, metadata=metadata)
        except ConnectorError as e:
            return DeliveryResult(
                ok=False,
                connector_type=self.connector_type.value,
                connector_id=self.connector_id,
                error=f"configuration error: {e}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as e:
            logger.warning(
                f"connector {self.connector_id} send failed: "
                f"{type(e).__name__}: {e}"
            )
            return DeliveryResult(
                ok=False,
                connector_type=self.connector_type.value,
                connector_id=self.connector_id,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        if not result.connector_type:
            result.connector_type = self.connector_type.value
        if not result.connector_id:
            result.connector_id = self.connector_id
        if result.latency_ms == 0:
            result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------
    def supports(self, cap: ConnectorCapability) -> bool:
        return cap in self.capabilities

    def has_all(self, *caps: ConnectorCapability) -> bool:
        return all(c in self.capabilities for c in caps)

    def has_any(self, *caps: ConnectorCapability) -> bool:
        return any(c in self.capabilities for c in caps)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def _do_test(self) -> ConnectorTestResult:
        """Subclass ping. Return ConnectorTestResult. May raise."""
        ...

    @abc.abstractmethod
    async def _do_publish(
        self, ref: DatasetReference, *, metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        """Subclass publish. Return DeliveryResult. May raise."""
        ...

    @abc.abstractmethod
    async def _do_send(
        self, payload: dict, *, metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        """Subclass send. Return DeliveryResult. May raise."""
        ...


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class NullConnector(BaseConnector):
        """Minimal subclass — implements all three hooks."""
        connector_type = ConnectorType.LOCAL_FILE

        async def _do_test(self):
            return ConnectorTestResult(ok=True, message="null ok")

        async def _do_publish(self, ref, *, metadata=None):
            return DeliveryResult(
                ok=True, destination="null",
                records_sent=len(ref.records),
            )

        async def _do_send(self, payload, *, metadata=None):
            return DeliveryResult(ok=True, destination="null")

    class BrokenConnector(BaseConnector):
        connector_type = ConnectorType.WEBHOOK

        async def _do_test(self):
            raise RuntimeError("can't reach")

        async def _do_publish(self, ref, *, metadata=None):
            raise ValueError("bad payload")

        async def _do_send(self, payload, *, metadata=None):
            raise ConnectionError("network down")

    async def run():
        # Capability gates
        c_no_caps = NullConnector()
        assert not c_no_caps.supports(ConnectorCapability.DELIVERS_FILES)

        # With capability → publish works
        c = NullConnector(capabilities={
            ConnectorCapability.DELIVERS_FILES,
            ConnectorCapability.DELIVERS_NOTIFICATIONS,
        })
        assert c.supports(ConnectorCapability.DELIVERS_FILES)
        assert c.has_all(ConnectorCapability.DELIVERS_FILES,
                         ConnectorCapability.DELIVERS_NOTIFICATIONS)
        assert c.has_any(ConnectorCapability.SUPPORTS_SIGNED_URLS,
                         ConnectorCapability.DELIVERS_FILES)

        # test_connection ok
        tr = await c.test_connection()
        assert tr.ok is True
        assert tr.message == "null ok"
        assert tr.latency_ms >= 0

        # publish ok — records_sent from subclass
        ref = DatasetReference(records=[{"a": 1}, {"b": 2}])
        dr = await c.publish(ref)
        assert dr.ok is True
        assert dr.records_sent == 2
        assert dr.connector_type == "local_file"
        assert dr.connector_id == c.connector_id
        assert dr.destination == "null"

        # send ok
        dr = await c.send({"text": "hello"})
        assert dr.ok is True

        # --- Without capability: publish rejects cleanly ---
        c2 = NullConnector()   # no capabilities
        dr = await c2.publish(ref)
        assert dr.ok is False
        assert "does not support publish" in dr.error

        dr = await c2.send({"text": "x"})
        assert dr.ok is False
        assert "does not support send" in dr.error

        # --- Exceptions become DeliveryResult, never raise ---
        b = BrokenConnector(capabilities={
            ConnectorCapability.DELIVERS_FILES,
            ConnectorCapability.DELIVERS_NOTIFICATIONS,
        })
        tr = await b.test_connection()
        assert tr.ok is False
        assert "RuntimeError" in tr.message

        dr = await b.publish(ref)
        assert dr.ok is False
        assert "ValueError" in dr.error

        dr = await b.send({"x": 1})
        assert dr.ok is False
        assert "ConnectionError" in dr.error

        # --- ConnectorError surfaces as "configuration error" ---
        class Misconfigured(BaseConnector):
            connector_type = ConnectorType.S3
            async def _do_test(self):
                raise ConnectorError("missing bucket")
            async def _do_publish(self, ref, *, metadata=None):
                raise ConnectorError("missing bucket")
            async def _do_send(self, payload, *, metadata=None):
                raise ConnectorError("missing bucket")

        mc = Misconfigured(capabilities={
            ConnectorCapability.DELIVERS_FILES,
            ConnectorCapability.DELIVERS_NOTIFICATIONS,
        })
        assert "configuration error" in (await mc.test_connection()).message
        assert "configuration error" in (await mc.publish(ref)).error
        assert "configuration error" in (await mc.send({})).error

        print("BaseConnector OK.")

    asyncio.run(run())