"""
Connector registry — spec §47A.

Holds connector instances by id. Provides lookup by type, by
capability, and by client scope (connectors are tenant-scoped).

The registry is deliberately simple: it doesn't own configuration
loading or persistence. Callers construct concrete connectors (from
environment variables, from the database, or from a config file) and
register them here. The registry just answers "which connectors can
do X for client Y?".
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from src.integrations.base import BaseConnector
from src.integrations.types import (
    ConnectorCapability,
    ConnectorType,
)


logger = logging.getLogger("integrations.registry")


class ConnectorNotFoundError(LookupError):
    """No connector with that id in the registry."""


class ConnectorRegistry:
    """
    In-memory connector registry.

    Every entry is tenant-scoped via `client_id`. A connector with an
    empty client_id is treated as "global" — available to every tenant.
    That's useful for the operator's own notification webhook, or for
    shared infrastructure like a common S3 bucket.
    """

    def __init__(self):
        # connector_id -> (client_id, connector)
        self._connectors: dict[str, tuple[str, BaseConnector]] = {}

    # ------------------------------------------------------------------
    # Register / unregister
    # ------------------------------------------------------------------
    def register(
        self,
        connector: BaseConnector,
        *,
        client_id: str = "default",
    ) -> None:
        """Add a connector. Replaces any existing entry with the same id."""
        cid = connector.connector_id
        if not cid:
            raise ValueError("connector must have a connector_id")
        self._connectors[cid] = (client_id, connector)

    def unregister(self, connector_id: str) -> bool:
        return self._connectors.pop(connector_id, None) is not None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(
        self,
        connector_id: str,
        *,
        client_id: str = "default",
    ) -> Optional[BaseConnector]:
        """
        Look up a connector by id, enforcing tenant scope.

        A connector registered with `client_id=""` is global — visible
        to every tenant. Otherwise the caller must match the tenant.
        """
        entry = self._connectors.get(connector_id)
        if entry is None:
            return None
        owner_client, connector = entry
        if owner_client and client_id and owner_client != client_id:
            return None
        return connector

    def require(
        self,
        connector_id: str,
        *,
        client_id: str = "default",
    ) -> BaseConnector:
        c = self.get(connector_id, client_id=client_id)
        if c is None:
            raise ConnectorNotFoundError(
                f"connector {connector_id!r} not found for client {client_id!r}"
            )
        return c

    def all(self, *, client_id: Optional[str] = None) -> list[BaseConnector]:
        """
        Every connector visible to a client (its own + globals), or
        every connector at all if `client_id` is None (operator view).
        """
        out: list[BaseConnector] = []
        for owner_client, connector in self._connectors.values():
            if client_id is None:
                out.append(connector)
                continue
            if not owner_client or owner_client == client_id:
                out.append(connector)
        return out

    def by_type(
        self,
        connector_type: ConnectorType,
        *,
        client_id: Optional[str] = None,
    ) -> list[BaseConnector]:
        return [
            c for c in self.all(client_id=client_id)
            if c.connector_type == connector_type
        ]

    def by_capability(
        self,
        capability: ConnectorCapability,
        *,
        client_id: Optional[str] = None,
    ) -> list[BaseConnector]:
        return [
            c for c in self.all(client_id=client_id)
            if c.supports(capability)
        ]

    def by_any_capability(
        self,
        capabilities: Iterable[ConnectorCapability],
        *,
        client_id: Optional[str] = None,
    ) -> list[BaseConnector]:
        wanted = set(capabilities)
        return [
            c for c in self.all(client_id=client_id)
            if c.capabilities & wanted
        ]

    def __len__(self) -> int:
        return len(self._connectors)

    def __contains__(self, connector_id: str) -> bool:
        return connector_id in self._connectors

    def clear(self) -> None:
        self._connectors.clear()

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        by_client: dict[str, int] = {}
        for owner_client, connector in self._connectors.values():
            t = connector.connector_type.value
            by_type[t] = by_type.get(t, 0) + 1
            c = owner_client or "(global)"
            by_client[c] = by_client.get(c, 0) + 1
        return {
            "total": len(self._connectors),
            "by_type": by_type,
            "by_client": by_client,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from src.integrations.types import (
        ConnectorTestResult, DatasetReference, DeliveryResult,
    )

    class Stub(BaseConnector):
        connector_type = ConnectorType.LOCAL_FILE

        async def _do_test(self):
            return ConnectorTestResult(ok=True)
        async def _do_publish(self, ref, *, metadata=None):
            return DeliveryResult(ok=True)
        async def _do_send(self, payload, *, metadata=None):
            return DeliveryResult(ok=True)

    r = ConnectorRegistry()

    # Empty
    assert len(r) == 0
    assert r.get("nothing") is None

    # Register + get with correct client
    c1 = Stub(capabilities={ConnectorCapability.DELIVERS_FILES})
    r.register(c1, client_id="acme")
    assert len(r) == 1
    assert r.get(c1.connector_id, client_id="acme") is c1

    # Wrong client sees nothing
    assert r.get(c1.connector_id, client_id="other") is None
    # require() raises for wrong client
    try:
        r.require(c1.connector_id, client_id="other")
        raise AssertionError("expected ConnectorNotFoundError")
    except ConnectorNotFoundError:
        pass

    # require() works for owner
    assert r.require(c1.connector_id, client_id="acme") is c1

    # Global connector visible to every client
    g = Stub(capabilities={ConnectorCapability.DELIVERS_NOTIFICATIONS})
    r.register(g, client_id="")
    assert r.get(g.connector_id, client_id="acme") is g
    assert r.get(g.connector_id, client_id="other") is g

    # ---- Listing ----
    # acme sees both its own + the global
    assert len(r.all(client_id="acme")) == 2
    # other sees only the global
    assert len(r.all(client_id="other")) == 1
    assert r.all(client_id="other") == [g]
    # operator view sees everything
    assert len(r.all()) == 2

    # by_type
    locals_files = r.by_type(ConnectorType.LOCAL_FILE, client_id="acme")
    assert len(locals_files) == 2   # both are LOCAL_FILE stubs

    # by_capability
    files = r.by_capability(ConnectorCapability.DELIVERS_FILES, client_id="acme")
    assert files == [c1]
    notifications = r.by_capability(
        ConnectorCapability.DELIVERS_NOTIFICATIONS, client_id="other",
    )
    assert notifications == [g]

    # by_any_capability
    both = r.by_any_capability(
        {ConnectorCapability.DELIVERS_FILES,
         ConnectorCapability.DELIVERS_NOTIFICATIONS},
        client_id="acme",
    )
    assert set(both) == {c1, g}

    # ---- Containment ----
    assert c1.connector_id in r
    assert "nope" not in r

    # ---- Unregister ----
    assert r.unregister(c1.connector_id) is True
    assert r.unregister(c1.connector_id) is False
    assert r.get(c1.connector_id, client_id="acme") is None

    # ---- Stats ----
    r.clear()
    r.register(Stub(), client_id="acme")
    r.register(Stub(), client_id="acme")
    r.register(Stub(), client_id="")
    s = r.stats()
    assert s["total"] == 3
    assert s["by_type"].get("local_file") == 3
    assert s["by_client"]["acme"] == 2
    assert s["by_client"]["(global)"] == 1

    print("ConnectorRegistry OK.")