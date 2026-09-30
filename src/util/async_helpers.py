"""
Cross-cutting async helpers.

`maybe_await` is the sync/async tolerance shim used by every code path
that must work with either a synchronous real implementation or an
async test double. It lives in one place so future callers can't
silently introduce a subtly-different version of the same logic.
"""
from __future__ import annotations

import inspect
from typing import TypeVar

T = TypeVar("T")


async def maybe_await(value):
    """
    Return `value`, awaiting it first iff it is awaitable.

    Bridges real (synchronous) storage backends and async test doubles
    or future async implementations. Callers pass the raw return value
    of a call; if it's a coroutine, we await it; otherwise we return it
    unchanged.

    Example:
        batch = await maybe_await(batch_store.get(client_id, batch_id))
    """
    if inspect.isawaitable(value):
        return await value
    return value