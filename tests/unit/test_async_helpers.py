"""Unit tests for src.util.async_helpers.maybe_await."""
import asyncio

import pytest

from src.util.async_helpers import maybe_await


def _run(coro):
    return asyncio.run(coro)


def test_plain_value_returned_unchanged():
    assert _run(maybe_await(42)) == 42
    assert _run(maybe_await("hi")) == "hi"
    assert _run(maybe_await(None)) is None
    assert _run(maybe_await(0)) == 0        # falsy but valid
    assert _run(maybe_await(False)) is False
    assert _run(maybe_await([])) == []


def test_awaitable_is_awaited():
    async def _inner():
        await asyncio.sleep(0)
        return "from-coroutine"

    assert _run(maybe_await(_inner())) == "from-coroutine"


def test_future_is_awaited():
    async def _driver():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        loop.call_soon(fut.set_result, "from-future")
        return await maybe_await(fut)

    assert _run(_driver()) == "from-future"


def test_custom_awaitable_is_awaited():
    class Awaitable:
        def __await__(self):
            async def _impl():
                return "from-custom"
            return _impl().__await__()

    assert _run(maybe_await(Awaitable())) == "from-custom"


def test_exception_propagates():
    async def _inner():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run(maybe_await(_inner()))