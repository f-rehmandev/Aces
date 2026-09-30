"""
Pagination / scroll progress detection — spec §14.3 ("no-progress detection").

Both pagination and infinite scroll need the same safety valve: if the
next "page" is effectively the same as the last one, stop. This module
holds that logic, isolated from the browser so it's cheap to unit-test.
"""

from __future__ import annotations
import re
from typing import Callable, Iterable


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_html(html: str) -> str:
    """Collapse whitespace so trivially-different-but-equivalent HTML matches."""
    return _WHITESPACE_RE.sub(" ", (html or "").strip())


def html_unchanged(a: str, b: str) -> bool:
    """
    True if two HTML strings are equivalent after whitespace normalization.
    Used by pagination and infinite-scroll loops to detect a stuck target.
    """
    return _normalize_html(a) == _normalize_html(b)


def records_unchanged(
    previous: Iterable[dict],
    current: Iterable[dict],
    key_fn: Callable[[dict], str],
) -> bool:
    """
    True if the two record sets have identical identity keys (order-insensitive).

    `key_fn` extracts an identity key from a record. Records whose key_fn
    returns falsy are ignored (they can't be compared meaningfully).
    """
    prev_keys = {key_fn(r) for r in previous if key_fn(r)}
    curr_keys = {key_fn(r) for r in current if key_fn(r)}
    return prev_keys == curr_keys


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # HTML comparison
    assert html_unchanged("<p>a</p>", "  <p>a</p>  ")
    assert html_unchanged("<p>a\n  b</p>", "<p>a b</p>")
    assert not html_unchanged("<p>a</p>", "<p>b</p>")
    assert not html_unchanged("", "<p>x</p>")

    # Records comparison
    def key(r): return r.get("id")

    assert records_unchanged(
        [{"id": "1"}, {"id": "2"}],
        [{"id": "2"}, {"id": "1"}],   # same set, different order
        key,
    )
    assert not records_unchanged(
        [{"id": "1"}, {"id": "2"}],
        [{"id": "1"}, {"id": "3"}],
        key,
    )
    assert not records_unchanged([{"id": "1"}], [], key)

    # Records with no keys -> "unchanged" (both empty sets)
    assert records_unchanged([{"x": 1}], [{"y": 2}], key)

    print("Progress detector OK.")