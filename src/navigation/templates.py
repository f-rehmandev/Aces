"""
URL template expansion — spec §14.1 ("URL templates with ranges").

Supports:
    {1..50}         -> 1, 2, 3, ..., 50
    {1..10:2}       -> 1, 3, 5, 7, 9         (step)
    {a,b,c}         -> a, b, c               (choices)

Multiple templates in one expression expand as a Cartesian product.

    expand("https://x.com/p?page={1..3}&cat={shoes,bags}")
    -> 6 URLs (3 pages × 2 categories)
"""

from __future__ import annotations
import re
from itertools import product


# --- Range:  {start..end}  or  {start..end:step} -------------------------
_RANGE_RE = re.compile(r"\{(-?\d+)\.\.(-?\d+)(?::(\d+))?\}")

# --- Choices:  {a,b,c}  (no ".." inside) -------------------------------
_CHOICE_RE = re.compile(r"\{([^{}.]+(?:,[^{}.]+)+)\}")


def _expand_ranges(template: str) -> list[str]:
    """
    Expand any single range placeholder, or Cartesian product of all.
    Returns templates with ranges substituted (one per variant).
    """
    matches = list(_RANGE_RE.finditer(template))
    if not matches:
        return [template]

    # Build the list of "value lists" for each range in order.
    value_lists: list[list[str]] = []
    for m in matches:
        start = int(m.group(1))
        end = int(m.group(2))
        step = int(m.group(3)) if m.group(3) else 1
        if step < 1:
            raise ValueError(f"Range step must be >= 1, got {step}")
        if end < start:
            raise ValueError(f"Range end ({end}) must be >= start ({start})")
        value_lists.append([str(i) for i in range(start, end + 1, step)])

    # Cartesian product across ranges, applied with re.sub by index.
    out: list[str] = []
    for combo in product(*value_lists):
        result = template
        for m, value in zip(matches, combo):
            result = result.replace(m.group(0), value, 1)
        out.append(result)
    return out


def _expand_choices(template: str) -> list[str]:
    """Expand any single choices placeholder, or Cartesian product of all."""
    matches = list(_CHOICE_RE.finditer(template))
    if not matches:
        return [template]

    value_lists: list[list[str]] = []
    for m in matches:
        value_lists.append([v.strip() for v in m.group(1).split(",")])

    out: list[str] = []
    for combo in product(*value_lists):
        result = template
        for m, value in zip(matches, combo):
            result = result.replace(m.group(0), value, 1)
        out.append(result)
    return out


def expand(template: str, max_urls: int = 10_000) -> list[str]:
    """
    Fully expand a URL template.

    Raises ValueError if any part is malformed, or if expansion would
    produce more than `max_urls` (safety valve for typos like {1..999999}).
    """
    if not template:
        return []

    # Range expansion first (so {1..3} inside {a,b} braces doesn't confuse choice regex)
    after_ranges = _expand_ranges(template)

    out: list[str] = []
    for partial in after_ranges:
        out.extend(_expand_choices(partial))
        if len(out) > max_urls:
            raise ValueError(
                f"Template expands to more than {max_urls} URLs; "
                f"narrow the range or raise max_urls."
            )
    return out


def has_template(expr: str) -> bool:
    """True if the expression contains a template placeholder."""
    return bool(_RANGE_RE.search(expr) or _CHOICE_RE.search(expr))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple range
    assert expand("https://x.com/p?page={1..3}") == [
        "https://x.com/p?page=1",
        "https://x.com/p?page=2",
        "https://x.com/p?page=3",
    ]

    # Range with step
    assert expand("https://x.com/p?page={1..10:2}") == [
        "https://x.com/p?page=1",
        "https://x.com/p?page=3",
        "https://x.com/p?page=5",
        "https://x.com/p?page=7",
        "https://x.com/p?page=9",
    ]

    # Choices
    assert expand("https://x.com/{a,b,c}") == [
        "https://x.com/a",
        "https://x.com/b",
        "https://x.com/c",
    ]

    # Cartesian — order not guaranteed by the spec; compare as sets.
    got = sorted(expand("https://x.com/{a,b}?p={1..2}"))
    expected = sorted([
        "https://x.com/a?p=1",
        "https://x.com/a?p=2",
        "https://x.com/b?p=1",
        "https://x.com/b?p=2",
    ])
    assert got == expected, got

    # No template
    assert expand("https://x.com/plain") == ["https://x.com/plain"]

    # Safety valve
    try:
        expand("https://x.com/p={1..100000}", max_urls=1000)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # has_template
    assert has_template("https://x.com/{1..3}")
    assert has_template("https://x.com/{a,b}")
    assert not has_template("https://x.com/plain")

    print("URL templates OK.")
