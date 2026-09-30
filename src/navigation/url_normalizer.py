"""
URL normalization — spec §14.2.

Conservative canonicalization: strip only what's provably safe, keep what
might be meaningful. Every change is recorded so a diff can distinguish
"URL actually changed" from "URL was normalized".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import (
    urlparse, urlunparse, urljoin, parse_qsl, urlencode, unquote,
)

# ---------------------------------------------------------------------------
# Tracking-prefix list (§14.2)
# ---------------------------------------------------------------------------
TRACKING_PARAM_PREFIXES = ("utm_",)

TRACKING_PARAM_EXACT = {
    "fbclid", "gclid", "dclid", "msclkid", "yclid",
    "mc_cid", "mc_eid", "igshid", "ref_src", "ref_url",
    "_ga", "_gl",
}

# Query params we always KEEP even if they're short and could look tracking-ish
MEANINGFUL_PARAMS = {
    "id", "sku", "page", "p", "category", "cat", "product", "product_id",
    "item", "item_id", "q", "query", "search", "sort", "order",
}


@dataclass
class NormalizationResult:
    """Everything the caller needs to know about what normalization did."""
    original: str
    normalized: str
    changes: list[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.original != self.normalized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_tracking_param(name: str) -> bool:
    low = name.lower()
    if low in MEANINGFUL_PARAMS:
        return False
    if low in TRACKING_PARAM_EXACT:
        return True
    for prefix in TRACKING_PARAM_PREFIXES:
        if low.startswith(prefix):
            return True
    return False


def _resolve_dot_segments(path: str) -> str:
    """
    Resolve ./ and ../ segments in a URL path (per RFC 3986 §5.2.4).
    We do this ourselves rather than via urljoin because urljoin also
    changes other parts of the URL.
    """
    if not path:
        return path
    # Use a synthetic base to borrow urllib's resolution, then extract path.
    resolved = urljoin("http://x/", path)
    return urlparse(resolved).path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_url(
    url: str,
    base: Optional[str] = None,
    preserve_query: Optional[set[str]] = None,
) -> NormalizationResult:
    """
    Canonicalize `url` per §14.2.

    If `base` is given and `url` is relative, it is resolved against `base`
    first.

    Returns a NormalizationResult. If the URL is unparseable, we return it
    unchanged (with a note) rather than raising — the caller decides how to
    treat that.
    """
    changes: list[str] = []
    original = url

    # Guard: an empty URL is not a URL. Return it unchanged so callers can
    # decide how to handle the failure. (Removing the old short-circuit means
    # the normal reconstruction path would otherwise turn "" into "/".)
    if not url:
        return NormalizationResult(original=original, normalized=original,
                                   changes=["empty"])

    # --- resolve relative URLs against base ---
    
    if base and not urlparse(url).scheme:
        url = urljoin(base, url)
        if url != original:
            changes.append("resolved_against_base")

    try:
        parts = urlparse(url)
    except ValueError:
        return NormalizationResult(original=original, normalized=url,
                                   changes=["unparseable"])

        # --- scheme + host lowercased ---
    # NOTE: Python's urlparse already lowercases the scheme for us, so we
    # can't detect the case from `parts.scheme`. We look at the raw string.
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()

    raw_scheme = url.split("://", 1)[0] if "://" in url else ""
    if raw_scheme and raw_scheme != raw_scheme.lower():
        changes.append("lowercased_scheme")
    if netloc and netloc != parts.netloc:
        changes.append("lowercased_host")

    # --- strip default ports ---
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]; changes.append("removed_default_port")
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]; changes.append("removed_default_port")

    # --- path: preserve case (§14.2), resolve dot segments ---
    path = parts.path or "/"
    resolved_path = _resolve_dot_segments(path)
    if resolved_path != path:
        changes.append("resolved_dot_segments")
        path = resolved_path
    if path == "":
        path = "/"

    # --- query: drop tracking, keep the rest, sort alphabetically ---
    raw_pairs = parse_qsl(parts.query, keep_blank_values=True)
    kept: list[tuple[str, str]] = []
    dropped = False
    for name, value in raw_pairs:
        if _is_tracking_param(name):
            # Caller can force-keep specific ones
            if preserve_query and name in preserve_query:
                kept.append((name, value))
            else:
                dropped = True
                continue
        else:
            kept.append((name, value))

    kept.sort(key=lambda kv: (kv[0], kv[1]))

    new_query = urlencode(kept)
    if dropped:
        changes.append("removed_tracking_params")
    if new_query != parts.query:
        changes.append("normalized_query")

    # --- fragment: always stripped (§14.2) ---
    if parts.fragment:
        changes.append("stripped_fragment")

    # --- rebuild ---
    normalized = urlunparse(
        (scheme, netloc, path, parts.params, new_query, "")
    )

    # NOTE: We intentionally do NOT short-circuit on "no changes". Even when
    # no explicit change is logged, urlunparse can still produce a slightly
    # different (but equivalent) string — e.g. adding a trailing slash to a
    # bare host, which is required so `example.com` and `example.com/` dedupe
    # correctly in the crawl frontier.
    return NormalizationResult(original=original, normalized=normalized, changes=changes)


def canonicalize(url: str, base: Optional[str] = None) -> str:
    """Convenience: just return the normalized string."""
    return normalize_url(url, base=base).normalized


def urls_are_equivalent(a: str, b: str, base: Optional[str] = None) -> bool:
    """Two URLs refer to the same resource after normalization?"""
    return canonicalize(a, base=base) == canonicalize(b, base=base)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("HTTP://Example.COM/Path/", None),
        ("https://example.com:443/a/b/../c", None),
        ("https://example.com/p?id=5&utm_source=x&utm_medium=y", None),
        ("https://example.com/p?b=2&a=1#frag", None),
        ("https://example.com/p?utm_campaign=z&id=42", None),
        ("https://example.com/p?q=shoes&utm_source=fb&sort=asc", None),
        ("/relative/path?x=1#y", "https://example.com/base"),
        ("https://example.com/a/./b/../c/", None),
    ]
    for raw, base in cases:
        r = normalize_url(raw, base=base)
        print(f"{raw!r}\n  -> {r.normalized!r}\n  changes: {r.changes}\n")

    # Equality check
    assert urls_are_equivalent(
        "https://example.com/p?id=5&utm_source=x",
        "https://example.com/p?id=5",
    )
    assert not urls_are_equivalent(
        "https://example.com/p?id=5",
        "https://example.com/p?id=6",
    )
    assert not urls_are_equivalent(
        "https://example.com/a",
        "https://example.com/A",  # path case preserved
    )

    print("URL normalizer OK.")