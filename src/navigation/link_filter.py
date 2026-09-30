"""
Link filtering — spec §14.1.

Decides whether a discovered link should be followed. Combines:
    - scheme allow-list (http/https only)
    - same-domain-only flag
    - include/exclude regex lists
    - optional max-depth gate

Precedence: exclude beats include. If include_patterns is non-empty, a URL
must match at least one include pattern.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


@dataclass
class LinkFilterPolicy:
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    same_domain_only: bool = True
    allowed_schemes: tuple[str, ...] = ("http", "https")
    max_depth: Optional[int] = None


class LinkFilter:
    """Stateless filter built from a policy. Reusable across a whole crawl."""

    def __init__(self, policy: LinkFilterPolicy):
        self.policy = policy
        self._includes = [re.compile(p) for p in policy.include_patterns]
        self._excludes = [re.compile(p) for p in policy.exclude_patterns]

    def allows(
        self,
        url: str,
        from_url: Optional[str] = None,
        depth: int = 0,
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason). `reason` is a short machine tag so the
        caller can log why a link was skipped.
        """
        try:
            parts = urlparse(url)
        except ValueError:
            return False, "unparseable"

        if parts.scheme not in self.policy.allowed_schemes:
            return False, "scheme_not_allowed"

        # Excludes first: they take priority over everything else.
        for rx in self._excludes:
            if rx.search(url):
                return False, "excluded_by_pattern"

        # Same-domain gate
        if self.policy.same_domain_only and from_url:
            from_host = (urlparse(from_url).netloc or "").lower()
            to_host = (parts.netloc or "").lower()
            if from_host and to_host and from_host != to_host:
                return False, "different_domain"

        # Include list: if non-empty, must match at least one pattern
        if self._includes:
            if not any(rx.search(url) for rx in self._includes):
                return False, "not_in_include_list"

        # Depth gate
        if self.policy.max_depth is not None and depth > self.policy.max_depth:
            return False, "max_depth_exceeded"

        return True, "allowed"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    policy = LinkFilterPolicy(
        include_patterns=[r"/products/"],
        exclude_patterns=[r"\.pdf$", r"/login"],
        same_domain_only=True,
        max_depth=2,
    )
    lf = LinkFilter(policy)
    origin = "https://shop.example.com/category/shoes"

    cases = [
        ("https://shop.example.com/products/a", 0, True),
        ("https://shop.example.com/about", 0, False),              # no include match
        ("https://shop.example.com/products/b.pdf", 0, False),     # excluded
        ("https://shop.example.com/login", 0, False),              # excluded
        ("https://other.com/products/a", 0, False),                # different domain
        ("https://shop.example.com/products/a", 3, False),         # too deep
        ("ftp://shop.example.com/products/a", 0, False),           # scheme
    ]
    for url, depth, expected in cases:
        ok, reason = lf.allows(url, from_url=origin, depth=depth)
        print(f"{url!r:55s} depth={depth}  -> {ok}  ({reason})")
        assert ok is expected, f"expected {expected} for {url}"

    print("\nLinkFilter OK.")