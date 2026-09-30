"""
Sitemap + robots.txt parsing — spec §14.1.

Parses XML/plain-text content directly. No network here — fetching lives in
the scraper engine. Both parsers are pure functions of the input text.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Sitemap
# ---------------------------------------------------------------------------

@dataclass
class SitemapEntry:
    url: str
    lastmod: Optional[str] = None
    changefreq: Optional[str] = None
    priority: Optional[float] = None


@dataclass
class SitemapParseResult:
    entries: list[SitemapEntry]
    child_sitemaps: list[str]
    is_index: bool


def _localname(tag: str) -> str:
    """Strip an XML namespace prefix: '{ns}foo' -> 'foo'."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def parse_sitemap(xml_text: str) -> SitemapParseResult:
    """
    Parse a sitemap.xml or sitemap-index.xml.

    <urlset>      -> entries
    <sitemapindex> -> child_sitemaps

    Empty input returns an empty result. Malformed XML raises ValueError.
    """
    if not xml_text or not xml_text.strip():
        return SitemapParseResult(entries=[], child_sitemaps=[], is_index=False)

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Malformed sitemap XML: {e}") from e

    root_name = _localname(root.tag)

    if root_name == "sitemapindex":
        children: list[str] = []
        for sm in root:
            if _localname(sm.tag) != "sitemap":
                continue
            for child in sm:
                if _localname(child.tag) == "loc" and child.text:
                    children.append(child.text.strip())
                    break
        return SitemapParseResult(entries=[], child_sitemaps=children, is_index=True)

    if root_name == "urlset":
        entries: list[SitemapEntry] = []
        for url_el in root:
            if _localname(url_el.tag) != "url":
                continue
            entry = SitemapEntry(url="")
            for child in url_el:
                name = _localname(child.tag)
                text = (child.text or "").strip() or None
                if name == "loc" and text:
                    entry.url = text
                elif name == "lastmod":
                    entry.lastmod = text
                elif name == "changefreq":
                    entry.changefreq = text
                elif name == "priority" and text:
                    try:
                        entry.priority = float(text)
                    except ValueError:
                        pass
            if entry.url:
                entries.append(entry)
        return SitemapParseResult(entries=entries, child_sitemaps=[], is_index=False)

    # Unknown root — treat as empty rather than crash.
    return SitemapParseResult(entries=[], child_sitemaps=[], is_index=False)


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

@dataclass
class RobotsRules:
    disallow: list[str]
    allow: list[str]
    sitemaps: list[str]


def parse_robots(text: str, user_agent: str = "*") -> RobotsRules:
    """
    Parse robots.txt content for a single user-agent.

    Section selection:
      - If a group explicitly lists our user_agent (case-insensitive), use
        only that group.
      - Otherwise, use the group whose User-agent is '*'.
      - Otherwise, no rules apply (everything allowed).
    """
    if not text:
        return RobotsRules(disallow=[], allow=[], sitemaps=[])

    target = user_agent.lower()
    groups: list[tuple[list[str], list[str], list[str]]] = []
    current_agents: list[str] = []
    current_disallow: list[str] = []
    current_allow: list[str] = []
    sitemaps: list[str] = []

    def flush() -> None:
        if current_agents:
            groups.append(
                (list(current_agents), list(current_disallow), list(current_allow))
            )

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "user-agent":
            # A new User-agent after any directives starts a new group.
            if current_disallow or current_allow:
                flush()
                current_agents.clear()
                current_disallow.clear()
                current_allow.clear()
            current_agents.append(value.lower())
        elif key == "disallow":
            current_disallow.append(value)
        elif key == "allow":
            current_allow.append(value)
        elif key == "sitemap":
            sitemaps.append(value)

    flush()

    selected: Optional[tuple[list[str], list[str], list[str]]] = None
    for agents, dis, al in groups:
        if target in agents:
            selected = (agents, dis, al)
            break
    if selected is None:
        for agents, dis, al in groups:
            if "*" in agents:
                selected = (agents, dis, al)
                break

    if selected is None:
        return RobotsRules(disallow=[], allow=[], sitemaps=sitemaps)

    _, disallow, allow = selected
    return RobotsRules(disallow=disallow, allow=allow, sitemaps=sitemaps)


def robots_allows(path: str, rules: RobotsRules) -> bool:
    """
    Standard robots.txt path match:
      - No rules at all -> allowed.
      - Longest matching pattern wins; on tie, allow wins.
      - '*' inside a pattern matches any characters.
    """
    import re

    if not rules.disallow and not rules.allow:
        return True

    def to_regex(pattern: str) -> str:
        # robots.txt semantics (matching Google's published rules):
        #   '*'            -> matches any characters
        #   '$' at the end -> end-of-URL anchor
        #   everything else -> literal
        anchored_end = pattern.endswith("$")
        body = pattern[:-1] if anchored_end else pattern
        escaped = re.escape(body).replace(r"\*", ".*")
        return "^" + escaped + ("$" if anchored_end else "")

    best_len = -1
    best_type: Optional[str] = None  # "allow" | "disallow"

    for pattern in rules.disallow:
        if not pattern:
            continue
        if re.match(to_regex(pattern), path) and len(pattern) > best_len:
            best_len = len(pattern)
            best_type = "disallow"

    for pattern in rules.allow:
        if not pattern:
            continue
        if re.match(to_regex(pattern), path) and len(pattern) >= best_len:
            best_len = len(pattern)
            best_type = "allow"

    return best_type != "disallow"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    urlset = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://x.com/a</loc>
        <lastmod>2026-09-01</lastmod>
        <changefreq>daily</changefreq>
        <priority>0.9</priority>
      </url>
      <url>
        <loc>https://x.com/b</loc>
        <lastmod>2026-09-15</lastmod>
      </url>
    </urlset>
    """
    r = parse_sitemap(urlset)
    assert not r.is_index and len(r.entries) == 2
    assert r.entries[0].url == "https://x.com/a"
    assert r.entries[0].priority == 0.9

    sitemapindex = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://x.com/sitemap-1.xml</loc></sitemap>
      <sitemap><loc>https://x.com/sitemap-2.xml</loc></sitemap>
    </sitemapindex>
    """
    r = parse_sitemap(sitemapindex)
    assert r.is_index
    assert r.child_sitemaps == ["https://x.com/sitemap-1.xml", "https://x.com/sitemap-2.xml"]

    robots = """User-agent: *
    Disallow: /admin
    Allow: /admin/public
    Disallow: /private
    Sitemap: https://x.com/sitemap.xml
    """
    rr = parse_robots(robots, user_agent="*")
    assert rr.disallow == ["/admin", "/private"]
    assert rr.allow == ["/admin/public"]
    assert rr.sitemaps == ["https://x.com/sitemap.xml"]

    assert not robots_allows("/admin/x", rr)
    assert robots_allows("/admin/public/y", rr)
    assert not robots_allows("/private/z", rr)
    assert robots_allows("/products/a", rr)

    print("Sitemap + robots OK.")
