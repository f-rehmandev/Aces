"""Unit tests for sitemap + robots.txt parsers (spec §14.1)."""
import pytest

from src.navigation.sitemap import parse_sitemap, parse_robots, robots_allows


# ---------------------------------------------------------------------
# sitemap
# ---------------------------------------------------------------------

URLSET = """<?xml version="1.0" encoding="UTF-8"?>
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

SITEMAPINDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://x.com/sitemap-1.xml</loc></sitemap>
  <sitemap><loc>https://x.com/sitemap-2.xml</loc></sitemap>
</sitemapindex>
"""


def test_parse_urlset():
    r = parse_sitemap(URLSET)
    assert not r.is_index
    assert len(r.entries) == 2
    assert r.entries[0].url == "https://x.com/a"
    assert r.entries[0].lastmod == "2026-09-01"
    assert r.entries[0].changefreq == "daily"
    assert r.entries[0].priority == 0.9
    assert r.entries[1].url == "https://x.com/b"
    assert r.entries[1].lastmod == "2026-09-15"


def test_parse_index():
    r = parse_sitemap(SITEMAPINDEX)
    assert r.is_index
    assert r.child_sitemaps == [
        "https://x.com/sitemap-1.xml",
        "https://x.com/sitemap-2.xml",
    ]
    assert r.entries == []


def test_parse_empty():
    r = parse_sitemap("")
    assert r.entries == [] and r.child_sitemaps == [] and not r.is_index


def test_malformed_xml_raises():
    with pytest.raises(ValueError):
        parse_sitemap("<urlset><url><loc>oops")


def test_skips_url_without_loc():
    xml = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><lastmod>2026-01-01</lastmod></url>
      <url><loc>https://x.com/kept</loc></url>
    </urlset>"""
    r = parse_sitemap(xml)
    assert [e.url for e in r.entries] == ["https://x.com/kept"]


def test_unknown_root_returns_empty():
    r = parse_sitemap("<somethingelse/>")
    assert r.entries == [] and r.child_sitemaps == [] and not r.is_index


# ---------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------

ROBOTS = """User-agent: *
Disallow: /admin
Allow: /admin/public
Disallow: /private
Sitemap: https://x.com/sitemap.xml
"""


def test_parse_robots_wildcard():
    r = parse_robots(ROBOTS, user_agent="*")
    assert r.disallow == ["/admin", "/private"]
    assert r.allow == ["/admin/public"]
    assert r.sitemaps == ["https://x.com/sitemap.xml"]


def test_robots_disallow():
    r = parse_robots(ROBOTS)
    assert not robots_allows("/admin/x", r)
    assert not robots_allows("/private/z", r)


def test_robots_longer_allow_wins():
    r = parse_robots(ROBOTS)
    assert robots_allows("/admin/public/y", r)


def test_robots_unlisted_path_allowed():
    r = parse_robots(ROBOTS)
    assert robots_allows("/products/a", r)


def test_robots_empty_disallow_means_all_allowed():
    r = parse_robots("User-agent: *\nDisallow:\n")
    assert robots_allows("/anything", r)


def test_robots_ua_specific_section():
    text = """User-agent: *
Disallow: /

User-agent: acmebot
Disallow: /nope
"""
    r = parse_robots(text, user_agent="AcmeBot")
    assert r.disallow == ["/nope"]


def test_robots_falls_back_to_wildcard():
    text = """User-agent: SomeOtherBot
Disallow: /x

User-agent: *
Disallow: /y
"""
    r = parse_robots(text, user_agent="Nobody")
    assert r.disallow == ["/y"]


def test_robots_wildcard_pattern():
    r = parse_robots("User-agent: *\nDisallow: /*.pdf$\n")
    assert not robots_allows("/docs/manual.pdf", r)
    assert robots_allows("/docs/manual.html", r)


def test_robots_comments_ignored():
    text = """User-agent: *   # this is the wildcard
Disallow: /secret  # keep out
"""
    r = parse_robots(text)
    assert r.disallow == ["/secret"]