"""Unit tests for SitemapResolver (spec §10, §14.1)."""
import pytest

from src.intake.sitemap_resolver import SitemapResolver


URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://x.com/a</loc></url>
  <url><loc>https://x.com/b</loc></url>
</urlset>
"""

SITEMAPINDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://x.com/sm-1.xml</loc></sitemap>
  <sitemap><loc>https://x.com/sm-2.xml</loc></sitemap>
</sitemapindex>
"""

CHILD_1 = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://x.com/p1</loc></url>
</urlset>
"""

CHILD_2 = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://x.com/p2</loc></url>
</urlset>
"""


def fake_fetcher(responses):
    def _f(url):
        if url not in responses:
            raise RuntimeError(f"no fake response for {url}")
        return responses[url]
    return _f


def test_urlset_direct():
    resolver = SitemapResolver(fetcher=fake_fetcher({"https://x.com/sitemap.xml": URLSET}))
    r = resolver.resolve("https://x.com/sitemap.xml")
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]
    assert r.warnings == []


def test_index_follows_children():
    responses = {
        "https://x.com/sitemap.xml": SITEMAPINDEX,
        "https://x.com/sm-1.xml": CHILD_1,
        "https://x.com/sm-2.xml": CHILD_2,
    }
    resolver = SitemapResolver(fetcher=fake_fetcher(responses))
    r = resolver.resolve("https://x.com/sitemap.xml")
    assert set(r.spec.target.start_urls) == {"https://x.com/p1", "https://x.com/p2"}


def test_duplicate_urls_deduped_after_normalization():
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://x.com/a</loc></url>
      <url><loc>https://x.com/a?utm_source=foo</loc></url>
    </urlset>
    """
    resolver = SitemapResolver(fetcher=fake_fetcher({"u": xml}))
    r = resolver.resolve("u")
    assert r.spec.target.start_urls == ["https://x.com/a"]


def test_child_fetch_failure_is_warning_not_raise():
    responses = {
        "https://x.com/sitemap.xml": SITEMAPINDEX,
        "https://x.com/sm-1.xml": CHILD_1,
        # sm-2 deliberately missing
    }
    resolver = SitemapResolver(fetcher=fake_fetcher(responses))
    r = resolver.resolve("https://x.com/sitemap.xml")
    assert r.spec.target.start_urls == ["https://x.com/p1"]
    assert any("sm-2" in w for w in r.warnings)


def test_malformed_root_returns_empty_with_warning():
    resolver = SitemapResolver(fetcher=fake_fetcher({"u": "<broken"}))
    r = resolver.resolve("u")
    assert r.spec.target.start_urls == []
    assert any("malformed" in w.lower() for w in r.warnings)


def test_root_fetch_failure_raises():
    def bad_fetcher(url):
        raise RuntimeError("network down")
    resolver = SitemapResolver(fetcher=bad_fetcher)
    with pytest.raises(RuntimeError):
        resolver.resolve("u")


def test_max_urls_cap():
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://x.com/a</loc></url>
      <url><loc>https://x.com/b</loc></url>
      <url><loc>https://x.com/c</loc></url>
    </urlset>
    """
    resolver = SitemapResolver(fetcher=fake_fetcher({"u": xml}), max_urls=2)
    r = resolver.resolve("u")
    assert len(r.spec.target.start_urls) == 2
    assert any("max_urls" in w for w in r.warnings)


def test_max_child_sitemaps_cap():
    xml_parts = [
        '<?xml version="1.0"?>',
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for i in range(5):
        xml_parts.append(f"<sitemap><loc>https://x.com/sm-{i}.xml</loc></sitemap>")
    xml_parts.append("</sitemapindex>")
    many = "".join(xml_parts)

    responses = {"https://x.com/sitemap.xml": many}
    for i in range(5):
        responses[f"https://x.com/sm-{i}.xml"] = CHILD_1

    resolver = SitemapResolver(fetcher=fake_fetcher(responses), max_child_sitemaps=2)
    r = resolver.resolve("https://x.com/sitemap.xml")
    assert any("truncated" in w for w in r.warnings)


def test_nested_index_warned_not_followed():
    nested = """<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://x.com/deeper.xml</loc></sitemap>
    </sitemapindex>
    """
    responses = {
        "https://x.com/sitemap.xml": SITEMAPINDEX,
        "https://x.com/sm-1.xml": nested,
        "https://x.com/sm-2.xml": CHILD_2,
    }
    resolver = SitemapResolver(fetcher=fake_fetcher(responses))
    r = resolver.resolve("https://x.com/sitemap.xml")
    assert r.spec.target.start_urls == ["https://x.com/p2"]
    assert any("nested" in w for w in r.warnings)


def test_source_description_mentions_sitemap():
    resolver = SitemapResolver(fetcher=fake_fetcher({"u": URLSET}))
    r = resolver.resolve("u")
    assert "sitemap" in r.source_description.lower()