"""Unit tests for website classification."""

import pytest

from src.discovery.website_classifier import (
    classify_website, has_real_website,
)


# ---------------------------------------------------------------------------
# Real websites
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://pizzavizza.pk",
    "http://example.com",
    "example.com",                          # bare host
    "https://www.example.com/shop/cart",
    "https://subdomain.example.co.uk/path",
    "https://pizzavizza.myshopify.com",
    "https://shop.pizzavizza.pk",
])
def test_own_website(url):
    assert classify_website(url) == "own"
    assert has_real_website(url) is True


# ---------------------------------------------------------------------------
# Social
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.facebook.com/pizzavizza",
    "http://facebook.com/pizzavizza",
    "facebook.com/pizzavizza",
    "https://m.facebook.com/pizzavizza",
    "https://business.facebook.com/pizzavizza",
    "https://fb.me/pizzavizza",
    "https://www.instagram.com/pizzavizza",
    "https://instagram.com/pizzavizza",
    "https://twitter.com/pizzavizza",
    "https://x.com/pizzavizza",
    "https://linkedin.com/company/pizzavizza",
    "https://youtube.com/@pizzavizza",
    "https://www.tiktok.com/@pizzavizza",
    "https://wa.me/923001234567",
    "https://api.whatsapp.com/send?phone=923001234567",
    "https://t.me/pizzavizza",
    "https://linktr.ee/pizzavizza",
    "https://beacons.ai/pizzavizza",
    "https://foodpanda.pk/restaurant/pizzavizza",
    "https://www.ubereats.com/pk/store/pizzavizza",
    "https://www.zomato.com/pizzavizza",
    "https://www.booking.com/hotel/pizzavizza",
    "https://www.amazon.com/dp/B000",
    "https://daraz.pk/products/pizzavizza",
    "https://www.yelp.com/biz/pizzavizza",
    "https://www.yellowpages.com/pizzavizza",
])
def test_social_platform(url):
    assert classify_website(url) == "social"
    assert has_real_website(url) is False


# ---------------------------------------------------------------------------
# None / empty
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", None, "   ", "\t\n"])
def test_no_website(value):
    assert classify_website(value) == "none"
    assert has_real_website(value) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_case_insensitive():
    assert classify_website("HTTPS://WWW.FACEBOOK.COM/X") == "social"
    assert classify_website("HTTPS://WWW.INSTAGRAM.COM/X") == "social"
    assert classify_website("HTTPS://WWW.PIZZAVIZZA.PK") == "own"


def test_www_stripped():
    assert classify_website("https://www.facebook.com/x") == "social"
    assert classify_website("https://www.pizzavizza.pk") == "own"


def test_subdomain_of_social_is_social():
    assert classify_website("https://m.facebook.com/x") == "social"
    assert classify_website("https://business.instagram.com/x") == "social"


def test_subdomain_of_own_is_own():
    assert classify_website("https://shop.mysite.com") == "own"
    assert classify_website("https://api.mysite.co") == "own"


def test_bare_domain():
    assert classify_website("facebook.com") == "social"
    assert classify_website("pizzavizza.pk") == "own"


def test_garbage_url_classified_as_none():
    # urlparse can't find a host
    assert classify_website("::::not a url") == "none"
    assert classify_website("   ") == "none"