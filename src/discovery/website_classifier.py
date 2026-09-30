"""
Website classification — spec §27 (data cleaning).

Google Maps returns `facebook.com/page-name`, `wa.me/923001234567`,
`linktr.ee/xyz`, etc. as "websites" for small businesses that never built
their own site. For lead-gen purposes, those do not count as having a
real website.

Public API:
    classify_website(url)  -> "own" | "social" | "none"
    has_real_website(url)  -> bool
"""

from __future__ import annotations
from typing import Optional
from urllib.parse import urlparse
import re

# Domains that are NOT real websites for lead-gen purposes:
# social networks, messengers, link aggregators, marketplaces.
_SOCIAL_DOMAINS = {
    # Meta
    "facebook.com", "fb.com", "fb.me", "m.facebook.com", "mbasic.facebook.com",
    "instagram.com", "instagr.am",
    # Twitter / X
    "twitter.com", "x.com", "t.co",
    # LinkedIn
    "linkedin.com", "lnkd.in",
    # Video / streaming
    "youtube.com", "youtu.be",
    "tiktok.com", "vm.tiktok.com",
    "vimeo.com", "twitch.tv",
    # Messaging
    "wa.me", "api.whatsapp.com", "whatsapp.com",
    "t.me", "telegram.me",
    "m.me", "line.me",
    # Link aggregators / bio pages
    "linktr.ee",
    "beacons.ai", "beacons.page",
    "carrd.co",
    # Food delivery platforms
    "foodpanda.pk", "foodpanda.com",
    "ubereats.com", "doordash.com", "grubhub.com",
    "zomato.com", "swiggy.com",
    # Booking platforms
    "booking.com", "airbnb.com", "tripadvisor.com",
    # E-commerce marketplaces
    "amazon.com", "daraz.pk", "ebay.com", "etsy.com",
    # Review / directory sites
    "yelp.com", "yellowpages.com",
}


def _hostname(url: str) -> str:
    if not url:
        return ""
    s = str(url).strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s
    try:
        return (urlparse(s).netloc or "").lower()
    except Exception:
        return ""


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


# A hostname must look like a real domain: letters/digits, hyphens, and
# at least one dot. This filters out garbage like "::::not a url".
_VALID_HOST_RE = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,}$",
    re.IGNORECASE,
)


def _host_looks_valid(host: str) -> bool:
    """True if the hostname resembles a real domain."""
    if not host:
        return False
    # IP addresses are technically valid hosts; allow them for completeness
    # (they're rare in this data but not impossible).
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        return True
    return bool(_VALID_HOST_RE.match(host))


def classify_website(url: Optional[str]) -> str:
    """
    Return one of:
        "own"     — a real (non-social, non-aggregator) website
        "social"  — a social/aggregator/marketplace URL
        "none"    — no URL at all, or a URL that doesn't look like a
                    real domain
    """
    if not url or not str(url).strip():
        return "none"

    host = _strip_www(_hostname(url))
    if not host:
        return "none"

    # Reject garbage that urlparse is too permissive to reject itself
    if not _host_looks_valid(host):
        return "none"

    if host in _SOCIAL_DOMAINS:
        return "social"

    # Suffix match handles dynamic subdomains like mypage.facebook.com
    for domain in _SOCIAL_DOMAINS:
        if host.endswith("." + domain):
            return "social"

    return "own"


def has_real_website(url: Optional[str]) -> bool:
    """True only if the URL is a non-social, non-aggregator site."""
    return classify_website(url) == "own"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Real website
    assert classify_website("https://pizzavizza.pk") == "own"
    assert classify_website("pizzavizza.pk") == "own"
    assert has_real_website("http://example.com/shop") is True

    # Social pages — should be classified as social
    assert classify_website("https://www.facebook.com/pizzavizza") == "social"
    assert classify_website("facebook.com/pizzavizza") == "social"
    assert classify_website("https://m.facebook.com/pizzavizza") == "social"
    assert classify_website("https://business.facebook.com/pizzavizza") == "social"
    assert classify_website("https://instagram.com/pizzavizza") == "social"
    assert classify_website("https://www.instagram.com/pizzavizza") == "social"
    assert classify_website("https://wa.me/923001234567") == "social"
    assert classify_website("https://t.me/pizzavizza") == "social"
    assert classify_website("https://linktr.ee/pizzavizza") == "social"
    assert classify_website("https://foodpanda.pk/restaurant/pizzavizza") == "social"

    # Marketplaces
    assert classify_website("https://www.amazon.com/x") == "social"
    assert classify_website("https://daraz.pk/shop") == "social"

    # No website
    assert classify_website("") == "none"
    assert classify_website(None) == "none"
    assert classify_website("   ") == "none"
    assert has_real_website("") is False

    # Case and www. handled
    assert classify_website("HTTPS://WWW.FACEBOOK.COM/X") == "social"
    assert classify_website("https://WWW.Facebook.com/X") == "social"
    assert classify_website("HTTPS://WWW.PIZZAVIZZA.PK") == "own"

    print("Website classifier OK.")