"""
SSRF guard — spec §50.

Protects ACES's own infrastructure from user-supplied URLs. Every fetch
target must pass `validate_url()` before any network call.

DNS resolution is injected so tests can simulate rebinding without making
real DNS queries.
"""

from __future__ import annotations
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse


# Suffixes that should never be fetched, regardless of what they resolve to.
_FORBIDDEN_SUFFIXES = (".local", ".internal", ".localhost", ".lan", ".home")

# Cloud metadata endpoints (AWS/GCP/Azure/Alibaba/Oracle).
_FORBIDDEN_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.goog",
    "100.100.100.200",   # Alibaba
    "192.0.0.192",       # Oracle
}

_ALLOWED_SCHEMES = ("http", "https")


@dataclass
class SSRFCheckResult:
    allowed: bool
    reason: str = ""
    resolved_ip: Optional[str] = None          # the pinned IP, if resolved
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IP classification
# ---------------------------------------------------------------------------

def _classify_ip(ip: ipaddress._BaseAddress) -> Optional[str]:
    """
    Returns a short reason string if the IP is private/reserved/etc., else None.
    """
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_private:
        return "private"
    # Explicit ranges that some ipaddress versions don't classify as private
    if ip.version == 4:
        n = int(ip)
        # 0.0.0.0/8
        if 0 <= n < (1 << 24):
            return "reserved"
    return None


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

class SSRFGuard:
    """
    Validates URLs against SSRF risks.

    `resolver` is a callable that takes a hostname and returns a list of IP
    strings. Default uses `socket.getaddrinfo`. Inject a fake in tests.
    """

    def __init__(
        self,
        resolver: Optional[Callable[[str], list[str]]] = None,
        max_redirects: int = 5,
        max_response_bytes: int = 10 * 1024 * 1024,   # 10 MB
    ):
        self._resolver = resolver or _default_resolver
        self.max_redirects = max_redirects
        self.max_response_bytes = max_response_bytes

    def validate_url(self, url: str) -> SSRFCheckResult:
        """Full validation: scheme, hostname, IP class, and resolution."""
        if not url:
            return SSRFCheckResult(False, "empty URL")

        try:
            parts = urlparse(url)
        except ValueError as e:
            return SSRFCheckResult(False, f"unparseable URL: {e}")

        # --- scheme ---
        if parts.scheme.lower() not in _ALLOWED_SCHEMES:
            return SSRFCheckResult(False, f"scheme not allowed: {parts.scheme!r}")

        # --- userinfo (user:pass@host) is a URL-spoofing risk ---
        if "@" in (parts.netloc or ""):
            return SSRFCheckResult(False, "userinfo in URL")

        host = (parts.hostname or "").lower()
        if not host:
            return SSRFCheckResult(False, "no hostname")

        # --- forbidden suffixes ---
        for suffix in _FORBIDDEN_SUFFIXES:
            if host == suffix.lstrip(".") or host.endswith(suffix):
                return SSRFCheckResult(False, f"forbidden hostname suffix: {suffix}")

        # --- cloud metadata endpoints ---
        if host in _FORBIDDEN_HOSTS:
            return SSRFCheckResult(False, "cloud metadata endpoint")

        # --- IP literal host ---
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip is not None:
            reason = _classify_ip(ip)
            if reason:
                return SSRFCheckResult(False, f"IP literal in {reason} range")
            return SSRFCheckResult(True, "ip-literal ok", resolved_ip=str(ip))

        # --- resolve ---
        try:
            ips = self._resolver(host)
        except Exception as e:
            return SSRFCheckResult(False, f"DNS resolution failed: {e}")

        if not ips:
            return SSRFCheckResult(False, "hostname did not resolve")

        # Every resolved IP must be public. If any is private, reject.
        for ip_str in ips:
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError:
                return SSRFCheckResult(False, f"unparseable IP from resolver: {ip_str}")
            reason = _classify_ip(ip_obj)
            if reason:
                return SSRFCheckResult(
                    False,
                    f"hostname {host!r} resolved to {reason} IP {ip_str}",
                )

        # Pin the first resolved IP for the caller to use.
        return SSRFCheckResult(True, "resolved ok", resolved_ip=ips[0])

    def validate_redirect_chain(self, urls: list[str]) -> SSRFCheckResult:
        """
        §50.3: every redirect hop must be re-validated.
        """
        if len(urls) > self.max_redirects:
            return SSRFCheckResult(False, f"redirect chain exceeds {self.max_redirects} hops")
        for i, u in enumerate(urls):
            result = self.validate_url(u)
            if not result.allowed:
                return SSRFCheckResult(
                    False,
                    f"redirect hop {i} rejected: {result.reason}",
                )
        return SSRFCheckResult(True, "all redirect hops ok")


def _default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return list({info[4][0] for info in infos})


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default = SSRFGuard()

def validate_url(url: str) -> SSRFCheckResult:
    return _default.validate_url(url)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def fake_resolver(host):
        return {
            "public.example": ["93.184.216.34"],
            "rebinding.evil": ["93.184.216.34", "127.0.0.1"],
            "private.internal": ["10.0.0.5"],
        }.get(host, [])

    guard = SSRFGuard(resolver=fake_resolver)

    # Ok
    assert guard.validate_url("https://public.example/a").allowed
    assert guard.validate_url("http://93.184.216.34/a").allowed
    assert guard.validate_url("http://[2606:4700:4700::1111]/").allowed

    # Bad scheme
    assert not guard.validate_url("file:///etc/passwd").allowed
    assert not guard.validate_url("gopher://x").allowed
    assert not guard.validate_url("ftp://x").allowed

    # Userinfo
    assert not guard.validate_url("https://user:pass@public.example/").allowed

    # Forbidden suffixes
    assert not guard.validate_url("http://foo.localhost/").allowed
    assert not guard.validate_url("http://svc.internal/").allowed

    # Metadata endpoints
    assert not guard.validate_url("http://169.254.169.254/latest/meta-data/").allowed
    assert not guard.validate_url("http://metadata.google.internal/").allowed

    # Private/reserved IP literals
    assert not guard.validate_url("http://127.0.0.1/").allowed
    assert not guard.validate_url("http://10.0.0.1/").allowed
    assert not guard.validate_url("http://192.168.1.1/").allowed
    assert not guard.validate_url("http://[::1]/").allowed
    assert not guard.validate_url("http://0.0.0.0/").allowed

    # DNS rebinding — one private IP in the set → reject
    assert not guard.validate_url("https://rebinding.evil/").allowed
    assert not guard.validate_url("http://private.internal/").allowed

    # Empty / invalid
    assert not guard.validate_url("").allowed
    assert not guard.validate_url("not a url").allowed

    # Redirect chain
    chain = ["https://public.example/a", "https://public.example/b"]
    assert guard.validate_redirect_chain(chain).allowed
    bad_chain = ["https://public.example/a", "http://169.254.169.254/"]
    assert not guard.validate_redirect_chain(bad_chain).allowed

    # Redirect hop cap
    too_many = ["https://public.example/a"] * 10
    assert not guard.validate_redirect_chain(too_many).allowed

    print("SSRF guard OK.")