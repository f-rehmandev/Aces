"""Unit tests for the SSRF guard (spec §50)."""
import pytest

from src.security.ssrf import SSRFGuard, validate_url


def _guard():
    def fake(host):
        return {
            "public.example": ["93.184.216.34"],
            "multi.example": ["93.184.216.34", "1.1.1.1"],
            "rebinding.evil": ["93.184.216.34", "127.0.0.1"],
            "private.internal": ["10.0.0.5"],
            "loopback.example": ["127.0.0.1"],
            "v6.example": ["2606:4700:4700::1111"],
        }.get(host, [])
    return SSRFGuard(resolver=fake)


# --- allowed ------------------------------------------------------------

def test_public_https_allowed():
    assert _guard().validate_url("https://public.example/a").allowed


def test_public_ip_literal_allowed():
    assert _guard().validate_url("http://93.184.216.34/a").allowed


def test_public_v6_literal_allowed():
    assert _guard().validate_url("http://[2606:4700:4700::1111]/").allowed


def test_multi_ip_resolution_allowed_when_all_public():
    assert _guard().validate_url("https://multi.example/").allowed


# --- scheme ------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["file", "gopher", "ftp", "dict", "data"])
def test_non_http_schemes_rejected(scheme):
    assert not _guard().validate_url(f"{scheme}://x.com/").allowed


# --- userinfo ----------------------------------------------------------

def test_userinfo_rejected():
    assert not _guard().validate_url("https://user:pass@public.example/").allowed


# --- forbidden suffixes -------------------------------------------------

@pytest.mark.parametrize("host", [
    "foo.localhost", "svc.internal", "printer.local", "thing.lan", "box.home",
])
def test_forbidden_suffixes_rejected(host):
    assert not _guard().validate_url(f"http://{host}/").allowed


# --- metadata endpoints -------------------------------------------------

def test_aws_metadata_rejected():
    assert not _guard().validate_url("http://169.254.169.254/").allowed


def test_gcp_metadata_rejected():
    assert not _guard().validate_url("http://metadata.google.internal/").allowed


def test_alibaba_metadata_rejected():
    assert not _guard().validate_url("http://100.100.100.200/").allowed


# --- IP literal classes ------------------------------------------------

@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1",
    "0.0.0.0", "224.0.0.1", "255.255.255.255", "169.254.1.1",
])
def test_private_and_reserved_ip_literals_rejected(ip):
    assert not _guard().validate_url(f"http://{ip}/").allowed


def test_loopback_v6_rejected():
    assert not _guard().validate_url("http://[::1]/").allowed


# --- DNS rebinding ------------------------------------------------------

def test_rebinding_rejected():
    assert not _guard().validate_url("https://rebinding.evil/").allowed


def test_hostname_resolving_to_private_rejected():
    assert not _guard().validate_url("http://private.internal/").allowed


def test_hostname_resolving_to_loopback_rejected():
    assert not _guard().validate_url("http://loopback.example/").allowed


def test_unresolvable_host_rejected():
    assert not _guard().validate_url("http://does-not-resolve.example/").allowed


# --- malformed inputs --------------------------------------------------

def test_empty_url_rejected():
    assert not _guard().validate_url("").allowed


def test_whitespace_only_rejected():
    assert not _guard().validate_url("   ").allowed


def test_no_hostname_rejected():
    assert not _guard().validate_url("http:///path").allowed


# --- redirect chain -----------------------------------------------------

def test_redirect_chain_all_public_ok():
    chain = ["https://public.example/a", "https://public.example/b"]
    assert _guard().validate_redirect_chain(chain).allowed


def test_redirect_chain_with_metadata_hop_rejected():
    chain = ["https://public.example/a", "http://169.254.169.254/"]
    assert not _guard().validate_redirect_chain(chain).allowed


def test_redirect_chain_too_many_hops_rejected():
    chain = ["https://public.example/a"] * 10
    assert not _guard().validate_redirect_chain(chain).allowed


# --- resolved_ip pinning -----------------------------------------------

def test_resolved_ip_is_pinned():
    r = _guard().validate_url("https://public.example/")
    assert r.allowed
    assert r.resolved_ip == "93.184.216.34"


def test_ip_literal_pins_itself():
    r = _guard().validate_url("http://93.184.216.34/a")
    assert r.allowed
    assert r.resolved_ip == "93.184.216.34"


# --- module-level convenience ------------------------------------------

def test_module_level_validate_url_rejects_localhost():
    # Uses the real resolver, but localhost is always forbidden
    assert not validate_url("http://localhost/").allowed