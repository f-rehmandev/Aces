"""Unit tests for the network provider abstraction (spec §15.5)."""
import pytest

from src.network.types import (
    BudgetExceeded, FetchResult, NetworkRoute, ProviderError,
    ProviderName, ProviderNotConfigured, Transport, looks_blocked,
)


def test_route_defaults():
    r = NetworkRoute()
    assert r.provider == "direct"
    assert r.transport == "direct_http"
    assert r.estimated_credits == 0


def test_fetch_result_ok():
    r = FetchResult(url="https://x", html="<html>" + "y" * 1000,
                    status_code=200, provider="direct")
    assert r.is_ok
    assert not r.is_empty()


def test_fetch_result_empty():
    r = FetchResult(url="https://x", html="", status_code=200)
    assert r.is_empty()


def test_fetch_result_blocked_not_ok():
    r = FetchResult(url="https://x", html="<html>" + "y" * 1000,
                    status_code=200, blocked=True)
    assert not r.is_ok


def test_looks_blocked_javascript_marker():
    blocked, reason = looks_blocked("<html>Please enable JavaScript</html>")
    assert blocked
    assert "javascript" in reason.lower()


def test_looks_blocked_cloudflare_marker():
    blocked, reason = looks_blocked("<html>Just a moment...</html>")
    assert blocked


def test_looks_blocked_tiny_body():
    blocked, reason = looks_blocked("hi")
    assert blocked
    assert "small" in reason.lower()


def test_looks_blocked_healthy_page():
    html = "<html><body>" + "content " * 500 + "</body></html>"
    blocked, reason = looks_blocked(html)
    assert not blocked
    assert reason == ""


def test_looks_blocked_none():
    blocked, reason = looks_blocked(None)
    assert blocked


def test_provider_error_shape():
    e = ProviderError("scraperapi", "bad", retryable=True)
    assert e.provider == "scraperapi"
    assert e.retryable is True


def test_provider_not_configured_message():
    e = ProviderNotConfigured("scraperapi", "SCRAPERAPI_KEY")
    assert "SCRAPERAPI_KEY" in str(e)
    assert e.missing == "SCRAPERAPI_KEY"


def test_budget_exceeded_message():
    e = BudgetExceeded("scraperapi", needed=5, available=2)
    assert "5" in str(e) and "2" in str(e)
    assert e.retryable is False