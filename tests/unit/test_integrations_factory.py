"""
Unit tests for the production connector factory (spec §47A).

Covers:
    - Empty env → empty registry, no exception
    - Each connector (local_file, slack, webhook, s3) registers when
      its env vars are present, skipped when they're not
    - Webhook requires BOTH url + secret
    - Multiple connectors register together
    - Bad config: strict=False skips, strict=True raises
    - `client_id` propagation into the registry
    - DNS-dependent SSRF checks are deferred to send time
      (regression guard for the constructor-time I/O bug we fixed in
      Y.7.1-fix)
"""
import os
import tempfile

import pytest
from unittest.mock import patch

from src.integrations.factory import build_production_registry


# ---------------------------------------------------------------------------
# Helper: patch.dict with clear=True so the real .env doesn't leak in.
# ---------------------------------------------------------------------------

def _with_env(**env):
    return patch.dict(os.environ, env, clear=True)


# ===========================================================================
# Empty env
# ===========================================================================

def test_empty_env_returns_empty_registry():
    with _with_env():
        reg = build_production_registry()
    assert len(reg) == 0


def test_empty_env_strict_still_ok():
    """No configured connectors is not an error, even in strict mode."""
    with _with_env():
        reg = build_production_registry(strict=True)
    assert len(reg) == 0


# ===========================================================================
# LocalFile
# ===========================================================================

def test_local_file_registers_when_dir_set():
    with tempfile.TemporaryDirectory() as tmp:
        with _with_env(ACES_LOCAL_OUTPUT_DIR=tmp):
            reg = build_production_registry()
    assert len(reg) == 1
    c = next(iter(reg.all()))
    assert c.connector_type.value == "local_file"


def test_local_file_skipped_when_dir_unset():
    with _with_env():
        reg = build_production_registry()
    types = [c.connector_type.value for c in reg.all()]
    assert "local_file" not in types




# ===========================================================================
# Slack
# ===========================================================================

def test_slack_registers_when_url_set():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
    ):
        reg = build_production_registry()
    assert len(reg) == 1
    c = next(iter(reg.all()))
    assert c.connector_type.value == "slack"


def test_slack_channel_override_applied():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
        SLACK_CHANNEL="#alerts",
    ):
        reg = build_production_registry()
    c = next(iter(reg.all()))
    assert c.channel == "#alerts"


def test_slack_username_default_applied():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
    ):
        reg = build_production_registry()
    c = next(iter(reg.all()))
    assert c.username == "ACES"


def test_slack_username_override_applied():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
        SLACK_USERNAME="ACES Prod",
    ):
        reg = build_production_registry()
    c = next(iter(reg.all()))
    assert c.username == "ACES Prod"


def test_slack_skipped_when_url_unset():
    with _with_env():
        reg = build_production_registry()
    types = [c.connector_type.value for c in reg.all()]
    assert "slack" not in types


def test_slack_bad_url_skipped_in_lenient_mode():
    with _with_env(SLACK_WEBHOOK_URL="not-a-url"):
        reg = build_production_registry(strict=False)
    assert len(reg) == 0


def test_slack_bad_url_raises_in_strict_mode():
    with _with_env(SLACK_WEBHOOK_URL="not-a-url"):
        with pytest.raises(Exception):
            build_production_registry(strict=True)


# ===========================================================================
# Webhook — requires BOTH url and secret
# ===========================================================================

def test_webhook_requires_url_and_secret():
    with _with_env(ACES_WEBHOOK_URL="https://example.com/hook"):
        reg = build_production_registry()
    assert len(reg) == 0


def test_webhook_requires_secret_even_with_url():
    with _with_env(ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx"):
        reg = build_production_registry()
    assert len(reg) == 0


def test_webhook_registers_when_both_set():
    with _with_env(
        ACES_WEBHOOK_URL="https://93.184.216.34/hook",
        ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx",
    ):
        reg = build_production_registry()
    # Use a public IP literal so the SSRF guard doesn't need DNS
    assert len(reg) == 1
    c = next(iter(reg.all()))
    assert c.connector_type.value == "webhook"


def test_webhook_allow_http_flag():
    # Note: we do NOT use http://localhost/ here — the SSRF guard
    # forbids the `.localhost` suffix regardless of scheme, which
    # would test the guard, not the allow_http flag. A public IP
    # literal keeps the guard happy and lets us exercise the flag.
    with _with_env(
        ACES_WEBHOOK_URL="http://93.184.216.34:9000/hook",
        ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx",
        ACES_WEBHOOK_ALLOW_HTTP="true",
    ):
        reg = build_production_registry()
    assert len(reg) == 1
    c = next(iter(reg.all()))
    assert c.url.startswith("http://")


def test_webhook_http_rejected_without_flag():
    """Without allow_http=true, an http URL must be refused."""
    with _with_env(
        ACES_WEBHOOK_URL="http://93.184.216.34:9000/hook",
        ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx",
        # ACES_WEBHOOK_ALLOW_HTTP deliberately unset
    ):
        reg = build_production_registry(strict=False)
    assert len(reg) == 0


# ===========================================================================
# S3
# ===========================================================================

def test_s3_registers_when_bucket_set():
    with _with_env(S3_BUCKET="my-bucket"):
        reg = build_production_registry()
    assert len(reg) == 1
    c = next(iter(reg.all()))
    assert c.connector_type.value == "s3"
    assert c.bucket == "my-bucket"


def test_s3_prefix_and_region_applied():
    with _with_env(
        S3_BUCKET="my-bucket",
        S3_PREFIX="datasets/2026",
        S3_REGION="eu-west-1",
    ):
        reg = build_production_registry()
    c = next(iter(reg.all()))
    assert c.prefix == "datasets/2026"
    assert c.region == "eu-west-1"


def test_s3_skipped_when_bucket_unset():
    with _with_env():
        reg = build_production_registry()
    types = [c.connector_type.value for c in reg.all()]
    assert "s3" not in types


# ===========================================================================
# Multiple connectors together
# ===========================================================================

def test_all_four_connectors_together():
    with tempfile.TemporaryDirectory() as tmp:
        with _with_env(
            ACES_LOCAL_OUTPUT_DIR=tmp,
            SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
            ACES_WEBHOOK_URL="https://93.184.216.34/hook",
            ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx",
            S3_BUCKET="my-bucket",
        ):
            reg = build_production_registry()
    assert len(reg) == 4
    types = sorted(c.connector_type.value for c in reg.all())
    assert types == ["local_file", "s3", "slack", "webhook"]


def test_two_connectors_together():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
        S3_BUCKET="my-bucket",
    ):
        reg = build_production_registry()
    assert len(reg) == 2


# ===========================================================================
# Client scoping
# ===========================================================================

def test_client_id_propagates_to_registry():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
    ):
        reg = build_production_registry(client_id="acme")
    # The connector is owned by "acme"
    c = next(iter(reg.all()))
    # registry.all() with the same client_id returns it
    assert c in reg.all(client_id="acme")
    # And NOT visible to other clients
    assert c not in reg.all(client_id="other")


def test_default_client_id_is_global():
    with _with_env(
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/X/Y/Z",
    ):
        reg = build_production_registry(client_id="")
    # Global connectors are visible to every client
    assert len(reg.all(client_id="acme")) == 1
    assert len(reg.all(client_id="other")) == 1


# ===========================================================================
# Regression: DNS-dependent SSRF checks must not block construction
# ===========================================================================

def test_webhook_construction_does_not_require_dns():
    """
    Regression guard for Y.7.1-fix.

    A webhook URL whose hostname cannot be resolved right now must
    STILL register — the SSRF check at construction time only enforces
    structural rules (scheme, userinfo, suffix, IP literal). The
    DNS-dependent check runs on every send, not at construction.

    This test uses a hostname that definitely has no DNS entry
    (`does-not-exist.invalid` is a reserved TLD per RFC 2606).
    """
    with _with_env(
        ACES_WEBHOOK_URL="https://does-not-exist.invalid/hook",
        ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx",
    ):
        reg = build_production_registry()
    assert len(reg) == 1
    c = next(iter(reg.all()))
    assert c.connector_type.value == "webhook"


def test_metadata_ip_still_rejected_at_construction():
    """
    The hard SSRF check (metadata IP literal) must still fire at
    construction — deferred DNS does not mean deferred safety.
    """
    with _with_env(
        ACES_WEBHOOK_URL="https://169.254.169.254/hook",
        ACES_WEBHOOK_SECRET="long-enough-secret-xxxxxxxx",
    ):
        reg = build_production_registry(strict=False)
    # Skipped, not registered
    assert len(reg) == 0


# ===========================================================================
# Whitespace / env hygiene
# ===========================================================================

def test_env_values_are_stripped():
    with _with_env(
        SLACK_WEBHOOK_URL="  https://hooks.slack.com/services/X/Y/Z  ",
        SLACK_CHANNEL="  #alerts  ",
    ):
        reg = build_production_registry()
    c = next(iter(reg.all()))
    assert c.channel == "#alerts"


def test_empty_string_env_treated_as_unset():
    with _with_env(
        SLACK_WEBHOOK_URL="",
        S3_BUCKET="",
        ACES_LOCAL_OUTPUT_DIR="",
    ):
        reg = build_production_registry()
    assert len(reg) == 0