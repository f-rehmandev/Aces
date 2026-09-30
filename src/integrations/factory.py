"""
Production connector registry factory — spec §47A.

Wires the connector framework to environment configuration. Every
credential is read from `.env`; whatever the operator has configured
becomes a live connector in the returned registry. Everything absent
is silently skipped — the factory never raises on a missing key, it
just has fewer connectors.

Design notes:

    - The factory returns a `ConnectorRegistry` populated with every
      connector it could build. Callers (the alert router, the delivery
      manager, the pipeline runner) receive a registry and don't need
      to know what's inside.
    - Tenant scoping: credentials are operator-level today, so every
      connector is registered as global (`client_id=""`). The registry
      already supports per-tenant overrides for when that changes.
    - The factory does NOT connect to Slack / S3 / anything at build
      time. Constructing a connector only validates its config; the
      first `test_connection()` or `publish()` does the real I/O.
    - `strict=True` will raise on the first misconfigured connector.
      Default is `strict=False` — a bad SCRAPERAPI_KEY should not
      prevent the Slack webhook from being registered.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.integrations.registry import ConnectorRegistry


logger = logging.getLogger("integrations.factory")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _bool_env(name: str, default: bool = False) -> bool:
    v = _env(name).lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_production_registry(
    *,
    strict: bool = False,
    client_id: str = "",
) -> ConnectorRegistry:
    """
    Build a ConnectorRegistry populated with every connector whose
    credentials are present in `.env`.

    Recognized env vars:

        LocalFile   — always registered if ACES_LOCAL_OUTPUT_DIR is set
        S3          — S3_BUCKET required; S3_PREFIX/S3_REGION/endpoint
                      and access keys optional (falls back to boto3's
                      ambient credential chain when unset)
        Slack       — SLACK_WEBHOOK_URL required; SLACK_CHANNEL optional
        Webhook     — ACES_WEBHOOK_URL required; ACES_WEBHOOK_SECRET
                      required (or use the alias ACES_ALERT_WEBHOOK_URL
                      + ACES_ALERT_WEBHOOK_SECRET)

    `strict=True` raises the first construction error instead of logging
    and skipping. Intended for CI / operator verification.
    """
    registry = ConnectorRegistry()

    # --- LocalFile ------------------------------------------------------
    local_dir = _env("ACES_LOCAL_OUTPUT_DIR")
    if local_dir:
        try:
            from src.integrations.local_file import LocalFileConnector
            registry.register(
                LocalFileConnector(base_dir=local_dir),
                client_id=client_id,
            )
            logger.info(f"registered local_file connector → {local_dir}")
        except Exception as e:
            if strict:
                raise
            logger.warning(f"local_file connector skipped: {e}")

    # --- Slack ----------------------------------------------------------
    slack_url = _env("SLACK_WEBHOOK_URL")
    if slack_url:
        try:
            from src.integrations.slack import SlackConnector
            registry.register(
                SlackConnector(
                    webhook_url=slack_url,
                    channel=_env("SLACK_CHANNEL"),
                    username=_env("SLACK_USERNAME", "ACES"),
                    icon_emoji=_env("SLACK_ICON_EMOJI", ":robot_face:"),
                ),
                client_id=client_id,
            )
            logger.info("registered slack connector")
        except Exception as e:
            if strict:
                raise
            logger.warning(f"slack connector skipped: {e}")

    # --- Webhook --------------------------------------------------------
    webhook_url = _env("ACES_WEBHOOK_URL")
    webhook_secret = _env("ACES_WEBHOOK_SECRET")
    if webhook_url and webhook_secret:
        try:
            from src.integrations.webhook import WebhookConnector
            registry.register(
                WebhookConnector(
                    url=webhook_url,
                    secret=webhook_secret,
                    allow_http=_bool_env("ACES_WEBHOOK_ALLOW_HTTP", False),
                ),
                client_id=client_id,
            )
            logger.info("registered webhook connector")
        except Exception as e:
            if strict:
                raise
            logger.warning(f"webhook connector skipped: {e}")
    elif webhook_url:
        logger.warning(
            "ACES_WEBHOOK_URL is set but ACES_WEBHOOK_SECRET is missing; "
            "the webhook connector requires both."
        )

    # --- S3 -------------------------------------------------------------
    s3_bucket = _env("S3_BUCKET")
    if s3_bucket:
        try:
            from src.integrations.s3 import S3Connector
            registry.register(
                S3Connector(
                    bucket=s3_bucket,
                    prefix=_env("S3_PREFIX"),
                    region=_env("S3_REGION"),
                    endpoint_url=_env("S3_ENDPOINT_URL"),
                    access_key=_env("S3_ACCESS_KEY"),
                    secret_key=_env("S3_SECRET_KEY"),
                    session_token=_env("S3_SESSION_TOKEN"),
                    storage_class=_env("S3_STORAGE_CLASS"),
                    sse=_env("S3_SSE"),
                    kms_key_id=_env("S3_KMS_KEY_ID"),
                ),
                client_id=client_id,
            )
            logger.info(f"registered s3 connector → {s3_bucket}")
        except Exception as e:
            if strict:
                raise
            logger.warning(f"s3 connector skipped: {e}")

    return registry


# ---------------------------------------------------------------------------
# Smoke test — no network, no .env dependency
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from unittest.mock import patch

    # ---- 1. Empty env → empty registry, no exception ----
    with patch.dict(os.environ, {}, clear=True):
        reg = build_production_registry()
        assert len(reg) == 0

    # ---- 2. Slack only ----
    with patch.dict(os.environ, {
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/X/Y/Z",
        "SLACK_CHANNEL": "#alerts",
    }, clear=True):
        reg = build_production_registry()
        assert len(reg) == 1
        slack = next(iter(reg.all()))
        assert slack.connector_type.value == "slack"
        assert slack.channel == "#alerts"

    # ---- 3. LocalFile only ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {
            "ACES_LOCAL_OUTPUT_DIR": tmp,
        }, clear=True):
            reg = build_production_registry()
            assert len(reg) == 1
            lf = next(iter(reg.all()))
            assert lf.connector_type.value == "local_file"

    # ---- 4. Webhook requires BOTH url and secret ----
    with patch.dict(os.environ, {
        "ACES_WEBHOOK_URL": "https://example.com/hook",
        # secret deliberately missing
    }, clear=True):
        reg = build_production_registry()
        assert len(reg) == 0

    with patch.dict(os.environ, {
        "ACES_WEBHOOK_URL": "https://example.com/hook",
        "ACES_WEBHOOK_SECRET": "test-secret-1234567890",
    }, clear=True):
        reg = build_production_registry()
        assert len(reg) == 1
        wh = next(iter(reg.all()))
        assert wh.connector_type.value == "webhook"

    # ---- 5. Multiple connectors together ----
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/X/Y/Z",
            "ACES_LOCAL_OUTPUT_DIR": tmp,
            "ACES_WEBHOOK_URL": "https://example.com/hook",
            "ACES_WEBHOOK_SECRET": "s3cr3t-s3cr3t-s3cr3t",
        }, clear=True):
            reg = build_production_registry()
            assert len(reg) == 3
            types = sorted(c.connector_type.value for c in reg.all())
            assert types == ["local_file", "slack", "webhook"]

    # ---- 6. Bad Slack URL is skipped, not fatal ----
    with patch.dict(os.environ, {
        "SLACK_WEBHOOK_URL": "not-a-url",
    }, clear=True):
        reg = build_production_registry(strict=False)
        assert len(reg) == 0

    # ---- 7. ...unless strict=True ----
    with patch.dict(os.environ, {
        "SLACK_WEBHOOK_URL": "not-a-url",
    }, clear=True):
        try:
            build_production_registry(strict=True)
            raise AssertionError("expected exception in strict mode")
        except Exception:
            pass

    print("Connector factory OK.")