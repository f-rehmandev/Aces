"""Unit tests for the Supabase auth backend adapter.

We mock the underlying supabase-py client entirely — no network, no
credentials. These tests verify the adapter translates between the
supabase-py shape and the AuthBackend protocol shape correctly.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.auth.supabase_backend import (
    SupabaseAuthBackend,
    _session_to_dict, _user_to_dict, _wrap,
)


# ---------------------------------------------------------------------------
# Helpers to build fake supabase responses
# ---------------------------------------------------------------------------

def _fake_user(uid="u-1", email="a@b.com", metadata=None):
    return SimpleNamespace(
        id=uid, email=email, user_metadata=metadata or {},
    )


def _fake_session(access="access-token", refresh="refresh-token",
                  expires=9_999_999_999, user=None):
    return SimpleNamespace(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires,
        user=user or _fake_user(),
    )


def _fake_auth_response(user=None, session=None):
    return SimpleNamespace(user=user, session=session)


def _build_backend_with_fake_client() -> SupabaseAuthBackend:
    """
    Build a SupabaseAuthBackend with a mocked `create_client` so no real
    Supabase project is contacted.
    """
    fake_client = MagicMock()
    with patch("src.auth.supabase_backend.create_client",
               return_value=fake_client):
        backend = SupabaseAuthBackend("https://example.supabase.co",
                                       "anon-key")
    return backend


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def test_user_to_dict_from_object():
    d = _user_to_dict(_fake_user("u-9", "x@y.com", {"full_name": "X"}))
    assert d["id"] == "u-9"
    assert d["email"] == "x@y.com"
    assert d["user_metadata"] == {"full_name": "X"}


def test_user_to_dict_from_dict():
    d = _user_to_dict({"id": "u-1", "email": "a@b.com"})
    assert d == {"id": "u-1", "email": "a@b.com"}


def test_user_to_dict_none():
    assert _user_to_dict(None) is None


def test_session_to_dict_from_object():
    s = _fake_session(access="A", refresh="R", expires=123)
    d = _session_to_dict(s)
    assert d["access_token"] == "A"
    assert d["refresh_token"] == "R"
    assert d["expires_at"] == 123
    assert d["user"]["id"] == "u-1"


def test_session_to_dict_none():
    assert _session_to_dict(None) is None


def test_wrap_builds_expected_shape():
    resp = _fake_auth_response(user=_fake_user(), session=_fake_session())
    d = _wrap(resp)
    assert d["user"]["id"] == "u-1"
    assert d["session"]["access_token"] == "access-token"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_constructor_requires_url_and_key():
    with pytest.raises(ValueError):
        SupabaseAuthBackend("", "")
    with pytest.raises(ValueError):
        SupabaseAuthBackend("https://x.supabase.co", "")


def test_constructor_calls_create_client():
    fake = MagicMock()
    with patch("src.auth.supabase_backend.create_client",
               return_value=fake) as mock_create:
        SupabaseAuthBackend("https://x.supabase.co", "anon")
    mock_create.assert_called_once_with("https://x.supabase.co", "anon")


# ---------------------------------------------------------------------------
# sign_up
# ---------------------------------------------------------------------------

def test_sign_up_passes_metadata_via_options():
    backend = _build_backend_with_fake_client()
    backend.client.auth.sign_up.return_value = _fake_auth_response(
        user=_fake_user(), session=_fake_session(),
    )
    result = backend.sign_up("a@b.com", "pw", {"full_name": "Alice"})

    call_args = backend.client.auth.sign_up.call_args[0][0]
    assert call_args["email"] == "a@b.com"
    assert call_args["password"] == "pw"
    assert call_args["options"]["data"] == {"full_name": "Alice"}

    assert result["user"]["id"] == "u-1"
    assert result["session"]["access_token"] == "access-token"


def test_sign_up_returns_empty_session_when_confirmation_required():
    backend = _build_backend_with_fake_client()
    backend.client.auth.sign_up.return_value = _fake_auth_response(
        user=_fake_user(), session=None,
    )
    result = backend.sign_up("a@b.com", "pw", {})
    assert result["user"]["id"] == "u-1"
    assert result["session"] is None


# ---------------------------------------------------------------------------
# sign_in
# ---------------------------------------------------------------------------

def test_sign_in_passes_credentials():
    backend = _build_backend_with_fake_client()
    backend.client.auth.sign_in_with_password.return_value = _fake_auth_response(
        user=_fake_user(), session=_fake_session(),
    )
    backend.sign_in("a@b.com", "pw")
    call_args = backend.client.auth.sign_in_with_password.call_args[0][0]
    assert call_args == {"email": "a@b.com", "password": "pw"}


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------

def test_sign_in_with_google_returns_url():
    backend = _build_backend_with_fake_client()
    backend.client.auth.sign_in_with_oauth.return_value = SimpleNamespace(
        url="https://accounts.google.com/o/oauth2/...",
    )
    result = backend.sign_in_with_oauth("google", "https://app/callback")
    assert result["url"].startswith("https://accounts.google.com")


def test_sign_in_with_google_passes_provider_and_redirect():
    backend = _build_backend_with_fake_client()
    backend.client.auth.sign_in_with_oauth.return_value = SimpleNamespace(
        url="https://x",
    )
    backend.sign_in_with_oauth("google", "https://app/callback")
    call_args = backend.client.auth.sign_in_with_oauth.call_args[0][0]
    assert call_args["provider"] == "google"
    assert call_args["options"]["redirect_to"] == "https://app/callback"


# ---------------------------------------------------------------------------
# exchange_code_for_session
# ---------------------------------------------------------------------------

def test_exchange_code_uses_new_sdk_path():
    backend = _build_backend_with_fake_client()
    backend.client.auth.exchange_code_for_session = MagicMock(
        return_value=_fake_auth_response(
            user=_fake_user(), session=_fake_session(),
        )
    )
    result = backend.exchange_code_for_session("abc")
    assert result["session"]["access_token"] == "access-token"


def test_exchange_code_raises_if_sdk_lacks_method():
    backend = _build_backend_with_fake_client()
    # Simulate old SDK without the method
    del backend.client.auth.exchange_code_for_session
    with pytest.raises(NotImplementedError):
        backend.exchange_code_for_session("abc")


# ---------------------------------------------------------------------------
# sign_out / refresh
# ---------------------------------------------------------------------------

def test_sign_out_never_raises():
    backend = _build_backend_with_fake_client()
    backend.client.auth.sign_out.side_effect = RuntimeError("boom")
    backend.sign_out("access-token")   # must not raise


def test_refresh_passes_token():
    backend = _build_backend_with_fake_client()
    backend.client.auth.refresh_session.return_value = _fake_auth_response(
        user=_fake_user(), session=_fake_session(access="new-access"),
    )
    result = backend.refresh_session("refresh-token")
    assert result["session"]["access_token"] == "new-access"
    backend.client.auth.refresh_session.assert_called_once_with("refresh-token")


# ---------------------------------------------------------------------------
# table reads
# ---------------------------------------------------------------------------

def test_select_user_queries_users_table():
    backend = _build_backend_with_fake_client()
    select_chain = MagicMock()
    backend.client.table.return_value = select_chain
    select_chain.select.return_value = select_chain
    select_chain.eq.return_value = select_chain
    select_chain.limit.return_value = select_chain
    select_chain.execute.return_value = SimpleNamespace(
        data=[{"id": "u-1", "email": "a@b.com", "full_name": "A"}]
    )

    row = backend.select_user("u-1")
    assert row["email"] == "a@b.com"
    backend.client.table.assert_called_with("users")


def test_select_user_returns_none_when_empty():
    backend = _build_backend_with_fake_client()
    select_chain = MagicMock()
    backend.client.table.return_value = select_chain
    select_chain.select.return_value = select_chain
    select_chain.eq.return_value = select_chain
    select_chain.limit.return_value = select_chain
    select_chain.execute.return_value = SimpleNamespace(data=[])
    assert backend.select_user("u-1") is None


def test_select_clients_for_user_flattens_nested_client():
    backend = _build_backend_with_fake_client()
    select_chain = MagicMock()
    backend.client.table.return_value = select_chain
    select_chain.select.return_value = select_chain
    select_chain.eq.return_value = select_chain
    select_chain.execute.return_value = SimpleNamespace(data=[
        {
            "client_id": "c-1", "role": "owner",
            "clients": {"id": "c-1", "name": "Acme", "slug": "acme"},
        },
        {
            "client_id": "c-2", "role": "viewer",
            "clients": {"id": "c-2", "name": "Beta", "slug": "beta"},
        },
    ])
    rows = backend.select_clients_for_user("u-1")
    assert len(rows) == 2
    assert {r["slug"] for r in rows} == {"acme", "beta"}


# ---------------------------------------------------------------------------
# set_user_session — the RLS-critical call
# ---------------------------------------------------------------------------

def test_set_user_session_calls_sdk():
    backend = _build_backend_with_fake_client()
    backend.set_user_session("access-token", "refresh-token")
    backend.client.auth.set_session.assert_called_once_with(
        "access-token", "refresh-token"
    )


def test_set_user_session_rejects_empty_token():
    backend = _build_backend_with_fake_client()
    with pytest.raises(ValueError):
        backend.set_user_session("", "")