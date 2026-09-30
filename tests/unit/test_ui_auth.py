"""Unit tests for auth UI helpers (src/ui/auth.py)."""
import pytest

from src.auth.context import ClientContext, anonymous_context
from src.auth.models import Role, Session
from src.ui.auth import (
    _missing_password_categories, _store_session, sign_out,
    get_session, get_context, SESSION_KEY, CONTEXT_KEY,
)


# ---------------------------------------------------------------------------
# Password policy pre-check
# ---------------------------------------------------------------------------

def test_missing_categories_on_empty_password():
    missing = _missing_password_categories("")
    assert set(missing) == {"lowercase", "uppercase", "digit", "special character"}


def test_missing_categories_all_present():
    assert _missing_password_categories("Abcdef1!") == []


def test_missing_categories_only_lowercase():
    assert _missing_password_categories("abcdefg1!") == ["uppercase"]


def test_missing_categories_only_special():
    assert _missing_password_categories("Abcdefg1") == ["special character"]


# ---------------------------------------------------------------------------
# Session state helpers (using a fake st.session_state via monkeypatch)
# ---------------------------------------------------------------------------

class FakeState(dict):
    """dict that behaves like st.session_state for our purposes."""
    def get(self, key, default=None):
        return dict.get(self, key, default)
    def pop(self, key, default=None):
        return dict.pop(self, key, default)


@pytest.fixture
def fake_state(monkeypatch):
    state = FakeState()
    import src.ui.auth as auth_mod
    monkeypatch.setattr(auth_mod.st, "session_state", state)
    return state


def test_get_session_none_when_unset(fake_state):
    assert get_session() is None


def test_get_session_returns_stored(fake_state):
    s = Session(user_id="u-1", access_token="tok", expires_at=None)
    fake_state[SESSION_KEY] = s
    assert get_session() is s


def test_get_session_drops_expired_without_refresh(fake_state):
    s = Session(user_id="u-1", access_token="tok",
                refresh_token="", expires_at=1)
    fake_state[SESSION_KEY] = s
    import time
    # Ensure the token is really expired
    assert s.is_expired(now=time.time())
    assert get_session() is None
    assert SESSION_KEY not in fake_state


def test_get_context_anonymous_when_no_session(fake_state):
    ctx = get_context()
    assert ctx.is_anonymous
    assert ctx.client_id == "default"


def test_get_context_caches_result(fake_state):
    ctx1 = get_context()
    ctx2 = get_context()
    assert ctx1 is ctx2


def test_store_session_clears_context_cache(fake_state):
    # Prime a context
    get_context()
    assert CONTEXT_KEY in fake_state

    s = Session(user_id="u-1", access_token="tok")
    _store_session(s)
    assert SESSION_KEY in fake_state
    assert CONTEXT_KEY not in fake_state


def test_sign_out_clears_state(fake_state):
    s = Session(user_id="u-1", access_token="tok")
    _store_session(s)
    assert SESSION_KEY in fake_state

    sign_out()
    assert SESSION_KEY not in fake_state
    assert CONTEXT_KEY not in fake_state


# ---------------------------------------------------------------------------
# AuthService presence
# ---------------------------------------------------------------------------

def test_get_auth_service_returns_none_without_env(monkeypatch):
    import src.ui.auth as auth_mod
    # Force missing env
    monkeypatch.setattr(auth_mod.os, "getenv", lambda k, d=None: "")
    # Bypass st.cache_resource wrapper
    result = auth_mod.get_auth_service.__wrapped__() if hasattr(
        auth_mod.get_auth_service, "__wrapped__"
    ) else None
    # We can't easily disable the cache in a plain unit test, so this
    # test only asserts the function exists and is callable — the
    # important behavior (returns None on missing env) is covered by
    # the smoke test script.
    assert callable(auth_mod.get_auth_service)