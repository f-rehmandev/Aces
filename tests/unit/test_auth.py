"""Unit tests for the auth service (spec §43)."""
import pytest

from src.auth.models import (
    AuthError, Client, ClientMember, EmailNotVerified, InvalidCredentials,
    PermissionDenied, Role, Session, SignInRequest, SignUpRequest,
    User, UserAlreadyExists,
)
from src.auth.service import AuthService, SignUpOutcome


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------

class FakeBackend:
    def __init__(self):
        self.users: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.require_email_confirmation = False
        self.raise_on_sign_in: Exception | None = None
        self.raise_on_sign_up: Exception | None = None
        self.raise_on_refresh: Exception | None = None
        self._next = 1

    def sign_up(self, email, password, metadata):
        if self.raise_on_sign_up:
            raise self.raise_on_sign_up
        if email in self.users:
            raise RuntimeError("User already registered")
        uid = f"u-{self._next}"
        self._next += 1
        self.users[email] = {"id": uid, "email": email,
                              "password": password, **metadata} 
        
        if self.require_email_confirmation:
            return {"session": None, "user": {"id": uid, "email": email}}
        session = {
            "access_token": f"access-{uid}",
            "refresh_token": f"refresh-{uid}",
            "expires_at": 9_999_999_999,
            "user": {"id": uid, "email": email},
        }
        return {"session": session, "user": session["user"]}

    def sign_in(self, email, password):
        if self.raise_on_sign_in:
            raise self.raise_on_sign_in
        user = self.users.get(email)
        if not user or user.get("password") != password:
            raise RuntimeError("Invalid login credentials")
        session = {
            "access_token": f"access-{user['id']}",
            "refresh_token": f"refresh-{user['id']}",
            "expires_at": 9_999_999_999,
            "user": {"id": user["id"], "email": email},
        }
        return {"session": session, "user": session["user"]}

    def sign_in_with_oauth(self, provider, redirect_to):
        return {"url": f"https://oauth.example/{provider}?r={redirect_to}"}

    def exchange_code_for_session(self, code):
        return {"session": {
            "access_token": "oauth-access", "refresh_token": "oauth-refresh",
            "user": {"id": "u-oauth", "email": "g@example.com"},
        }}

    def sign_out(self, access_token):
        pass

    def refresh_session(self, refresh_token):
        if self.raise_on_refresh:
            raise self.raise_on_refresh
        return {"session": {
            "access_token": "new-access",
            "refresh_token": refresh_token,
            "user": {"id": "u-1", "email": "a@b.com"},
        }}

    def select_user(self, user_id):
        for u in self.users.values():
            if u["id"] == user_id:
                return {"id": u["id"], "email": u["email"],
                        "full_name": u.get("full_name", "")}
        return None

    def select_clients_for_user(self, user_id):
        return [
            {"client_id": "c-1", "name": "Acme", "slug": "acme", "role": "owner"},
            {"client_id": "c-2", "name": "Beta", "slug": "beta", "role": "viewer"},
        ]


@pytest.fixture
def svc():
    return AuthService(backend=FakeBackend())


# ---------------------------------------------------------------------------
# Sign-up
# ---------------------------------------------------------------------------

def test_sign_up_returns_session(svc):
    out = svc.sign_up(SignUpRequest(email="a@b.com", password="pw",
                                     full_name="Alice"))
    assert isinstance(out, SignUpOutcome)
    assert out.session is not None
    assert out.session.email == "a@b.com"
    assert not out.needs_email_confirmation


def test_sign_up_with_email_confirmation():
    b = FakeBackend()
    b.require_email_confirmation = True
    svc = AuthService(backend=b)
    out = svc.sign_up(SignUpRequest(email="a@b.com", password="pw"))
    assert out.session is None
    assert out.needs_email_confirmation is True


def test_sign_up_duplicate_email(svc):
    svc.sign_up(SignUpRequest(email="a@b.com", password="pw"))
    with pytest.raises(UserAlreadyExists):
        svc.sign_up(SignUpRequest(email="a@b.com", password="pw"))


def test_sign_up_unknown_error_wrapped(svc):
    svc.backend.raise_on_sign_up = RuntimeError("boom")
    with pytest.raises(AuthError):
        svc.sign_up(SignUpRequest(email="a@b.com", password="pw"))


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------

def test_sign_in_valid(svc):
    svc.sign_up(SignUpRequest(email="a@b.com", password="pw"))
    session = svc.sign_in(SignInRequest(email="a@b.com", password="pw"))
    assert session.user_id.startswith("u-")
    assert session.access_token.startswith("access-")


def test_sign_in_invalid_password(svc):
    svc.sign_up(SignUpRequest(email="a@b.com", password="pw"))
    with pytest.raises(InvalidCredentials):
        svc.sign_in(SignInRequest(email="a@b.com", password="wrong"))


def test_sign_in_unverified_email():
    b = FakeBackend()
    svc = AuthService(backend=b)

    def raise_unverified(*a, **k):
        raise RuntimeError("Email not confirmed")
    b.sign_in = raise_unverified

    with pytest.raises(EmailNotVerified):
        svc.sign_in(SignInRequest(email="a@b.com", password="pw"))


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------

def test_google_oauth_returns_url(svc):
    url = svc.sign_in_with_google("https://app.example/callback")
    assert url.startswith("https://oauth.example/google")
    assert "callback" in url


def test_google_oauth_requires_redirect(svc):
    with pytest.raises(AuthError):
        svc.sign_in_with_google("")


def test_exchange_code_returns_session(svc):
    session = svc.exchange_code_for_session("auth-code-123")
    assert session.email == "g@example.com"
    assert session.user_id == "u-oauth"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_sign_out_never_raises():
    b = FakeBackend()
    svc = AuthService(backend=b)

    def raise_now(*a, **k):
        raise RuntimeError("network down")
    b.sign_out = raise_now

    session = Session(user_id="u-1", access_token="tok")
    svc.sign_out(session)  # must not raise


def test_refresh_returns_new_session(svc):
    session = Session(user_id="u-1", access_token="old",
                      refresh_token="refresh-u-1")
    new_session = svc.refresh(session)
    assert new_session.access_token == "new-access"


def test_refresh_without_token_raises(svc):
    session = Session(user_id="u-1", access_token="old")
    with pytest.raises(AuthError):
        svc.refresh(session)


# ---------------------------------------------------------------------------
# Profile + clients
# ---------------------------------------------------------------------------

def test_get_profile(svc):
    out = svc.sign_up(SignUpRequest(email="a@b.com", password="pw",
                                     full_name="Alice"))
    profile = svc.get_profile(out.session)
    assert profile is not None
    assert profile.email == "a@b.com"
    assert profile.full_name == "Alice"


def test_get_profile_unknown_user_returns_none(svc):
    session = Session(user_id="u-does-not-exist", access_token="tok")
    assert svc.get_profile(session) is None


def test_list_clients(svc):
    session = Session(user_id="u-1", access_token="tok")
    clients = svc.list_clients(session)
    assert len(clients) == 2
    slugs = {c.slug for c, _ in clients}
    assert slugs == {"acme", "beta"}


def test_require_role_owner_ok(svc):
    session = Session(user_id="u-1", access_token="tok")
    svc.require_role(session, "c-1", Role.OWNER)


def test_require_role_denied(svc):
    session = Session(user_id="u-1", access_token="tok")
    with pytest.raises(PermissionDenied):
        svc.require_role(session, "c-1", Role.VIEWER)


def test_require_role_not_a_member(svc):
    session = Session(user_id="u-1", access_token="tok")
    with pytest.raises(PermissionDenied):
        svc.require_role(session, "c-nope", Role.OWNER)


def test_require_role_accepts_list(svc):
    session = Session(user_id="u-1", access_token="tok")
    # Beta client role is viewer
    svc.require_role(session, "c-2", [Role.VIEWER, Role.MEMBER])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_session_serialization_hides_tokens():
    s = Session(user_id="u-1", access_token="secret", refresh_token="r")
    d = s.to_dict()
    assert "access_token" not in d
    assert "refresh_token" not in d


def test_session_expiry():
    s = Session(user_id="u-1", access_token="tok", expires_at=1_000)
    assert not s.is_expired(now=500)
    assert s.is_expired(now=1_000)
    assert s.is_expired(now=5_000)


def test_client_member_serialization():
    m = ClientMember(client_id="c-1", user_id="u-1", role=Role.ADMIN)
    d = m.to_dict()
    assert d["role"] == "admin"


def test_role_helpers():
    assert Role.can_write(Role.OWNER)
    assert Role.can_write(Role.ADMIN)
    assert Role.can_write(Role.MEMBER)
    assert not Role.can_write(Role.VIEWER)
    assert Role.can_administer(Role.OWNER)
    assert not Role.can_administer(Role.MEMBER)