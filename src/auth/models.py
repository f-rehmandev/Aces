"""
Auth domain models — spec §43.

Our own representation of users, clients, and memberships. We do NOT
store credentials — that's Supabase Auth's job. These tables exist to
answer: "who is this, and which tenants do they belong to?"
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Role — must match the SQL enum in schema_auth.sql
# ---------------------------------------------------------------------------

class Role(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

    @staticmethod
    def can_write(role: "Role") -> bool:
        return role in (Role.OWNER, Role.ADMIN, Role.MEMBER)

    @staticmethod
    def can_administer(role: "Role") -> bool:
        return role in (Role.OWNER, Role.ADMIN)


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

@dataclass
class User:
    id: str                                     # UUID, matches auth.users.id
    email: Optional[str] = None
    full_name: str = ""
    avatar_url: Optional[str] = None
    default_client_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Client (tenant)
# ---------------------------------------------------------------------------

@dataclass
class Client:
    id: str
    name: str
    slug: str
    created_at: str = ""
    created_by: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------

@dataclass
class ClientMember:
    client_id: str
    user_id: str
    role: Role = Role.MEMBER
    invited_by: Optional[str] = None
    joined_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d


# ---------------------------------------------------------------------------
# Session — the human-readable view of an active auth session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    The active session for a signed-in user.

    `access_token` and `refresh_token` are Supabase-issued JWTs. We pass
    them through to Supabase client calls but never inspect or forge them.
    """
    user_id: str
    access_token: str
    refresh_token: str = ""
    expires_at: Optional[int] = None            # unix epoch seconds
    email: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        return now >= self.expires_at

    def to_dict(self) -> dict:
        d = asdict(self)
        # Never expose tokens in a serialization meant for the UI/logs
        d.pop("access_token", None)
        d.pop("refresh_token", None)
        return d


# ---------------------------------------------------------------------------
# Sign-up / sign-in inputs
# ---------------------------------------------------------------------------

@dataclass
class SignUpRequest:
    email: str
    password: str
    full_name: str = ""


@dataclass
class SignInRequest:
    email: str
    password: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Base class for all authentication errors raised by ACES."""


class InvalidCredentials(AuthError):
    """Wrong email or password."""


class EmailNotVerified(AuthError):
    """Supabase requires email confirmation before sign-in."""


class UserAlreadyExists(AuthError):
    """Sign-up attempted with an already-registered email."""


class SessionExpired(AuthError):
    """Access token expired and refresh failed."""


class PermissionDenied(AuthError):
    """User is not a member of the requested client, or lacks the role."""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    u = User(id="u-1", email="a@b.com", full_name="Alice")
    assert u.to_dict()["email"] == "a@b.com"

    c = Client(id="c-1", name="Acme", slug="acme")
    assert c.to_dict()["slug"] == "acme"

    m = ClientMember(client_id="c-1", user_id="u-1", role=Role.OWNER)
    assert m.to_dict()["role"] == "owner"

    s = Session(
        user_id="u-1", access_token="secret.jwt.token",
        refresh_token="refresh.secret", expires_at=10_000_000_000,
    )
    assert not s.is_expired(now=1_000)
    assert s.is_expired(now=11_000_000_000)
    # Tokens never leak out of to_dict
    d = s.to_dict()
    assert "access_token" not in d
    assert "refresh_token" not in d

    # Role helpers
    assert Role.can_write(Role.OWNER)
    assert Role.can_write(Role.MEMBER)
    assert not Role.can_write(Role.VIEWER)
    assert Role.can_administer(Role.ADMIN)
    assert not Role.can_administer(Role.MEMBER)

    print("Auth models OK.")