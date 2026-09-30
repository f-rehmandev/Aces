"""
Auth service — spec §43.

Wraps Supabase Auth behind a small, testable interface:

    sign_up(email, password, full_name) -> Session | needs verification
    sign_in(email, password)            -> Session
    sign_in_with_google(redirect_to)    -> OAuth URL to open
    exchange_code_for_session(code)     -> Session
    sign_out(session)                   -> None
    refresh(session)                    -> Session
    get_profile(session)                -> User
    list_clients(session)               -> list[(Client, Role)]

The `SupabaseLike` protocol lets tests inject a fake without touching
the network. The real implementation uses `supabase-py`'s `.auth` API.

Password handling: Supabase hashes and stores credentials. We never see
them again after the sign-in call. This is deliberate.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from src.auth.models import (
    AuthError, Client, ClientMember, EmailNotVerified, InvalidCredentials,
    PermissionDenied, Role, Session, SignInRequest, SignUpRequest, User,
    UserAlreadyExists,
)


# ---------------------------------------------------------------------------
# Protocol — anything that quacks like supabase-py's client
# ---------------------------------------------------------------------------

class AuthBackend(Protocol):
    """
    Minimal shape we need from the backend. Matches the subset of
    `supabase.auth` and `.table(...)` we use.

    Fakes in tests only need to implement these methods.
    """

    def sign_up(self, email: str, password: str, metadata: dict) -> dict: ...
    def sign_in(self, email: str, password: str) -> dict: ...
    def sign_in_with_oauth(self, provider: str, redirect_to: str) -> dict: ...
    def exchange_code_for_session(self, code: str) -> dict: ...
    def sign_out(self, access_token: str) -> None: ...
    def refresh_session(self, refresh_token: str) -> dict: ...
    def select_user(self, user_id: str) -> Optional[dict]: ...
    def select_clients_for_user(self, user_id: str) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@dataclass
class SignUpOutcome:
    """
    Sign-up has two valid outcomes:
        - session is set  -> user is signed in immediately
        - session is None -> email confirmation required before sign-in
    """
    session: Optional[Session]
    user_id: str
    needs_email_confirmation: bool = False


class AuthService:
    def __init__(self, backend: AuthBackend):
        self.backend = backend

    # ------------------------------------------------------------------
    # Email / password
    # ------------------------------------------------------------------
    def sign_up(self, req: SignUpRequest) -> SignUpOutcome:
        metadata = {"full_name": req.full_name} if req.full_name else {}
        try:
            result = self.backend.sign_up(req.email, req.password, metadata)
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg or "exists" in msg:
                raise UserAlreadyExists(str(e)) from e
            raise AuthError(str(e)) from e

        session = self._session_from_backend(result)
        user_id = self._user_id_from_backend(result)
        if session is None:
            return SignUpOutcome(
                session=None, user_id=user_id,
                needs_email_confirmation=True,
            )
        return SignUpOutcome(session=session, user_id=user_id)

    def sign_in(self, req: SignInRequest) -> Session:
        try:
            result = self.backend.sign_in(req.email, req.password)
        except Exception as e:
            msg = str(e).lower()
            if "email" in msg and ("confirm" in msg or "verif" in msg):
                raise EmailNotVerified(str(e)) from e
            raise InvalidCredentials("invalid email or password") from e

        session = self._session_from_backend(result)
        if session is None:
            # Backend returned no tokens but no exception — treat as error
            raise AuthError("sign-in returned no session")
        return session

    # ------------------------------------------------------------------
    # Google OAuth
    # ------------------------------------------------------------------
    def sign_in_with_google(self, redirect_to: str) -> str:
        """
        Returns the URL the caller should redirect the browser to.
        Callback handling uses `exchange_code_for_session`.
        """
        if not redirect_to:
            raise AuthError("redirect_to is required for OAuth")
        try:
            result = self.backend.sign_in_with_oauth("google", redirect_to)
        except Exception as e:
            raise AuthError(str(e)) from e
        url = result.get("url") if isinstance(result, dict) else None
        if not url:
            raise AuthError("OAuth backend did not return a redirect URL")
        return url

    def exchange_code_for_session(self, code: str) -> Session:
        try:
            result = self.backend.exchange_code_for_session(code)
        except Exception as e:
            raise AuthError(str(e)) from e
        session = self._session_from_backend(result)
        if session is None:
            raise AuthError("OAuth exchange returned no session")
        return session

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def sign_out(self, session: Session) -> None:
        try:
            self.backend.sign_out(session.access_token)
        except Exception:
            # Best-effort — sign-out must never raise to the caller.
            pass

    def refresh(self, session: Session) -> Session:
        if not session.refresh_token:
            raise AuthError("cannot refresh without a refresh token")
        try:
            result = self.backend.refresh_session(session.refresh_token)
        except Exception as e:
            raise AuthError(str(e)) from e
        new_session = self._session_from_backend(result)
        if new_session is None:
            raise AuthError("refresh returned no session")
        return new_session

    # ------------------------------------------------------------------
    # Profile / membership
    # ------------------------------------------------------------------
    def get_profile(self, session: Session) -> Optional[User]:
        row = self.backend.select_user(session.user_id)
        if not row:
            return None
        return User(
            id=str(row.get("id")),
            email=row.get("email"),
            full_name=row.get("full_name") or "",
            avatar_url=row.get("avatar_url"),
            default_client_id=row.get("default_client_id"),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
            metadata=row.get("metadata") or {},
        )

    def list_clients(self, session: Session) -> list[tuple[Client, Role]]:
        rows = self.backend.select_clients_for_user(session.user_id)
        out: list[tuple[Client, Role]] = []
        for row in rows:
            client = Client(
                id=str(row.get("client_id") or row.get("id")),
                name=row.get("name", ""),
                slug=row.get("slug", ""),
                created_at=str(row.get("created_at") or ""),
                created_by=row.get("created_by"),
                metadata=row.get("metadata") or {},
            )
            try:
                role = Role(row.get("role", "member"))
            except ValueError:
                role = Role.VIEWER
            out.append((client, role))
        return out

    def require_role(
        self,
        session: Session,
        client_id: str,
        required: Role | list[Role],
    ) -> None:
        """
        Raise PermissionDenied if the session user is not a member of
        `client_id` with one of `required` roles. Callers use this to
        gate operations that RLS enforces at the database level — it's a
        defensive early check, not the authoritative gate.
        """
        memberships = dict(
            (c.id, r) for (c, r) in self.list_clients(session)
        )
        if client_id not in memberships:
            raise PermissionDenied(f"not a member of client {client_id}")
        role = memberships[client_id]
        wanted = [required] if isinstance(required, Role) else list(required)
        if role not in wanted:
            raise PermissionDenied(
                f"role {role.value} cannot perform this action"
            )

    # ------------------------------------------------------------------
    # Backend → our model
    # ------------------------------------------------------------------
    @staticmethod
    def _session_from_backend(result: Any) -> Optional[Session]:
        """
        Accepts either `{"session": {...}, "user": {...}}` (supabase-py
        sign-in shape) or a raw session dict.
        """
        if not isinstance(result, dict):
            return None
        session = result.get("session") or result
        if not isinstance(session, dict):
            return None
        access = session.get("access_token")
        if not access:
            return None
        user = result.get("user") or session.get("user") or {}
        return Session(
            user_id=str(user.get("id") or session.get("user_id") or ""),
            access_token=access,
            refresh_token=session.get("refresh_token", "") or "",
            expires_at=session.get("expires_at"),
            email=user.get("email"),
            metadata=user.get("user_metadata") or {},
        )

    @staticmethod
    def _user_id_from_backend(result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        user = result.get("user") or {}
        return str(user.get("id") or "")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def make_supabase_backend(url: str, anon_key: str) -> AuthBackend:
    """
    Build the real Supabase-backed AuthBackend. Returns an object that
    implements the AuthBackend protocol.

    Lazy-imports supabase so tests don't need it installed.
    """
    from src.auth.supabase_backend import SupabaseAuthBackend
    return SupabaseAuthBackend(url, anon_key)


# ---------------------------------------------------------------------------
# Smoke test — fake backend, no network
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class FakeBackend:
        def __init__(self):
            self.users = {}
            self.sessions = {}
            self._next = 1

        def sign_up(self, email, password, metadata):
            if email in self.users:
                raise RuntimeError("User already registered")
            uid = f"u-{self._next}"
            self._next += 1
            self.users[email] = {"id": uid, "email": email,
                                  "password": password, **metadata}
            session = {
                "access_token": f"access-{uid}",
                "refresh_token": f"refresh-{uid}",
                "expires_at": 9_999_999_999,
                "user": {"id": uid, "email": email},
            }
            return {"session": session, "user": session["user"]}

        def sign_in(self, email, password):
            user = self.users.get(email)
            if not user or user.get("password") != password:
                raise RuntimeError("Invalid login credentials")
            session = {
                "access_token": f"access-{user['id']}",
                "refresh_token": f"refresh-{user['id']}",
                "user": {"id": user["id"], "email": email},
            }
            return {"session": session, "user": session["user"]}

        def sign_in_with_oauth(self, provider, redirect_to):
            return {"url": f"https://accounts.google.com/o/oauth2/...?r={redirect_to}"}

        def exchange_code_for_session(self, code):
            return {"session": {
                "access_token": "oauth-token", "refresh_token": "oauth-refresh",
                "user": {"id": "u-oauth", "email": "g@example.com"},
            }}

        def sign_out(self, access_token): pass

        def refresh_session(self, refresh_token):
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
            return [{
                "client_id": "c-1", "name": "Acme", "slug": "acme",
                "role": "owner",
            }]

    svc = AuthService(backend=FakeBackend())

    out = svc.sign_up(SignUpRequest(email="a@b.com", password="x",
                                     full_name="Alice"))
    assert out.session is not None
    assert out.session.email == "a@b.com"
    assert not out.needs_email_confirmation

    # Duplicate sign-up
    try:
        svc.sign_up(SignUpRequest(email="a@b.com", password="x"))
        raise AssertionError("expected UserAlreadyExists")
    except UserAlreadyExists:
        pass

    # Sign-in
    session = svc.sign_in(SignInRequest(email="a@b.com", password="x"))
    assert session.user_id.startswith("u-")

    # Wrong password
    try:
        svc.sign_in(SignInRequest(email="a@b.com", password="wrong"))
        raise AssertionError("expected InvalidCredentials")
    except InvalidCredentials:
        pass

    # Google OAuth URL
    url = svc.sign_in_with_google("https://app.example/callback")
    assert url.startswith("https://accounts.google.com")

    # OAuth exchange
    s2 = svc.exchange_code_for_session("abc")
    assert s2.email == "g@example.com"

    # Refresh
    s3 = svc.refresh(session)
    assert s3.access_token == "new-access"

    # Profile
    profile = svc.get_profile(session)
    assert profile is not None and profile.email == "a@b.com"

    # Clients + role gate
    clients = svc.list_clients(session)
    assert len(clients) == 1
    client, role = clients[0]
    assert client.slug == "acme" and role == Role.OWNER

    svc.require_role(session, client.id, Role.OWNER)  # ok
    try:
        svc.require_role(session, client.id, Role.VIEWER)
        raise AssertionError("expected PermissionDenied")
    except PermissionDenied:
        pass
    try:
        svc.require_role(session, "c-nonexistent", Role.OWNER)
        raise AssertionError("expected PermissionDenied")
    except PermissionDenied:
        pass

    print("Auth service OK.")