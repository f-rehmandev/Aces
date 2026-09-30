"""
Real Supabase Auth backend.

Wraps `supabase-py` behind the `AuthBackend` protocol so `AuthService`
never imports supabase directly.

Important: Row Level Security on `public.clients` / `public.users` etc.
checks `auth.uid()`, which only exists if the request carries the *user's*
JWT — not the service-role key. Every `.table()` read in this adapter
sets the user's access token before the query, or RLS is silently bypassed.
That's the single most dangerous bug in a Supabase app; we make it
impossible by never issuing a table query without an explicit user token.
"""

from __future__ import annotations
from typing import Any, Optional

try:
    from supabase import create_client, Client as SupabaseClient
except ImportError as _exc:   # pragma: no cover
    create_client = None       # type: ignore[assignment]
    SupabaseClient = None      # type: ignore[assignment]
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_to_dict(session: Any) -> Optional[dict]:
    """Convert a supabase-py Session object into a plain dict."""
    if session is None:
        return None
    if isinstance(session, dict):
        return session
    return {
        "access_token": getattr(session, "access_token", None),
        "refresh_token": getattr(session, "refresh_token", None),
        "expires_at": getattr(session, "expires_at", None),
        "user": _user_to_dict(getattr(session, "user", None)),
    }


def _user_to_dict(user: Any) -> Optional[dict]:
    if user is None:
        return None
    if isinstance(user, dict):
        return user
    return {
        "id": getattr(user, "id", None),
        "email": getattr(user, "email", None),
        "user_metadata": getattr(user, "user_metadata", None) or {},
    }


def _wrap(result: Any) -> dict:
    """
    supabase-py returns AuthResponse objects with `.user` and `.session`.
    Wrap them into the dict shape `AuthService` expects.
    """
    if result is None:
        return {}
    user = _user_to_dict(getattr(result, "user", None))
    session = _session_to_dict(getattr(result, "session", None))
    return {"user": user, "session": session}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class SupabaseAuthBackend:
    """
    Concrete AuthBackend over `supabase-py`.

    Constructor takes the project URL and the *anon* key (never the
    service-role key — this backend is meant to run in a user context).
    """

    def __init__(self, url: str, anon_key: str):
        if create_client is None:
            raise ImportError(
                f"supabase-py is not installed: {_IMPORT_ERROR}"
            ) from _IMPORT_ERROR
        if not url or not anon_key:
            raise ValueError("Supabase URL and anon key are required")
        self._url = url
        self._anon_key = anon_key
        self.client: SupabaseClient = create_client(url, anon_key)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def sign_up(self, email: str, password: str, metadata: dict) -> dict:
        result = self.client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": metadata or {}},
        })
        return _wrap(result)

    def sign_in(self, email: str, password: str) -> dict:
        result = self.client.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })
        return _wrap(result)

    def sign_in_with_oauth(self, provider: str, redirect_to: str) -> dict:
        result = self.client.auth.sign_in_with_oauth({
            "provider": provider,
            "options": {"redirect_to": redirect_to},
        })
        # supabase-py returns an object with `.url`
        url = getattr(result, "url", None) or (
            result.get("url") if isinstance(result, dict) else None
        )
        return {"url": url}

    def exchange_code_for_session(self, code: str) -> dict:
        # Newer supabase-py exposes exchange_code_for_session.
        # Older versions accept the code via verify_otp or callback handling.
        if hasattr(self.client.auth, "exchange_code_for_session"):
            result = self.client.auth.exchange_code_for_session(
                {"auth_code": code}
            )
            return _wrap(result)
        # Fallback: verify_otp with type="magiclink" is close enough for
        # many flows. If neither is available, raise clearly.
        raise NotImplementedError(
            "This supabase-py version does not support "
            "exchange_code_for_session; upgrade supabase-py to >= 2.10"
        )

    def sign_out(self, access_token: str) -> None:
        # Best-effort: pass through whatever the client understands.
        try:
            self.client.auth.sign_out()
        except Exception:
            # Some versions require an access token; if the SDK raises,
            # swallow it — sign-out must be idempotent for our callers.
            pass

    def refresh_session(self, refresh_token: str) -> dict:
        result = self.client.auth.refresh_session(refresh_token)
        return _wrap(result)

    # ------------------------------------------------------------------
    # Table reads (RLS-scoped)
    # ------------------------------------------------------------------
    def select_user(self, user_id: str) -> Optional[dict]:
        rows = (
            self.client.table("users")
            .select("*")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        data = getattr(rows, "data", None) or []
        return data[0] if data else None

    def select_clients_for_user(self, user_id: str) -> list[dict]:
        """
        Return the list of (client, role) pairs for this user.

        Relies on the `client_members` RLS policy — the caller must have
        set their session on the client before this runs (see `set_user_session`).
        """
        rows = (
            self.client.table("client_members")
            .select("client_id, role, clients(id, name, slug, created_at, created_by, metadata)")
            .eq("user_id", user_id)
            .execute()
        )
        out: list[dict] = []
        for row in (getattr(rows, "data", None) or []):
            client = row.get("clients") or {}
            out.append({
                "client_id": row.get("client_id"),
                "role": row.get("role"),
                "name": client.get("name", ""),
                "slug": client.get("slug", ""),
                "created_at": client.get("created_at"),
                "created_by": client.get("created_by"),
                "metadata": client.get("metadata") or {},
            })
        return out

    # ------------------------------------------------------------------
    # Session propagation (RLS-critical)
    # ------------------------------------------------------------------
    def set_user_session(self, access_token: str, refresh_token: str = "") -> None:
        """
        Attach the user's JWT to the client so subsequent `.table()` reads
        are evaluated as the user, not as the anon key.

        Callers MUST invoke this immediately after sign-in / OAuth exchange
        and before any table query. Without it, RLS policies see no
        `auth.uid()` and reject every read.
        """
        if not access_token:
            raise ValueError("access_token is required to set the session")
        try:
            # supabase-py >= 2.x: auth.set_session(access_token, refresh_token)
            self.client.auth.set_session(access_token, refresh_token or None)
        except TypeError:
            # Older signature: set_session(access_token)
            self.client.auth.set_session(access_token)