"""
Client context — the tenant scope for every pipeline operation.

Every read/write in ACES should be scoped to a specific client (spec §43).
We represent that with a small `ClientContext` object, resolvable from a
`Session` (authenticated) or produced as a fallback for anonymous dev use.

Design:
    - `client_id`     — the *string* the pipeline writes into text columns
                        (existing tables use `client_id text`). For an
                        authenticated user this is the client's UUID as a
                        string; for anonymous it's `"default"`.
    - `client_uuid`   — the real UUID for RLS-aware tables (or None when
                        anonymous).
    - `user_id`       — the auth user id, or None when anonymous.
    - `role`          — the user's role in this client, or None when
                        anonymous.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.auth.models import Client, PermissionDenied, Role, Session


# ---------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------

@dataclass
class ClientContext:
    client_id: str                       # string form used by existing tables
    client_uuid: Optional[str] = None    # real UUID for RLS tables
    user_id: Optional[str] = None
    role: Optional[Role] = None
    client_name: str = ""
    client_slug: str = ""
    is_anonymous: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def is_authenticated(self) -> bool:
        return not self.is_anonymous and self.user_id is not None

    def can_write(self) -> bool:
        if self.is_anonymous:
            return True          # local dev fallback
        return self.role is not None and Role.can_write(self.role)

    def can_administer(self) -> bool:
        if self.is_anonymous:
            return False         # anon never administers
        return self.role is not None and Role.can_administer(self.role)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value if self.role else None
        return d


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class NoClientError(Exception):
    """The signed-in user has no client to operate in."""


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

def anonymous_context() -> ClientContext:
    """
    Fallback used when no session exists. Keeps local dev and
    pre-auth code paths working. Downstream code MUST NOT treat this
    as authenticated — the flag is_anonymous exists for exactly that.
    """
    return ClientContext(
        client_id="default",
        client_uuid=None,
        user_id=None,
        role=None,
        client_name="Anonymous (dev)",
        client_slug="default",
        is_anonymous=True,
    )


def resolve_client_context(
    session: Session,
    auth_service,
    preferred_client_id: Optional[str] = None,
) -> ClientContext:
    """
    Build a ClientContext for a signed-in user.

    Resolution order:
        1. `preferred_client_id` if the user is a member of it.
        2. The user's `default_client_id` from their profile.
        3. The user's first membership.

    Raises `NoClientError` if the user has no memberships.
    """
    memberships = auth_service.list_clients(session)
    if not memberships:
        raise NoClientError(
            f"user {session.user_id} has no client memberships"
        )

    by_id: dict[str, tuple[Client, Role]] = {c.id: (c, r) for c, r in memberships}

    # 1. Explicit preference
    if preferred_client_id and preferred_client_id in by_id:
        client, role = by_id[preferred_client_id]
        return _from_membership(session, client, role)

    # 2. Default from profile
    profile = auth_service.get_profile(session)
    if profile and profile.default_client_id:
        match = by_id.get(profile.default_client_id)
        if match:
            client, role = match
            return _from_membership(session, client, role)

    # 3. First membership (stable order: sorted by client name)
    ordered = sorted(by_id.values(), key=lambda t: t[0].name.lower())
    client, role = ordered[0]
    return _from_membership(session, client, role)


def _from_membership(session: Session, client: Client, role: Role) -> ClientContext:
    return ClientContext(
        client_id=client.id,          # UUID as string
        client_uuid=client.id,
        user_id=session.user_id,
        role=role,
        client_name=client.name,
        client_slug=client.slug,
        is_anonymous=False,
        metadata={"client_metadata": client.metadata},
    )


def require_writable(ctx: ClientContext) -> None:
    """Raise PermissionDenied if this context cannot write."""
    if not ctx.can_write():
        raise PermissionDenied(
            f"role {ctx.role.value if ctx.role else 'none'} cannot write"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.auth.models import Client, Role, Session

    # Anonymous
    anon = anonymous_context()
    assert anon.is_anonymous
    assert anon.client_id == "default"
    assert anon.client_uuid is None
    assert anon.can_write()
    assert not anon.can_administer()

    # Stub auth service
    class StubService:
        def __init__(self, clients, default_id=None):
            self._clients = clients
            self._default_id = default_id

        def list_clients(self, session):
            return self._clients

        def get_profile(self, session):
            from src.auth.models import User
            return User(id=session.user_id, default_client_id=self._default_id)

    acme = Client(id="acme-uuid", name="Acme", slug="acme")
    beta = Client(id="beta-uuid", name="Beta", slug="beta")
    session = Session(user_id="u-1", access_token="tok")

    # Case: two memberships, no default -> sorts by name
    svc = StubService([(beta, Role.VIEWER), (acme, Role.OWNER)])
    ctx = resolve_client_context(session, svc)
    assert ctx.client_slug == "acme"
    assert ctx.role == Role.OWNER
    assert ctx.client_uuid == "acme-uuid"
    assert ctx.can_administer()

    # Case: default_client_id wins
    svc = StubService([(beta, Role.VIEWER), (acme, Role.OWNER)],
                       default_id="beta-uuid")
    ctx = resolve_client_context(session, svc)
    assert ctx.client_slug == "beta"
    assert ctx.role == Role.VIEWER
    assert not ctx.can_administer()

    # Case: explicit preference wins over default
    ctx = resolve_client_context(session, svc, preferred_client_id="acme-uuid")
    assert ctx.client_slug == "acme"

    # Case: no memberships -> NoClientError
    try:
        resolve_client_context(session, StubService([]))
        raise AssertionError("expected NoClientError")
    except NoClientError:
        pass

    # Case: require_writable — viewer cannot write
    ctx = resolve_client_context(session, StubService([(acme, Role.VIEWER)]))
    try:
        require_writable(ctx)
        raise AssertionError("expected PermissionDenied for viewer")
    except PermissionDenied:
        pass
    # Case: require_writable rejects a context with no role
    no_role_ctx = ClientContext(
        client_id="x", user_id="u", role=None, is_anonymous=False,
    )
    try:
        require_writable(no_role_ctx)
        raise AssertionError("expected PermissionDenied")
    except PermissionDenied:
        pass

    # to_dict serializes the role enum as a string
    viewer_ctx = _from_membership(session, acme, Role.VIEWER)
    d = viewer_ctx.to_dict()
    assert d["role"] == "viewer"
    assert d["is_anonymous"] is False

    # to_dict on a no-role context yields None for role
    d2 = no_role_ctx.to_dict()
    assert d2["role"] is None
    assert d2["is_anonymous"] is False

    print("ClientContext OK.")