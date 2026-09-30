"""Unit tests for client context resolution (spec §43)."""
import pytest

from src.auth.context import (
    ClientContext, NoClientError, anonymous_context,
    resolve_client_context, require_writable, _from_membership,
)
from src.auth.models import Client, PermissionDenied, Role, Session, User


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------

class StubService:
    def __init__(self, clients, default_id=None, profile=None):
        self._clients = clients
        self._default_id = default_id
        self._profile = profile

    def list_clients(self, session):
        return list(self._clients)

    def get_profile(self, session):
        if self._profile is not None:
            return self._profile
        return User(id=session.user_id, default_client_id=self._default_id)


def _session():
    return Session(user_id="u-1", access_token="tok")


def _acme():
    return Client(id="c-acme", name="Acme", slug="acme")


def _beta():
    return Client(id="c-beta", name="Beta", slug="beta")


# ---------------------------------------------------------------------------
# Anonymous
# ---------------------------------------------------------------------------

def test_anonymous_context():
    ctx = anonymous_context()
    assert ctx.is_anonymous
    assert not ctx.is_authenticated
    assert ctx.client_id == "default"
    assert ctx.client_uuid is None
    assert ctx.user_id is None
    assert ctx.role is None


def test_anonymous_can_write_but_not_administer():
    ctx = anonymous_context()
    assert ctx.can_write()
    assert not ctx.can_administer()


def test_anonymous_require_writable_ok():
    require_writable(anonymous_context())  # must not raise


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolves_first_client_when_no_default():
    svc = StubService([(_beta(), Role.VIEWER), (_acme(), Role.OWNER)])
    ctx = resolve_client_context(_session(), svc)
    # Sorted by name -> Acme
    assert ctx.client_slug == "acme"
    assert ctx.role == Role.OWNER
    assert ctx.client_uuid == "c-acme"
    assert not ctx.is_anonymous
    assert ctx.is_authenticated


def test_default_client_id_wins():
    svc = StubService([(_beta(), Role.VIEWER), (_acme(), Role.OWNER)],
                       default_id="c-beta")
    ctx = resolve_client_context(_session(), svc)
    assert ctx.client_slug == "beta"
    assert ctx.role == Role.VIEWER


def test_explicit_preference_wins_over_default():
    svc = StubService([(_beta(), Role.VIEWER), (_acme(), Role.OWNER)],
                       default_id="c-beta")
    ctx = resolve_client_context(_session(), svc, preferred_client_id="c-acme")
    assert ctx.client_slug == "acme"


def test_preference_ignored_if_not_a_member():
    svc = StubService([(_acme(), Role.OWNER)])
    ctx = resolve_client_context(_session(), svc,
                                  preferred_client_id="c-not-a-member")
    # Falls back to the only real membership
    assert ctx.client_slug == "acme"


def test_default_ignored_if_not_a_member():
    svc = StubService([(_acme(), Role.OWNER)], default_id="c-ghost")
    ctx = resolve_client_context(_session(), svc)
    assert ctx.client_slug == "acme"


def test_no_memberships_raises():
    svc = StubService([])
    with pytest.raises(NoClientError):
        resolve_client_context(_session(), svc)


# ---------------------------------------------------------------------------
# Role gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,can_write,can_admin", [
    (Role.OWNER,  True,  True),
    (Role.ADMIN,  True,  True),
    (Role.MEMBER, True,  False),
    (Role.VIEWER, False, False),     # viewer can only read
])
def test_role_gates(role, can_write, can_admin):
    ctx = _from_membership(_session(), _acme(), role)
    assert ctx.can_write() == can_write
    assert ctx.can_administer() == can_admin


def test_require_writable_raises_for_no_role():
    ctx = ClientContext(client_id="x", user_id="u-1", role=None,
                        is_anonymous=False)
    with pytest.raises(PermissionDenied):
        require_writable(ctx)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_shape():
    ctx = _from_membership(_session(), _acme(), Role.OWNER)
    d = ctx.to_dict()
    assert d["client_id"] == "c-acme"
    assert d["client_uuid"] == "c-acme"
    assert d["user_id"] == "u-1"
    assert d["role"] == "owner"
    assert d["is_anonymous"] is False
    assert d["client_slug"] == "acme"


def test_anonymous_to_dict_has_none_role():
    d = anonymous_context().to_dict()
    assert d["role"] is None
    assert d["is_anonymous"] is True