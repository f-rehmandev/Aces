"""
Live auth smoke test — runs against the REAL Supabase project.

Uses the ADMIN API to create the test user (no email sent, no rate limit),
then verifies the user-facing sign-in flow works end-to-end.

Verifies, in order:
    1. .env loads; publishable and service keys are present
    2. Admin API creates a user with email_confirm=True
    3. Public sign-in returns a session
    4. public.users row exists (created by the auth trigger)
    5. RLS lets the user read their own profile
    6. list_clients() returns [] (no memberships yet)
    7. Sign-out works
    8. Admin API deletes the test user (cleanup)

Run:
    python -m scripts.auth_smoke
"""

from __future__ import annotations
import os
import random
import string
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("!! python-dotenv is not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("!! supabase-py is not installed. Run: pip install supabase")
    sys.exit(1)

from src.auth.models import SignInRequest, AuthError, Role        # noqa: E402
from src.auth.supabase_backend import SupabaseAuthBackend         # noqa: E402
from src.auth.service import AuthService                          # noqa: E402


BANNER = "─" * 62

def _line(label: str, value: object = "") -> None:
    print(f"  {label:<24} {value}")

def _fail(msg: str) -> None:
    print(f"\n✗ {msg}")
    print(BANNER)
    sys.exit(1)

def _ok(msg: str) -> None:
    print(f"✓ {msg}")


def _random_email() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"aces.smoke.{suffix}@gmail.com"


def _random_password() -> str:
    """
    Build a password that is GUARANTEED to contain at least one lowercase,
    uppercase, digit, and special character — Supabase's password policy
    requires all four, and a purely random suffix can miss a category.
    """
    import secrets

    alphabet_lower = string.ascii_lowercase
    alphabet_upper = string.ascii_uppercase
    alphabet_digits = string.digits
    # Supabase's allowed special set (see the AuthWeakPasswordError message)
    alphabet_special = "!@#$%^&*()_+-=[]{}"

    # One guaranteed character from each category
    required = [
        secrets.choice(alphabet_lower),
        secrets.choice(alphabet_upper),
        secrets.choice(alphabet_digits),
        secrets.choice(alphabet_special),
    ]
    # Fill the rest from a mixed pool
    pool = alphabet_lower + alphabet_upper + alphabet_digits + alphabet_special
    filler = [secrets.choice(pool) for _ in range(12)]
    chars = required + filler
    # Shuffle so the required chars aren't all at the front
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def main() -> None:
    print(BANNER)
    print("  ACES — live auth smoke test  (admin path)")
    print(BANNER)

    # --- Env ---
    url = os.getenv("SUPABASE_URL", "").strip()
    pub_key = os.getenv("SUPABASE_KEY", "").strip()
    svc_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

    if not url:
        _fail("SUPABASE_URL is missing in .env")
    if not pub_key:
        _fail("SUPABASE_KEY (publishable) is missing in .env")
    if not svc_key:
        _fail(
            "SUPABASE_SERVICE_KEY is missing in .env. The admin path "
            "needs the service key for user provisioning."
        )

    _ok("Loaded .env")
    _line("Project URL:", url)
    _line("Publishable key:", pub_key[:18] + "…")
    _line("Service key:", svc_key[:18] + "…")

    if not pub_key.startswith(("sb_publishable_", "eyJ")):
        _fail("SUPABASE_KEY does not look like a publishable/anon key")
    if not svc_key.startswith(("sb_secret_", "eyJ")):
        _fail("SUPABASE_SERVICE_KEY does not look like a secret/service key")

    # --- Build clients ---
    try:
        admin_client = create_client(url, svc_key)
        backend = SupabaseAuthBackend(url, pub_key)
        svc = AuthService(backend=backend)
    except Exception as e:
        _fail(f"Could not construct Supabase clients: {type(e).__name__}: {e}")
    _ok("Built admin client + SupabaseAuthBackend + AuthService")

    # --- Provision test user via admin API (no email sent) ---
    email = _random_email()
    password = _random_password()
    print()
    _line("Test email:", email)
    _line("Test password:", "(hidden)")

    user_id = ""
    try:
        resp = admin_client.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True,           # <-- key: skip confirmation
            "user_metadata": {"full_name": "Smoke Test"},
        })
    except Exception as e:
        _fail(f"admin.create_user raised: {type(e).__name__}: {e}")

    # supabase-py returns the created user object
    user_obj = getattr(resp, "user", None) or (
        resp.get("user") if isinstance(resp, dict) else None
    )
    user_id = getattr(user_obj, "id", None) or (
        user_obj.get("id") if isinstance(user_obj, dict) else ""
    )
    if not user_id:
        _fail(f"Admin create_user returned no user id: {resp!r}")
    _ok(f"Admin created user {user_id[:8]}… (email pre-confirmed)")

    # From here we do cleanup even if we bail out
    try:
        _run_public_flow(svc, backend, email, password, user_id)
    finally:
        # Cleanup: delete the smoke user via admin API
        try:
            admin_client.auth.admin.delete_user(user_id)
            print()
            _ok(f"Cleanup: admin deleted user {user_id[:8]}…")
        except Exception as e:
            print()
            print(f"  ⚠ Cleanup failed: {type(e).__name__}: {e}")
            print(f"    You can delete manually in Supabase → Authentication → Users")

    print()
    print(BANNER)
    print("  ✓ ALL CHECKS PASSED")
    print(BANNER)
    print()
    print("  Auth is wired end-to-end against your real Supabase project.")
    print("  The smoke user was created and then deleted automatically.")


def _run_public_flow(svc: AuthService, backend, email: str,
                      password: str, user_id: str) -> None:
    """Sign in publicly, verify trigger + RLS, sign out."""
    # --- Sign-in with the publishable key (the real user flow) ---
    print()
    _line("Signing in with publishable key…")
    try:
        session = svc.sign_in(SignInRequest(email=email, password=password))
    except Exception as e:
        _fail(f"sign_in raised: {type(e).__name__}: {e}")
    _ok(f"Got session for user {session.user_id[:8]}…")

    # --- Attach session so RLS sees auth.uid() ---
    try:
        backend.set_user_session(session.access_token, session.refresh_token)
        _ok("Session attached (RLS will now see auth.uid())")
    except Exception as e:
        _fail(f"set_user_session raised: {type(e).__name__}: {e}")

    # --- public.users row (trigger should have created it) ---
    print()
    _line("Fetching profile from public.users…")
    try:
        profile = svc.get_profile(session)
    except Exception as e:
        _fail(f"get_profile raised: {type(e).__name__}: {e}")

    if profile is None:
        _fail(
            "get_profile returned None. Either the auth.users trigger did "
            "not populate public.users, or RLS is blocking the read. "
            "Confirm supabase/schema_auth.sql ran without errors."
        )
    _line("Profile id:", profile.id[:8] + "…")
    _line("Profile email:", profile.email)
    _line("Profile full_name:", profile.full_name)
    _ok("Trigger + RLS verified: public.users row created and readable")

    # --- Client memberships (should be empty) ---
    print()
    _line("Listing client memberships…")
    try:
        clients = svc.list_clients(session)
    except Exception as e:
        _fail(f"list_clients raised: {type(e).__name__}: {e}")
    _line("Memberships:", f"{len(clients)} (expected 1 — auto-created workspace)")
    if len(clients) != 1:
        _fail(
            f"Expected exactly 1 auto-created workspace, got {len(clients)}. "
            "The trigger in supabase/migrations/002_auto_personal_client.sql "
            "may not be installed."
        )
    client, role = clients[0]
    _line("Client slug:", client.slug)
    _line("Client name:", client.name)
    _line("Role:", role.value)
    if role != Role.OWNER:
        _fail(f"Expected role=owner on the auto-created workspace, got {role.value}")
    _ok("Auto-created workspace verified (owner role)")

    # --- Sign-out ---
    print()
    try:
        svc.sign_out(session)
        _ok("sign_out() completed")
    except Exception as e:
        _fail(f"sign_out raised: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()