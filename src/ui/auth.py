"""
Auth UI + session state.

Streamlit is stateless per rerun, so the signed-in Session lives in
st.session_state["aces_session"] and is re-validated on every rerun.

Exposes:
    get_auth_service()            -> AuthService (or None if env unset)
    get_session()                 -> Session | None
    get_context()                 -> ClientContext
    sign_in_email(email, pw)      -> (ok, error)
    sign_up_email(email, pw, name)-> (ok, error, needs_verification)
    sign_out()                    -> None
    render_auth_screen()          -> blocks with sign-in / sign-up UI
    render_user_chip()            -> small user + sign-out chip for header

The `get_context()` helper returns an anonymous context when unset, so
the rest of the app keeps working in dev without Supabase.
"""

from __future__ import annotations
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from src.auth.context import (
    ClientContext, NoClientError, anonymous_context, resolve_client_context,
)
from src.auth.models import (
    AuthError, EmailNotVerified, InvalidCredentials, Role, Session,
    SignInRequest, SignUpRequest, UserAlreadyExists,
)
from src.auth.service import AuthService, make_supabase_backend


SESSION_KEY = "aces_session"
CONTEXT_KEY = "aces_context"


# ---------------------------------------------------------------------------
# Service construction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_auth_service() -> Optional[AuthService]:
    """
    Build the AuthService once per process. Returns None if SUPABASE_URL
    or SUPABASE_KEY is missing, so the app can fall back to anonymous mode.
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    try:
        backend = make_supabase_backend(url, key)
        return AuthService(backend=backend)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def get_session() -> Optional[Session]:
    s = st.session_state.get(SESSION_KEY)
    if s is None:
        return None
    # Expired -> attempt refresh if we can, else drop
    if s.is_expired():
        svc = get_auth_service()
        if svc and s.refresh_token:
            try:
                refreshed = svc.refresh(s)
                st.session_state[SESSION_KEY] = refreshed
                return refreshed
            except AuthError:
                st.session_state.pop(SESSION_KEY, None)
                return None
        st.session_state.pop(SESSION_KEY, None)
        return None
    return s


def get_context() -> ClientContext:
    """
    Resolve the ClientContext for the current session. Falls back to
    anonymous on any failure so the app never hard-crashes on auth.
    """
    cached = st.session_state.get(CONTEXT_KEY)
    if cached is not None:
        return cached

    session = get_session()
    if session is None:
        ctx = anonymous_context()
        st.session_state[CONTEXT_KEY] = ctx
        return ctx

    svc = get_auth_service()
    if svc is None:
        ctx = anonymous_context()
        st.session_state[CONTEXT_KEY] = ctx
        return ctx

    # Tell the backend to use this user's JWT for RLS
    try:
        svc.backend.set_user_session(session.access_token, session.refresh_token)
    except Exception:
        pass

    try:
        ctx = resolve_client_context(session, svc)
    except NoClientError:
        ctx = anonymous_context()
    except Exception:
        ctx = anonymous_context()

    st.session_state[CONTEXT_KEY] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Auth actions
# ---------------------------------------------------------------------------

def sign_in_email(email: str, password: str) -> tuple[bool, str]:
    svc = get_auth_service()
    if svc is None:
        return False, "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env."
    try:
        session = svc.sign_in(SignInRequest(email=email.strip(), password=password))
    except InvalidCredentials:
        return False, "Wrong email or password."
    except EmailNotVerified:
        return False, "Check your inbox to confirm your email, then try again."
    except AuthError as e:
        return False, str(e)

    _store_session(session)
    return True, ""


def sign_up_email(email: str, password: str, full_name: str) -> tuple[bool, str, bool]:
    """
    Returns (ok, error_message, needs_email_verification).
    """
    svc = get_auth_service()
    if svc is None:
        return False, "Supabase is not configured.", False

    # Client-side password policy pre-check (matches Supabase's policy)
    missing = _missing_password_categories(password)
    if missing:
        return False, f"Password needs at least one: {', '.join(missing)}.", False

    try:
        outcome = svc.sign_up(SignUpRequest(
            email=email.strip(), password=password, full_name=full_name.strip(),
        ))
    except UserAlreadyExists:
        return False, "That email is already registered. Try signing in.", False
    except AuthError as e:
        return False, str(e), False

    if outcome.session is None:
        return True, "", True

    _store_session(outcome.session)
    return True, "", False


def sign_out() -> None:
    session = get_session()
    svc = get_auth_service()
    if session and svc:
        try:
            svc.sign_out(session)
        except Exception:
            pass
    st.session_state.pop(SESSION_KEY, None)
    st.session_state.pop(CONTEXT_KEY, None)


def _store_session(session: Session) -> None:
    st.session_state[SESSION_KEY] = session
    # Clear any cached context so it re-resolves with the new identity
    st.session_state.pop(CONTEXT_KEY, None)
    # Attach JWT to the backend for subsequent RLS reads
    svc = get_auth_service()
    if svc:
        try:
            svc.backend.set_user_session(session.access_token, session.refresh_token)
        except Exception:
            pass


def _missing_password_categories(pw: str) -> list[str]:
    missing: list[str] = []
    if not any(c.islower() for c in pw):
        missing.append("lowercase")
    if not any(c.isupper() for c in pw):
        missing.append("uppercase")
    if not any(c.isdigit() for c in pw):
        missing.append("digit")
    if not any(c in "!@#$%^&*()_+-=[]{};:'\"|<>?,./`~" for c in pw):
        missing.append("special character")
    return missing


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

_AUTH_CSS = """
<style>
.aces-auth-wrap {
    max-width: 420px;
    margin: 3rem auto 2rem auto;
}
.aces-auth-card {
    background: #252B33;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 10px;
    padding: 2rem 1.6rem 1.4rem 1.6rem;
}
.aces-auth-title {
    color: #DCE2EC;
    font-size: 1.4rem;
    font-weight: 640;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}
.aces-auth-sub {
    color: #87919F;
    font-size: 0.85rem;
    margin: 0 0 1.4rem 0;
    line-height: 1.55;
}
.aces-auth-brand {
    text-align: center;
    margin-bottom: 1.6rem;
}
.aces-auth-brand-mark {
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 38px; height: 38px;
    border-radius: 9px;
    background: #2D333B;
    border: 1px solid rgba(129, 144, 255, 0.22);
    color: #B9C0FF;
    font-weight: 700; font-size: 16px;
    margin-bottom: 0.6rem;
}
.aces-auth-brand-name {
    color: #D8DEE8;
    font-weight: 640;
    font-size: 0.95rem;
    letter-spacing: -0.01em;
}
.aces-auth-brand-sub {
    color: #697381;
    font-size: 0.72rem;
    margin-top: 2px;
}
.aces-auth-oauth-btn {
    display: block; width: 100%;
    padding: 0.7rem 1rem;
    background: #2A313A;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 8px;
    color: #D8DEE8;
    font-size: 0.85rem;
    font-weight: 560;
    text-align: center;
    text-decoration: none !important;
    margin-bottom: 1.2rem;
    transition: border-color 120ms ease, background 120ms ease;
}
.aces-auth-oauth-btn:hover {
    background: #303842;
    border-color: rgba(129, 144, 255, 0.35);
    color: #E3E7F0;
}
.aces-auth-divider {
    display: flex; align-items: center; gap: 0.75rem;
    color: #5A6370;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.2rem 0 1rem 0;
}
.aces-auth-divider::before,
.aces-auth-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(148, 163, 184, 0.11);
}
.aces-auth-hint {
    color: #697381;
    font-size: 0.72rem;
    text-align: center;
    margin-top: 1.2rem;
    line-height: 1.55;
}
</style>
"""


def render_auth_screen() -> None:
    """
    Renders the sign-in / sign-up card and calls st.stop() if the user
    isn't authenticated. The caller writes:

        if not render_auth_screen():
            st.stop()
    """
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    svc = get_auth_service()
    st.markdown('<div class="aces-auth-wrap">', unsafe_allow_html=True)

    st.markdown(
        '<div class="aces-auth-brand">'
        '<div class="aces-auth-brand-mark">A</div>'
        '<div class="aces-auth-brand-name">ACES</div>'
        '<div class="aces-auth-brand-sub">Autonomous Cognitive Extraction System</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if svc is None:
        st.warning(
            "Supabase is not configured. Set `SUPABASE_URL` and "
            "`SUPABASE_KEY` in `.env` and restart the app."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="aces-auth-card">', unsafe_allow_html=True)

    tab_signin, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_signin:
        st.markdown(
            '<div class="aces-auth-title">Welcome back</div>'
            '<div class="aces-auth-sub">Sign in to your workspace.</div>',
            unsafe_allow_html=True,
        )
        with st.form("signin_form", clear_on_submit=False):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True,
                                              type="primary")
        if submitted:
            if not email or not password:
                st.error("Enter your email and password.")
            else:
                ok, err = sign_in_email(email, password)
                if ok:
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error(err)

    with tab_signup:
        st.markdown(
            '<div class="aces-auth-title">Create your workspace</div>'
            '<div class="aces-auth-sub">'
            'Every account gets a personal workspace automatically.'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.form("signup_form", clear_on_submit=False):
            name = st.text_input("Full name", key="signup_name")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input(
                "Password", type="password", key="signup_password",
                help="At least 8 characters, with lowercase, uppercase, digit, and a special character.",
            )
            submitted = st.form_submit_button("Create account",
                                              use_container_width=True,
                                              type="primary")
        if submitted:
            if not new_email or not new_password:
                st.error("Enter an email and password.")
            else:
                ok, err, needs_verify = sign_up_email(new_email, new_password, name)
                if not ok:
                    st.error(err)
                elif needs_verify:
                    st.success(
                        "Account created. Check your inbox to confirm your "
                        "email, then sign in."
                    )
                else:
                    st.success("Account created and signed in.")
                    st.rerun()

    # Google OAuth — informational for now
    st.markdown('<div class="aces-auth-divider">or</div>', unsafe_allow_html=True)
    st.markdown(
        '<a href="#" class="aces-auth-oauth-btn" onclick="return false;">'
        'Continue with Google'
        '</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="aces-auth-hint">'
        'Google sign-in becomes fully functional once the web frontend '
        'handles OAuth callbacks. Until then, use email sign-in above.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)   # card
    st.markdown('</div>', unsafe_allow_html=True)   # wrap


def render_user_chip() -> None:
    """Small signed-in user widget for the app header."""
    session = get_session()
    if session is None:
        return
    ctx = get_context()
    label = session.email or ctx.user_id or "(unknown user)"
    role = ctx.role.value if ctx.role else "?"
    st.caption(f"{label} · {ctx.client_name or 'workspace'} · {role}")
    if st.button("Sign out", key="header_signout", use_container_width=False):
        sign_out()
        st.rerun()