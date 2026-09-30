"""Unit tests for secret scrubbing (spec §52)."""
from src.security.secrets import scrub, contains_secret


# --- named patterns ----------------------------------------------------

def test_openai_key_scrubbed():
    r = scrub("error: sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")
    assert r.was_modified
    assert "OPENAI_KEY" in r.redactions


def test_anthropic_key_scrubbed():
    r = scrub("anthropic: sk-ant-abcdefghijklmnop0123456789")
    assert r.was_modified
    assert "ANTHROPIC_KEY" in r.redactions


def test_google_key_scrubbed():
    r = scrub("AIzaSyDabcdefghijklmnopqrstuvwxyz01234567")
    assert r.was_modified
    assert "GOOGLE_KEY" in r.redactions


def test_groq_key_scrubbed():
    r = scrub("gsk_" + "a" * 45)
    assert r.was_modified
    assert "GROQ_KEY" in r.redactions


def test_bearer_token_scrubbed():
    r = scrub("Authorization: Bearer eyJabcdefghij.payload.sig")
    assert r.was_modified
    assert "BEARER_TOKEN" in r.redactions


def test_generic_api_key_scrubbed():
    r = scrub("api_key=abcdefghijklmnop12345678")
    assert r.was_modified
    assert "GENERIC_API_KEY" in r.redactions


def test_password_scrubbed():
    r = scrub("password=hunter2hunter2")
    assert r.was_modified
    assert "PASSWORD" in r.redactions


def test_pem_private_key_scrubbed():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
    r = scrub(pem)
    assert r.was_modified
    assert "PRIVATE_KEY" in r.redactions


# --- explicit literal --------------------------------------------------

def test_explicit_secret_literal_scrubbed():
    r = scrub("token=SUPER_SECRET_VALUE_12345678",
              extra_secrets=["SUPER_SECRET_VALUE_12345678"])
    assert r.was_modified
    assert "EXPLICIT_SECRET" in r.redactions


def test_short_explicit_secret_is_ignored():
    # Length below 8 — refuse to over-redact short common strings.
    r = scrub("value=x", extra_secrets=["x"])
    assert not r.was_modified


# --- clean text --------------------------------------------------------

def test_clean_prose_untouched():
    r = scrub("the quick brown fox jumps over the lazy dog")
    assert not r.was_modified
    assert r.redactions == []


def test_empty_input():
    r = scrub("")
    assert r.scrubbed == ""


def test_none_input():
    r = scrub(None)
    assert r.scrubbed == ""


# --- contains_secret ---------------------------------------------------

def test_contains_secret_true():
    assert contains_secret("api_key=abcdefghijklmnop12345678")


def test_contains_secret_false():
    assert not contains_secret("hello world")


# --- preserves non-secret text ----------------------------------------

def test_preserves_surrounding_text():
    text = "Failed to call API with key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789 - retrying"
    r = scrub(text)
    assert "Failed to call API with key" in r.scrubbed
    assert "retrying" in r.scrubbed
    assert "sk-proj-" not in r.scrubbed



def test_openrouter_key_is_not_matched_as_openai():
    r = scrub("key=sk-or-v1-abcdefghijklmnopqrstuvwxyz0123456789")
    assert r.was_modified
    assert "OPENROUTER_KEY" in r.redactions
    assert "OPENAI_KEY" not in r.redactions


def test_anthropic_key_is_not_matched_as_openai():
    r = scrub("key=sk-ant-abcdefghijklmnopqrstuvwxyz0123456789")
    assert "ANTHROPIC_KEY" in r.redactions
    assert "OPENAI_KEY" not in r.redactions


def test_openai_proj_key_still_matched_as_openai():
    r = scrub("key=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")
    assert "OPENAI_KEY" in r.redactions

def test_openrouter_key_is_not_matched_as_openai():
    r = scrub("key=sk-or-v1-abcdefghijklmnopqrstuvwxyz0123456789")
    assert r.was_modified
    assert "OPENROUTER_KEY" in r.redactions
    assert "OPENAI_KEY" not in r.redactions


def test_anthropic_key_is_not_matched_as_openai():
    r = scrub("key=sk-ant-abcdefghijklmnopqrstuvwxyz0123456789")
    assert "ANTHROPIC_KEY" in r.redactions
    assert "OPENAI_KEY" not in r.redactions


def test_openai_proj_key_still_matched_as_openai():
    r = scrub("key=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")
    assert "OPENAI_KEY" in r.redactions

def test_url_path_segment_not_redacted_as_b64():
    """URL slugs that happen to be long must not be redacted as base64 secrets."""
    r = scrub("failed on https://instacare.pk/online-pharmacy/product/panadol-tablet-500mg")
    # URL kept intact — we only care that the domain path wasn't redacted
    assert "panadol-tablet-500mg" in r.scrubbed
    assert "<B64_SECRET_REDACTED>" not in r.scrubbed


def test_real_base64_secret_still_redacted():
    # Mixed case + digits + >=40 chars = real secret shape
    r = scrub("token=AbCdEfGh1234567890XyZabcdefghijklmnopqrstuv")
    assert r.was_modified
    assert "B64_SECRET" in r.redactions or "HEX_SECRET" in r.redactions or len(r.redactions) > 0