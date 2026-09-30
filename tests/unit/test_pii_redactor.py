"""Unit tests for the PII redactor (spec §11.1)."""
from src.intake.pii_redactor import redact, has_pii


# --- emails -------------------------------------------------------------

def test_single_email_is_redacted():
    r = redact("email me at alice@example.com please")
    assert "<EMAIL_1>" in r.redacted
    assert "alice@example.com" not in r.redacted
    assert r.replacements["<EMAIL_1>"] == "alice@example.com"


def test_multiple_emails_numbered_in_order():
    r = redact("a@b.com then c@d.org")
    assert r.redacted == "<EMAIL_1> then <EMAIL_2>"
    assert r.replacements["<EMAIL_1>"] == "a@b.com"
    assert r.replacements["<EMAIL_2>"] == "c@d.org"


def test_duplicate_email_reuses_token():
    r = redact("ping a@b.com and a@b.com again")
    assert r.redacted.count("<EMAIL_1>") == 2
    assert len(r.replacements) == 1


# --- phones -------------------------------------------------------------

def test_international_phone_redacted():
    r = redact("call +92 300 1234567 tomorrow")
    assert "<PHONE_1>" in r.redacted
    assert r.replacements["<PHONE_1>"] == "+92 300 1234567"


def test_dashed_phone_redacted():
    r = redact("call 0300-1234567")
    assert "<PHONE_1>" in r.redacted


def test_parens_phone_redacted():
    r = redact("call (021) 1234 5678")
    assert "<PHONE_1>" in r.redacted


def test_short_digit_string_is_not_a_phone():
    # Fewer than 9 digits -> leave alone
    r = redact("order #123-456")
    assert r.replacements == {}


def test_iso_date_is_not_a_phone():
    r = redact("created on 2026-09-22")
    assert "2026-09-22" in r.redacted
    assert r.replacements == {}


# --- mixing -------------------------------------------------------------

def test_email_and_phone_together():
    r = redact("a@b.com or +92 300 1234567")
    assert "<EMAIL_1>" in r.redacted
    assert "<PHONE_1>" in r.redacted


def test_phone_like_substring_inside_email_does_not_become_phone():
    # "1234@x.com" — the "1234" should not be tokenized as a phone.
    r = redact("contact 1234@x.com now")
    assert r.redacted == "contact <EMAIL_1> now"
    assert "PHONE" not in str(r.replacements.keys())


# --- round trip ---------------------------------------------------------

def test_restore_reverses_redaction():
    original = "Email a@b.com or phone +92 300 1234567."
    r = redact(original)
    assert r.restore(r.redacted) == original


# --- no-op cases --------------------------------------------------------

def test_clean_text_unchanged():
    text = "find the best laptop prices in Pakistan"
    r = redact(text)
    assert r.redacted == text
    assert r.replacements == {}


def test_empty_string():
    r = redact("")
    assert r.redacted == ""
    assert r.replacements == {}


# --- has_pii ------------------------------------------------------------

def test_has_pii_true_for_email():
    assert has_pii("a@b.com")


def test_has_pii_true_for_phone():
    assert has_pii("call +92 300 1234567")


def test_has_pii_false_for_plain_text():
    assert not has_pii("best laptop prices")



def test_bare_nine_digit_number_matches():
    # Backtracking used to leave a trailing digit unmatched; ensure the
    # full run is captured.
    r = redact("call 123456789")
    assert "<PHONE_1>" in r.redacted
    assert r.replacements["<PHONE_1>"] == "123456789"


def test_long_number_with_internal_separators():
    r = redact("call 1-800-555-0123")
    assert "<PHONE_1>" in r.redacted