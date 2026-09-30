"""Unit tests for intent-inference helpers (spec §11.1)."""
from src.intake.intent_helpers import (
    infer_navigation, infer_output_format, infer_compliance,
    make_safe_prompt, infer_language,
)


# --- navigation: pagination --------------------------------------------

def test_plural_pages_signals_auto():
    assert infer_navigation("get all items across pages").pagination == "auto"


def test_word_pagination_signals_auto():
    assert infer_navigation("paginate through results").pagination == "auto"


def test_next_page_signals_auto():
    assert infer_navigation("follow the next page link").pagination == "auto"


def test_load_more_signals_auto():
    assert infer_navigation("click load more to get everything").pagination == "auto"


def test_just_this_one_page_is_none():
    assert infer_navigation("just this one page").pagination == "none"


def test_single_page_phrase_is_none():
    assert infer_navigation("only scrape the single page").pagination == "none"


def test_first_page_only_is_none():
    assert infer_navigation("first page only please").pagination == "none"


def test_neutral_prompt_defaults_to_none():
    assert infer_navigation("find laptop prices").pagination == "none"


def test_bare_singular_page_is_not_a_signal():
    # Regression: "page" alone used to wrongly trigger auto.
    assert infer_navigation("this page").pagination == "none"


def test_llm_hint_overrides_heuristic():
    h = infer_navigation("anything", llm_hints={"pagination": "cursor"})
    assert h.pagination == "cursor"


# --- navigation: detail pages ------------------------------------------

def test_detail_pages_required_when_mentioned():
    h = infer_navigation("get the detail page for each item")
    assert h.detail_pages == "required"


def test_detail_pages_optional_when_pagination_auto():
    h = infer_navigation("browse all pages")
    assert h.detail_pages == "optional"


def test_detail_pages_none_when_neither():
    h = infer_navigation("find laptop prices")
    assert h.detail_pages == "none"


# --- navigation: max_pages ---------------------------------------------

def test_max_pages_extracted_from_prompt():
    assert infer_navigation("get the first 3 pages").max_pages == 3


def test_max_pages_capped_at_500():
    assert infer_navigation("get the first 9999 pages").max_pages == 500


def test_max_pages_default_when_absent():
    assert infer_navigation("find prices").max_pages == 5


def test_llm_hint_overrides_max_pages():
    h = infer_navigation("find prices", llm_hints={"max_pages": 25})
    assert h.max_pages == 25


# --- output format -----------------------------------------------------

def test_output_csv():
    assert infer_output_format("give me a csv") == "csv"


def test_output_excel_equals_xlsx():
    assert infer_output_format("as an Excel file") == "xlsx"


def test_output_json():
    assert infer_output_format("export as json please") == "json"


def test_output_parquet():
    assert infer_output_format("parquet format") == "parquet"


def test_output_default():
    assert infer_output_format("find prices") == "xlsx"


def test_output_specificity_parquet_beats_csv():
    # Prompt mentions both; parquet is more specific and wins.
    assert infer_output_format("parquet is fine but csv also ok") == "parquet"


# --- compliance --------------------------------------------------------

def test_compliance_allows_normal_prompt():
    h = infer_compliance("scrape public prices from example.com")
    assert h.allowed is True
    assert h.refusal_reason == ""


def test_compliance_refuses_login_bypass():
    h = infer_compliance("bypass the login wall")
    assert h.allowed is False
    assert "bypass" in h.refusal_reason.lower() or "access" in h.refusal_reason.lower()


def test_compliance_refuses_captcha_solving():
    h = infer_compliance("solve captcha on this site")
    assert h.allowed is False


def test_compliance_refuses_credential_theft():
    h = infer_compliance("steal passwords from users")
    assert h.allowed is False


# --- PII ---------------------------------------------------------------

def test_safe_prompt_redacts_email():
    sp = make_safe_prompt("email me at a@b.com")
    assert "<EMAIL_1>" in sp.redacted
    assert "a@b.com" not in sp.redacted


def test_safe_prompt_round_trip():
    sp = make_safe_prompt("email a@b.com or call +92 300 1234567")
    assert sp.restore(sp.redacted) == "email a@b.com or call +92 300 1234567"


# --- language ----------------------------------------------------------

def test_language_latin():
    assert infer_language("hello world") == "latin"


def test_language_cjk():
    assert infer_language("你好") == "cjk"


def test_language_unknown_for_digits():
    assert infer_language("12345") == "unknown"