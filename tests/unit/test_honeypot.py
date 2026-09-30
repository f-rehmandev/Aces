"""Unit tests for honeypot detection (spec §49.3)."""
from src.security.honeypot import (
    HoneypotDetector, HoneypotFinding, HoneypotReport,
)


# --- hidden links -------------------------------------------------------

def test_display_none_link_flagged():
    html = '<a href="/x" style="display:none">hidden</a>'
    r = HoneypotDetector().detect_in_html(html)
    assert r.is_suspicious
    assert any(f.kind == "hidden_link" for f in r.findings)


def test_visibility_hidden_link_flagged():
    html = '<a href="/x" style="visibility:hidden">hidden</a>'
    r = HoneypotDetector().detect_in_html(html)
    assert r.is_suspicious


def test_hidden_class_link_flagged():
    html = '<a href="/x" class="hidden-spam">text</a>'
    r = HoneypotDetector().detect_in_html(html)
    assert r.is_suspicious


def test_nofollow_with_mismatched_visible_url_flagged():
    html = '<a rel="nofollow" href="https://evil.example/path">https://example.com/ok</a>'
    r = HoneypotDetector().detect_in_html(html)
    assert r.is_suspicious
    assert any("nofollow" in f.description for f in r.findings)


def test_plain_visible_link_not_flagged():
    html = '<a href="/x">x</a>'
    r = HoneypotDetector().detect_in_html(html)
    assert not r.is_suspicious


def test_nofollow_with_matching_text_not_flagged():
    html = '<a rel="nofollow" href="https://x.com/a">https://x.com/a</a>'
    r = HoneypotDetector().detect_in_html(html)
    assert not r.is_suspicious


# --- hidden forms -------------------------------------------------------

def test_honeypot_form_field_flagged():
    html = '<input type="hidden" name="honeypot" value="">'
    r = HoneypotDetector().detect_in_html(html)
    assert r.is_suspicious
    assert any(f.kind == "hidden_form" for f in r.findings)


def test_trap_field_name_flagged():
    html = '<input type="hidden" name="trap_field">'
    r = HoneypotDetector().detect_in_html(html)
    assert r.is_suspicious


def test_regular_csrf_token_not_flagged():
    html = '<input type="hidden" name="csrf_token" value="abc">'
    r = HoneypotDetector().detect_in_html(html)
    assert not r.is_suspicious


def test_visible_input_not_flagged():
    html = '<input type="text" name="q">'
    r = HoneypotDetector().detect_in_html(html)
    assert not r.is_suspicious


# --- duplicate burst ---------------------------------------------------

def test_duplicate_burst_threshold_flagged():
    records = [{"t": "same"} for _ in range(6)]
    r = HoneypotDetector().detect_duplicate_burst(records, key_fn=lambda x: x["t"])
    assert r.is_suspicious
    assert r.findings[0].kind == "duplicate_burst"


def test_below_duplicate_threshold_not_flagged():
    records = [{"t": "a"}, {"t": "b"}, {"t": "c"}]
    r = HoneypotDetector().detect_duplicate_burst(records, key_fn=lambda x: x["t"])
    assert not r.is_suspicious


def test_ratio_burst_flagged():
    records = [{"t": "a"}] * 4 + [{"t": "b"}] * 4 + [{"t": "c"}] * 2
    detector = HoneypotDetector(
        duplicate_threshold=5,
        duplicate_ratio_threshold=0.35,
    )
    r = detector.detect_duplicate_burst(records, key_fn=lambda x: x["t"])
    assert r.is_suspicious


def test_no_records_no_finding():
    r = HoneypotDetector().detect_duplicate_burst([], key_fn=lambda x: x["t"])
    assert not r.is_suspicious


def test_records_without_keys_are_ignored():
    records = [{"junk": i} for i in range(20)]
    r = HoneypotDetector().detect_duplicate_burst(records, key_fn=lambda x: "")
    assert not r.is_suspicious


# --- combined detect() -------------------------------------------------

def test_combined_html_and_records():
    html = '<a href="/x" style="display:none">trap</a>'
    records = [{"t": "same"} for _ in range(10)]
    r = HoneypotDetector().detect(html=html, records=records, key_fn=lambda x: x["t"])
    assert r.is_suspicious
    assert "hidden_link" in r.suspicious_kinds
    assert "duplicate_burst" in r.suspicious_kinds


def test_combined_only_html():
    html = '<a href="/x" style="display:none">trap</a>'
    r = HoneypotDetector().detect(html=html)
    assert r.is_suspicious


def test_empty_html():
    r = HoneypotDetector().detect_in_html("")
    assert not r.is_suspicious


# --- summary -----------------------------------------------------------

def test_summary_counts_kinds():
    html = (
        '<a href="/x" style="display:none">a</a>'
        '<a href="/y" style="visibility:hidden">b</a>'
    )
    r = HoneypotDetector().detect_in_html(html)
    s = r.summary()
    assert "hidden_link" in s