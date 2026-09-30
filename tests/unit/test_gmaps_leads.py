"""Unit tests for the Google Maps leads pipeline."""

import pytest

from src.discovery.gmaps_client import GmapsJobResult, GmapsScraperUnavailable
from src.discovery.gmaps_leads import (
    LeadsResult, _normalize_row, run_leads_query,
)


# ---------------------------------------------------------------------------
# Fake client — duck-types GmapsClient without network
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self, records=None, raise_unavailable=False,
                 succeed=True, error=""):
        self._records = list(records or [])
        self._raise = raise_unavailable
        self._succeed = succeed
        self._error = error
        self.last_keywords = None

    def scrape(self, keywords, **kwargs):
        self.last_keywords = list(keywords)
        if self._raise:
            raise GmapsScraperUnavailable("fake down")
        return GmapsJobResult(
            job_id="fake",
            status="ok" if self._succeed else "failed",
            records=list(self._records),
            error=self._error,
        )


def _good_rows():
    return [
        {"title": "Pizza A", "address": "A St", "phone": "0300a",
         "website": "https://a.example"},
        {"title": "Pizza B", "address": "B St", "phone": "0300b",
         "website": ""},
        {"title": "Pizza C", "address": "C St", "phone": "0300c",
         "website": "https://c.example"},
    ]


# ---------------------------------------------------------------------------
# _normalize_row
# ---------------------------------------------------------------------------

def test_normalize_row_maps_all_columns():
    raw = {
        "title": "Pizza Palace",
        "address": "123 Main St",
        "phone": "03001234567",
        "website": "https://palace.example",
        "email": "hi@palace.example",
        "category": "Pizza",
        "review_rating": "4.5",
        "review_count": "120",
        "latitude": "31.5",
        "longitude": "74.3",
        "place_id": "abc123",
        "link": "https://maps.google.com/...",
    }
    out = _normalize_row(raw)
    assert out["business_name"] == "Pizza Palace"
    assert out["phone"] == "03001234567"
    assert out["website"] == "https://palace.example"
    assert out["has_website"] is True
    assert out["source_url"].startswith("https://maps.google")


def test_normalize_row_empty_website_sets_false():
    out = _normalize_row({"title": "X", "website": ""})
    assert out["has_website"] is False


def test_normalize_row_missing_fields_become_empty_strings():
    out = _normalize_row({"title": "X"})
    assert out["phone"] == ""
    assert out["address"] == ""
    assert out["email"] == ""


def test_normalize_row_null_values_become_empty():
    out = _normalize_row({"title": "X", "phone": None})
    assert out["phone"] == ""

def test_normalize_row_facebook_counts_as_no_website():
    out = _normalize_row({"title": "X", "website": "https://facebook.com/pizzavizza"})
    assert out["has_website"] is False
    assert out["website_type"] == "social"
    assert out["website"] == "https://facebook.com/pizzavizza"  # raw preserved


def test_normalize_row_instagram_counts_as_no_website():
    out = _normalize_row({"title": "X", "website": "https://www.instagram.com/xyz"})
    assert out["has_website"] is False
    assert out["website_type"] == "social"


def test_normalize_row_own_site_still_counts():
    out = _normalize_row({"title": "X", "website": "https://pizzavizza.pk"})
    assert out["has_website"] is True
    assert out["website_type"] == "own"

# ---------------------------------------------------------------------------
# run_leads_query — happy path
# ---------------------------------------------------------------------------

def test_run_leads_query_returns_all_rows():
    client = FakeClient(_good_rows())
    result = run_leads_query(["pizza shops in Lahore"], client=client)
    assert result.succeeded
    assert result.total_scraped == 3
    assert result.total_returned == 3
    assert result.filtered_no_website == 0


def test_run_leads_query_no_website_filter_keeps_one():
    client = FakeClient(_good_rows())
    result = run_leads_query(
        ["pizza shops in Lahore"], no_website_only=True, client=client,
    )
    assert result.total_scraped == 3
    assert result.total_returned == 1
    assert result.filtered_no_website == 2
    assert result.records[0]["business_name"] == "Pizza B"


def test_run_leads_query_passes_keywords_to_client():
    client = FakeClient(_good_rows())
    run_leads_query(["dentists in Karachi"], client=client)
    assert client.last_keywords == ["dentists in Karachi"]


# ---------------------------------------------------------------------------
# run_leads_query — failure paths
# ---------------------------------------------------------------------------

def test_run_leads_query_handles_unavailable_server():
    client = FakeClient(raise_unavailable=True)
    result = run_leads_query(["x"], client=client)
    assert not result.succeeded
    assert "unavailable" in result.error.lower()


def test_run_leads_query_handles_failed_job():
    client = FakeClient(succeed=False, error="blocked by Google")
    result = run_leads_query(["x"], client=client)
    assert not result.succeeded
    assert "blocked" in result.error.lower()


def test_run_leads_query_empty_records_returns_zero():
    client = FakeClient(records=[])
    result = run_leads_query(["x"], client=client)
    assert result.total_scraped == 0
    assert result.total_returned == 0


# ---------------------------------------------------------------------------
# run_leads_query — output writing
# ---------------------------------------------------------------------------

def test_run_leads_query_writes_output_when_records(tmp_path):
    client = FakeClient(_good_rows())
    out = tmp_path / "leads.xlsx"
    result = run_leads_query(
        ["pizza"], client=client, output_path=out,
    )
    assert result.output_path is not None
    assert out.exists()


def test_run_leads_query_no_output_when_empty(tmp_path):
    client = FakeClient(records=[])
    out = tmp_path / "empty.xlsx"
    result = run_leads_query(
        ["x"], client=client, output_path=out,
    )
    assert result.output_path is None
    assert not out.exists()


def test_run_leads_query_no_output_when_path_not_given():
    client = FakeClient(_good_rows())
    result = run_leads_query(["pizza"], client=client)
    assert result.output_path is None