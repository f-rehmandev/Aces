"""Unit tests for the Google Maps lead provider client (spec §14)."""
import pytest

from src.discovery.gmaps_client import (
    GmapsClient, GmapsError, GmapsJobFailed, GmapsScraperUnavailable,
    filter_no_website, _extract_inline_records,
)


# ---------------------------------------------------------------------------
# Fake transport — no network
# ---------------------------------------------------------------------------

class FakeTransport:
    def __init__(self):
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self.bytes_gets: list[str] = []
        self.jobs: dict[str, dict] = {}
        self.csv_payload: bytes = b""

        # behaviour knobs
        self.available = True
        self.next_job_id = "job-1"
        self.status_sequence: list[str] = ["ok"]
        self.fail_create = False
        self.fail_download = False

    def post_json(self, path, body):
        self.posts.append((path, body))
        if self.fail_create:
            return 500, {"error": "boom"}
        if path == "/api/v1/jobs":
            # Don't clobber a job entry the test pre-seeded. Only create
            # the default placeholder if the id is not already present.
         if path == "/api/v1/jobs":
            # Don't clobber a pre-seeded job entry (some tests set the
            # payload manually before calling scrape()).
            if self.next_job_id not in self.jobs:
                self.jobs[self.next_job_id] = {"status": self.status_sequence[0]}
            return 201, {"id": self.next_job_id}
        return 404, {}

    def get_json(self, path):
        self.gets.append(path)
        if not self.available:
            raise GmapsScraperUnavailable("down")
        if path == "/api/v1/jobs":
            return 200, {"jobs": list(self.jobs.keys())}
        if path.startswith("/api/v1/jobs/"):
            jid = path.rsplit("/", 1)[-1]
            payload = self.jobs.get(jid)
            if payload is None:
                return 404, {}
            return 200, payload
        return 404, {}

    def get_bytes(self, path):
        self.bytes_gets.append(path)
        if self.fail_download:
            return 500, b""
        return 200, self.csv_payload


def _client(transport=None, **kwargs):
    return GmapsClient(
        base_url="http://test:8080",
        transport=transport or FakeTransport(),
        poll_interval=0.01,
        timeout_seconds=5,
        sleep=lambda _s: None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_is_available_true_when_server_responds():
    t = FakeTransport()
    assert _client(t).is_available() is True


def test_is_available_false_when_unreachable():
    t = FakeTransport()
    t.available = False
    assert _client(t).is_available() is False


# ---------------------------------------------------------------------------
# Create job
# ---------------------------------------------------------------------------

def test_create_job_posts_expected_body():
    t = FakeTransport()
    c = _client(t)
    job_id = c.create_job(
        keywords=["pizza shops in Lahore"], name="lead-run",
        lang="en", depth=1, email=False,
    )
    assert job_id == "job-1"
    path, body = t.posts[0]
    assert path == "/api/v1/jobs"
    assert body["name"] == "lead-run"
    assert body["keywords"] == ["pizza shops in Lahore"]
    assert body["lang"] == "en"
    assert body["depth"] == 1
    assert body["email"] is False
    assert body["max_time"] == 15

def test_create_job_rejects_empty_keywords():
    c = _client()
    with pytest.raises(ValueError):
        c.create_job(keywords=[])


def test_create_job_raises_on_http_error():
    t = FakeTransport()
    t.fail_create = True
    c = _client(t)
    with pytest.raises(GmapsError):
        c.create_job(keywords=["x"])


@pytest.mark.parametrize("key", ["id", "ID", "job_id", "jobId"])
def test_create_job_accepts_alt_id_keys(key):
    class AltIdTransport(FakeTransport):
        def post_json(self, path, body):
            self.posts.append((path, body))
            return 201, {key: "abc"}
    c = _client(AltIdTransport())
    assert c.create_job(keywords=["x"]) == "abc"


def test_poll_job_accepts_capitalized_status():
    """gosom's real API returns 'Status' (capital S); must still work."""
    t = FakeTransport()
    t.jobs["job-1"] = {"ID": "job-1", "Status": "ok", "Name": "x"}
    c = _client(t)
    assert c.poll_job("job-1").succeeded is True


def test_poll_job_capitalized_failed_status():
    t = FakeTransport()
    t.jobs["job-1"] = {"ID": "job-1", "Status": "failed", "Name": "x"}
    c = _client(t)
    with pytest.raises(GmapsJobFailed):
        c.poll_job("job-1")


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def test_poll_job_returns_when_done():
    t = FakeTransport()
    c = _client(t)
    c.create_job(keywords=["x"])
    result = c.poll_job("job-1")
    assert result.succeeded is True
    assert result.status == "ok"


def test_poll_job_raises_on_failed_status():
    t = FakeTransport()
    t.jobs["job-1"] = {"status": "failed", "error": "blocked"}
    c = _client(t)
    with pytest.raises(GmapsJobFailed):
        c.poll_job("job-1")


def test_poll_job_times_out_cleanly():
    t = FakeTransport()
    t.jobs["job-1"] = {"status": "working"}   # never finishes
    # clock that jumps past timeout on the second call
    ticks = iter([0.0, 100.0, 200.0])
    c = GmapsClient(
        base_url="http://test:8080", transport=t,
        poll_interval=0.01, timeout_seconds=10,
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
    )
    result = c.poll_job("job-1")
    assert result.status == "timeout"
    assert result.succeeded is False


def test_poll_job_treats_extra_done_aliases():
    for alias in ("completed", "success", "done", "finished"):
        t = FakeTransport()
        t.jobs["job-1"] = {"status": alias}
        c = _client(t)
        assert c.poll_job("job-1").succeeded is True


# ---------------------------------------------------------------------------
# Download CSV
# ---------------------------------------------------------------------------

def test_download_csv_parses_rows():
    t = FakeTransport()
    t.csv_payload = (
        "input_id,title,address,phone,website\n"
        "1,Pizza Palace,Main St,03001234567,https://palace.example\n"
        "1,Pizza Corner,Side St,03007654321,\n"
    ).encode("utf-8")
    c = _client(t)
    rows = c.download_csv("job-1")
    assert len(rows) == 2
    assert rows[0]["title"] == "Pizza Palace"
    assert rows[1]["website"] == ""


def test_download_csv_empty_payload_returns_empty_list():
    t = FakeTransport()
    c = _client(t)
    assert c.download_csv("job-1") == []


# ---------------------------------------------------------------------------
# scrape() — full flow
# ---------------------------------------------------------------------------

def test_scrape_full_flow_downloads_csv():
    t = FakeTransport()
    t.csv_payload = (
        "title,website,phone\nA,,0300\nB,https://b.example,0301\n"
    ).encode("utf-8")
    c = _client(t)
    result = c.scrape(keywords=["pizza shops in Lahore"])
    assert result.succeeded
    assert len(result.records) == 2


def test_scrape_raises_when_server_unavailable():
    t = FakeTransport()
    t.available = False
    c = _client(t)
    with pytest.raises(GmapsScraperUnavailable):
        c.scrape(keywords=["x"])


def test_scrape_uses_inline_records_when_present():
    t = FakeTransport()
    t.jobs["job-1"] = {
        "status": "ok",
        "results": [{"title": "A"}, {"title": "B"}],
    }
    c = _client(t)
    result = c.scrape(keywords=["x"])
    assert len(result.records) == 2
    # no download needed
    assert t.bytes_gets == []


# ---------------------------------------------------------------------------
# filter_no_website
# ---------------------------------------------------------------------------

def test_filter_no_website_keeps_only_empty_site():
    records = [
        {"title": "A", "website": "https://a.example"},
        {"title": "B", "website": ""},
        {"title": "C"},                       # missing key
        {"title": "D", "website": None},
        {"title": "E", "website": "   "},     # whitespace only
    ]
    out = filter_no_website(records)
    titles = [r["title"] for r in out]
    assert titles == ["B", "C", "D", "E"]


def test_filter_no_website_empty_input():
    assert filter_no_website([]) == []


# ---------------------------------------------------------------------------
# _extract_inline_records
# ---------------------------------------------------------------------------

def test_extract_inline_records_variants():
    assert _extract_inline_records({"results": [{"a": 1}]}) == [{"a": 1}]
    assert _extract_inline_records({"data": [{"b": 2}]}) == [{"b": 2}]
    assert _extract_inline_records({"places": [{"c": 3}]}) == [{"c": 3}]
    assert _extract_inline_records({"unrelated": True}) == []
    assert _extract_inline_records({"results": "not a list"}) == []