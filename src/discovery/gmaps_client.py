"""
Google Maps lead provider — client for the local gosom/google-maps-scraper.

The scraper runs as a Docker container exposing a REST API on
http://localhost:8080. We submit a job, poll until it finishes, then
download the CSV and return parsed records.

Design:
    - `Transport` is injected so unit tests never touch the network.
    - The client is a thin HTTP wrapper. Any business logic (e.g. "only
      shops without a website") lives in `filter_no_website()` or later
      in the pipeline.
    - Job status strings from the server are treated tolerantly: any of
      {ok, completed, success, done} means finished; {failed, error}
      means failed; anything else means keep waiting.
    - `GmapsScraperUnavailable` is raised when the server can't be
      reached so the caller can fall back to other sources.
"""

from __future__ import annotations
import csv
import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


logger = logging.getLogger("gmaps_client")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT_SECONDS = 900        # 15 min — enough for a small job
DEFAULT_POLL_INTERVAL = 5.0

_DONE_STATUSES = {"ok", "completed", "success", "done", "finished"}
_FAILED_STATUSES = {"failed", "error"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GmapsError(Exception):
    """Base class for Google Maps provider errors."""


class GmapsScraperUnavailable(GmapsError):
    """The local scraper server can't be reached."""


class GmapsJobFailed(GmapsError):
    """The job ran but the server reported a failure."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GmapsJobResult:
    job_id: str
    status: str
    records: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status.lower() in _DONE_STATUSES and not self.error


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------

class Transport(Protocol):
    """Minimal HTTP surface used by the client. Fake in tests."""

    def post_json(self, path: str, body: dict) -> tuple[int, dict]: ...
    def get_json(self, path: str) -> tuple[int, dict]: ...
    def get_bytes(self, path: str) -> tuple[int, bytes]: ...


class RequestsTransport:
    """Default transport using `requests`."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, body: dict) -> tuple[int, dict]:
        import requests
        try:
            r = requests.post(
                self.base_url + path, json=body, timeout=self.timeout,
            )
        except Exception as e:
            raise GmapsScraperUnavailable(f"POST {path} failed: {e}") from e
        return r.status_code, _safe_json(r)

    def get_json(self, path: str) -> tuple[int, dict]:
        import requests
        try:
            r = requests.get(self.base_url + path, timeout=self.timeout)
        except Exception as e:
            raise GmapsScraperUnavailable(f"GET {path} failed: {e}") from e
        return r.status_code, _safe_json(r)

    def get_bytes(self, path: str) -> tuple[int, bytes]:
        import requests
        try:
            r = requests.get(self.base_url + path, timeout=self.timeout)
        except Exception as e:
            raise GmapsScraperUnavailable(f"GET {path} failed: {e}") from e
        return r.status_code, r.content


def _safe_json(r) -> dict:
    try:
        return r.json() or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GmapsClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        transport: Optional[Transport] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = (
            base_url
            or os.getenv("GMAPS_SCRAPER_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.transport = transport or RequestsTransport(self.base_url)
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------
    # Server health
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """True if the scraper server responds to GET /api/v1/jobs."""
        try:
            status, _ = self.transport.get_json("/api/v1/jobs")
        except GmapsScraperUnavailable:
            return False
        return 200 <= status < 300

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    def create_job(
        self,
        keywords: list[str],
        name: str = "aces-job",
        lang: str = "en",
        depth: int = 1,
        email: bool = False,
        max_time_minutes: int = 15,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Submit a scraping job. Returns the job id.

        `keywords` — one search phrase per entry, e.g.
        ["pizza shops in Lahore"].
        """
        if not keywords:
            raise ValueError("keywords must be a non-empty list")

        body = {
            "name": name,
            "keywords": list(keywords),
            "lang": lang,
            "depth": int(depth),
            "email": bool(email),
            # gosom's API requires a max wall-clock budget per keyword.
            # The server aborts the scrape if this is exceeded and returns
            # whatever it has. 15 minutes is plenty for a small/medium run.
            "max_time": int(max_time_minutes),
        }
        if extra:
            body.update(extra)

        status, data = self.transport.post_json("/api/v1/jobs", body)
        if not (200 <= status < 300):
            raise GmapsError(f"create_job HTTP {status}: {data}")

        # gosom returns capitalized JSON keys ("ID"), so check both cases
        # plus the common snake/camel alternatives.
        job_id = (
            data.get("id")
            or data.get("ID")
            or data.get("job_id")
            or data.get("jobId")
        )
        if not job_id:
            raise GmapsError(f"create_job returned no job id: {data}")
        return str(job_id)

    def get_job(self, job_id: str) -> dict:
        status, data = self.transport.get_json(f"/api/v1/jobs/{job_id}")
        if not (200 <= status < 300):
            raise GmapsError(f"get_job({job_id}) HTTP {status}: {data}")
        return data

    def poll_job(
        self,
        job_id: str,
        timeout_seconds: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> GmapsJobResult:
        """
        Wait until the job finishes or the timeout expires.

        Never raises on timeout — returns a job result with status
        "timeout" and whatever the last observed payload was.
        """
        timeout = timeout_seconds or self.timeout_seconds
        interval = poll_interval or self.poll_interval
        started = self._clock()
        last: dict = {}

        while True:
            try:
                last = self.get_job(job_id)
            except GmapsError as e:
                logger.warning(f"poll_job transient error on {job_id}: {e}")
                last = {}

            # gosom capitalizes "Status"; accept both cases.
            raw_status = last.get("status") or last.get("Status") or ""
            status = str(raw_status).lower()

            if status in _DONE_STATUSES:
                return GmapsJobResult(
                    job_id=job_id, status=status,
                    records=_extract_inline_records(last),
                )
            if status in _FAILED_STATUSES:
                raise GmapsJobFailed(
                    f"job {job_id} failed: {last.get('error', 'unknown error')}"
                )

            if self._clock() - started >= timeout:
                return GmapsJobResult(
                    job_id=job_id, status="timeout",
                    error=f"did not finish within {timeout}s",
                )

            self._sleep(interval)

    def download_csv(self, job_id: str) -> list[dict]:
        """Download the job's CSV and parse into a list of dicts."""
        status, blob = self.transport.get_bytes(
            f"/api/v1/jobs/{job_id}/download"
        )
        if not (200 <= status < 300):
            raise GmapsError(f"download_csv HTTP {status}")
        if not blob:
            return []
        text = blob.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    def delete_job(self, job_id: str) -> None:
        """Best effort — never raises."""
        try:
            # requests-based default transport
            import requests
            requests.delete(
                self.base_url + f"/api/v1/jobs/{job_id}", timeout=10,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Convenience: submit → poll → download
    # ------------------------------------------------------------------
    def scrape(
        self,
        keywords: list[str],
        name: str = "aces-job",
        lang: str = "en",
        depth: int = 1,
        email: bool = False,
        timeout_seconds: Optional[float] = None,
        max_time_minutes: int = 15,
    ) -> GmapsJobResult:
        if not self.is_available():
            raise GmapsScraperUnavailable(
                f"Google Maps scraper not reachable at {self.base_url}"
            )

        job_id = self.create_job(
            keywords=keywords, name=name, lang=lang, depth=depth,
            email=email, max_time_minutes=max_time_minutes,
        )
        result = self.poll_job(job_id, timeout_seconds=timeout_seconds)

        if result.succeeded and not result.records:
            result.records = self.download_csv(job_id)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_inline_records(job_payload: dict) -> list[dict]:
    """
    Some server versions embed results inline in the job payload instead
    of requiring a separate CSV download. Accept both shapes.
    """
    for key in ("results", "data", "places"):
        value = job_payload.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]
    return []


def filter_no_website(records: list[dict]) -> list[dict]:
    """
    Return only the records whose `website` field is empty.

    This is the "give me pizza shops that have no website" filter.
    """
    out: list[dict] = []
    for r in records:
        site = (r.get("website") or "").strip()
        if not site:
            out.append(r)
    return out