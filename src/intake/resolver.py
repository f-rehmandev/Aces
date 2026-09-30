"""
Input-mode resolvers — spec §10.

Every input mode (natural language, URL list, sitemap, saved task, ...)
resolves into a single `TaskSpec` so downstream stages don't care where
the request came from.

This module defines:
    - the `InputResolver` protocol
    - `InputResolution` (result object)
    - a first concrete resolver: `UrlListResolver` (CSV / TXT / newline)

Later rounds add: sitemap, inline multi-URL, JSON task-file. DB-backed
resolvers wait until Part IX when the tasks/task_versions tables exist.
"""

from __future__ import annotations
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Union

from src.core.task_spec import TaskSpec
from src.navigation.url_normalizer import normalize_url


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class InputResolution:
    """The output of any resolver."""
    spec: TaskSpec
    source_description: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class InputResolver(Protocol):
    """Everything that turns an input into a TaskSpec implements this."""

    def resolve(self, raw: object) -> InputResolution: ...


# ---------------------------------------------------------------------------
# URL list resolver
# ---------------------------------------------------------------------------

class UrlListResolver:
    """
    Reads a list of URLs from:
        - a file path (.csv, .txt, or any extension treated as plain text)
        - a raw string containing newline-separated URLs
        - an iterable of URL strings

    CSV: uses a column literally named "url" or "urls" if present;
         otherwise uses the first column.
    TXT: one URL per line; blank lines and lines starting with "#" are skipped.

    URLs are canonicalized via normalize_url (spec §14.2), then deduplicated
    while preserving first-seen order.
    """

    def resolve(self, raw: Union[str, Path, list[str]]) -> InputResolution:
        warnings: list[str] = []
        lines_or_rows: list[str] = []
        source_description = ""

        # --- iterable of URLs ---
        if isinstance(raw, (list, tuple)):
            lines_or_rows = [str(x) for x in raw]
            source_description = f"inline URL list ({len(raw)} entries)"

        # --- path ---
        elif isinstance(raw, Path) or (
            isinstance(raw, str) and _looks_like_path(raw)
        ):
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"URL list file not found: {path}")
            source_description = f"URL list file: {path}"
            if path.suffix.lower() == ".csv":
                lines_or_rows = _read_csv_column(path, warnings)
            else:
                lines_or_rows = _read_text_lines(path)

        # --- raw multiline string ---
        elif isinstance(raw, str):
            lines_or_rows = _split_lines(raw)
            source_description = f"inline URL string ({len(lines_or_rows)} entries)"

        else:
            raise TypeError(f"UrlListResolver cannot handle {type(raw).__name__}")

        # --- canonicalize + dedupe ---
        seen: set[str] = set()
        urls: list[str] = []
        dropped_invalid = 0
        for candidate in lines_or_rows:
            candidate = candidate.strip()
            if not candidate:
                continue
            result = normalize_url(candidate)
            normalized = result.normalized
            if not normalized or not normalized.startswith(("http://", "https://")):
                dropped_invalid += 1
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)

        if dropped_invalid:
            warnings.append(f"dropped {dropped_invalid} non-URL / invalid entries")

        if not urls:
            warnings.append("no valid URLs found in input")

        spec = TaskSpec(
            natural_language_prompt=source_description,
            target=_target_with_urls(urls),
        )
        return InputResolution(
            spec=spec,
            source_description=source_description,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_path(s: str) -> bool:
    """Heuristic: a short string with a known extension is likely a path."""
    if "\n" in s:
        return False
    if len(s) > 260:
        return False
    return s.endswith((".csv", ".txt", ".list", ".urls"))


def _read_text_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        text = f.read()
    return _split_lines(text)


def _split_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return out


def _read_csv_column(path: Path, warnings: list[str]) -> list[str]:
    """
    Read a CSV file. Prefer a column named url/urls; otherwise use the
    first column. Header row is assumed present if the first row's values
    look like labels (any non-URL token).
    """
    with path.open(encoding="utf-8", newline="") as f:
        text = f.read()
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if not rows:
        return []

    header = [c.strip().lower() for c in rows[0]]
    url_col = None
    for name in ("url", "urls", "link", "links", "href"):
        if name in header:
            url_col = header.index(name)
            break

    # No named column → assume first column and treat row 0 as header if it
    # doesn't look like a URL.
    if url_col is None:
        url_col = 0
        if not rows[0][url_col].strip().lower().startswith(("http://", "https://")):
            rows = rows[1:]
    else:
        rows = rows[1:]  # skip header

    out = []
    for r in rows:
        if len(r) <= url_col:
            continue
        out.append(r[url_col])
    return out


def _target_with_urls(urls: list[str]):
    """Small helper to keep the resolver body readable."""
    from src.core.task_spec import Target
    return Target(start_urls=urls)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- list input ---
    r = UrlListResolver()
    res = r.resolve([
        "https://x.com/a",
        "HTTPS://X.com/b?utm_source=foo#frag",
        "https://x.com/a",              # duplicate after normalization
        "not a url",
    ])
    assert res.spec.target.start_urls == [
        "https://x.com/a",
        "https://x.com/b",
    ], res.spec.target.start_urls
    assert any("non-URL" in w for w in res.warnings)

    # --- text string input ---
    text = """
    # comment line
    https://x.com/p1
    https://x.com/p2

    https://x.com/p1
    """
    res = r.resolve(text)
    assert res.spec.target.start_urls == ["https://x.com/p1", "https://x.com/p2"]

    # --- CSV input ---
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8", newline="") as f:
        f.write("name,url\n")
        f.write("A,https://x.com/a\n")
        f.write("B,https://x.com/b\n")
        csv_path = f.name
    try:
        res = r.resolve(csv_path)
        assert res.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]
    finally:
        os.unlink(csv_path)

    print("UrlListResolver OK.")