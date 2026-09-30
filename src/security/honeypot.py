"""
Honeypot detection — spec §49.3.

Three detectors:
    1. Hidden links      — anchors a human can't see (display:none, visibility:hidden,
                           zero-size, off-screen). Attackers plant these so scrapers
                           follow them and either get flagged as bots or get fed
                           fake data.
    2. Hidden form fields — <input type="hidden"> with a name (already stripped by
                            the DOM sanitizer when passing to LLMs, but we still
                            flag them at the page level).
    3. Duplicate-record bursts — a listing page with an unusually high share of
                                 identical records is a strong honeypot signal.

Deferred:
    - Label/position mismatch detection requires DOM geometry (bounding boxes),
      which isn't available from static HTML parsing. Tracked for a future round.

A task that repeatedly hits honeypots on a domain should be flagged for human
review rather than retried in a loop (per §49.3, last paragraph).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass
class HoneypotFinding:
    kind: str                     # "hidden_link" | "hidden_form" | "duplicate_burst"
    description: str
    evidence: str = ""            # short snippet for logs


@dataclass
class HoneypotReport:
    findings: list[HoneypotFinding] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.findings)

    @property
    def suspicious_kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})

    def summary(self) -> str:
        if not self.findings:
            return "no honeypot signals"
        counts = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        parts = [f"{n} × {k}" for k, n in sorted(counts.items())]
        return "honeypot signals: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_INVISIBLE_STYLE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|opacity\s*:\s*0(?:[^.\d]|$)"
    r"|width\s*:\s*0(?:px)?(?:[^.\d]|$)"
    r"|height\s*:\s*0(?:px)?(?:[^.\d]|$)"
    r"|left\s*:\s*-?\d{4,}"
    r"|text-indent\s*:\s*-\d{4,}"
    r"|transform\s*:\s*scale\(\s*0\s*\)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HoneypotDetector:
    """
    Runs the three detectors against HTML and/or a list of records.

    `duplicate_threshold` — a record key appearing this many times or more
    triggers the burst signal (default: 5).
    `duplicate_ratio_threshold` — if >= this fraction of records share any
    single key, trigger (default: 0.5).
    """

    def __init__(
        self,
        duplicate_threshold: int = 5,
        duplicate_ratio_threshold: float = 0.5,
    ):
        self.duplicate_threshold = duplicate_threshold
        self.duplicate_ratio_threshold = duplicate_ratio_threshold

    # ------------------------------------------------------------------
    # HTML-based detectors
    # ------------------------------------------------------------------
    def detect_in_html(self, html: str) -> HoneypotReport:
        if not html:
            return HoneypotReport()
        soup = BeautifulSoup(html, "lxml")
        findings: list[HoneypotFinding] = []
        findings.extend(self._find_hidden_links(soup))
        findings.extend(self._find_hidden_forms(soup))
        return HoneypotReport(findings=findings)

    def _find_hidden_links(self, soup: BeautifulSoup) -> list[HoneypotFinding]:
        findings: list[HoneypotFinding] = []
        for a in soup.find_all("a", href=True):
            style = (a.get("style") or "")
            rel = a.get("rel") or []
            classes = " ".join(a.get("class") or [])

            # Hidden by inline style
            if _INVISIBLE_STYLE.search(style):
                findings.append(HoneypotFinding(
                    kind="hidden_link",
                    description="anchor with inline style hiding it",
                    evidence=_snippet(a),
                ))
                continue

            # Hidden by class name (common honeypot convention)
            if re.search(r"\b(hidden|hide|invisible|offscreen|spam)\b",
                         classes, re.IGNORECASE):
                findings.append(HoneypotFinding(
                    kind="hidden_link",
                    description=f"anchor with hidden-style class: {classes!r}",
                    evidence=_snippet(a),
                ))
                continue

            # rel="nofollow" *and* href pointing somewhere obviously different
            # from the visible text — a common honeypot pattern.
            if "nofollow" in [r.lower() for r in rel]:
                text = (a.get_text() or "").strip()
                href = a["href"]
                # If text looks like a URL and href disagrees, that's suspicious.
                if text.startswith(("http://", "https://")) and text != href:
                    findings.append(HoneypotFinding(
                        kind="hidden_link",
                        description="nofollow anchor whose visible URL differs from target",
                        evidence=_snippet(a),
                    ))
        return findings

    def _find_hidden_forms(self, soup: BeautifulSoup) -> list[HoneypotFinding]:
        findings: list[HoneypotFinding] = []
        for inp in soup.find_all("input", {"type": "hidden"}):
            name = inp.get("name") or "(unnamed)"
            # Plain CSRF tokens and framework state fields are benign —
            # flag only if the name suggests a scraper trap.
            if re.search(r"\b(honeypot|trap|trap_field|bot|spam|email_confirm|"
                         r"do_not_fill|leave_blank)\b",
                         name, re.IGNORECASE):
                findings.append(HoneypotFinding(
                    kind="hidden_form",
                    description=f"suspicious hidden form field: {name!r}",
                    evidence=_snippet(inp),
                ))
        return findings

    # ------------------------------------------------------------------
    # Record-based detector
    # ------------------------------------------------------------------
    def detect_duplicate_burst(
        self,
        records: Iterable[dict],
        key_fn: Callable[[dict], str],
    ) -> HoneypotReport:
        records = list(records or [])
        if not records:
            return HoneypotReport()

        counts: dict[str, int] = {}
        for r in records:
            k = key_fn(r)
            if not k:
                continue
            counts[k] = counts.get(k, 0) + 1

        findings: list[HoneypotFinding] = []
        for k, n in counts.items():
            if n >= self.duplicate_threshold:
                findings.append(HoneypotFinding(
                    kind="duplicate_burst",
                    description=(
                        f"record key {k!r} appears {n} times "
                        f"(threshold {self.duplicate_threshold})"
                    ),
                    evidence=str(k),
                ))
                break   # one finding is enough; the signal is the burst itself

        if not findings and len(records) >= self.duplicate_threshold:
            max_count = max(counts.values()) if counts else 0
            ratio = max_count / len(records)
            if ratio >= self.duplicate_ratio_threshold:
                findings.append(HoneypotFinding(
                    kind="duplicate_burst",
                    description=(
                        f"{ratio:.0%} of records share a single key "
                        f"(ratio threshold {self.duplicate_ratio_threshold:.0%})"
                    ),
                    evidence=f"top key count={max_count} of {len(records)}",
                ))

        return HoneypotReport(findings=findings)

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------
    def detect(
        self,
        html: Optional[str] = None,
        records: Optional[Iterable[dict]] = None,
        key_fn: Optional[Callable[[dict], str]] = None,
    ) -> HoneypotReport:
        combined: list[HoneypotFinding] = []
        if html is not None:
            combined.extend(self.detect_in_html(html).findings)
        if records is not None and key_fn is not None:
            combined.extend(self.detect_duplicate_burst(records, key_fn).findings)
        return HoneypotReport(findings=combined)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snippet(el: Tag, length: int = 120) -> str:
    text = str(el)
    return text if len(text) <= length else text[:length] + "..."


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    html = """
    <html><body>
      <a href="/real">Real link</a>
      <a href="/trap" style="display:none">Hidden trap link</a>
      <a href="/trap2" class="hidden-spam">Class-hidden trap</a>
      <a rel="nofollow" href="https://evil.example/path">https://example.com/different</a>
      <form>
        <input type="hidden" name="csrf_token" value="ok">
        <input type="hidden" name="honeypot" value="">
        <input type="text" name="q">
      </form>
    </body></html>
    """

    detector = HoneypotDetector()
    report = detector.detect_in_html(html)
    print(report.summary())
    for f in report.findings:
        print(f"  [{f.kind}] {f.description}")

    assert report.is_suspicious
    assert "hidden_link" in report.suspicious_kinds
    assert "hidden_form" in report.suspicious_kinds
    # csrf_token must NOT be flagged
    for f in report.findings:
        assert "csrf_token" not in f.description

    # Duplicate burst
    records = [{"title": "Same Book"} for _ in range(10)]
    report = detector.detect_duplicate_burst(records, key_fn=lambda r: r["title"])
    assert report.is_suspicious
    assert report.findings[0].kind == "duplicate_burst"

    # Ratio-based burst
    records = [{"t": "a"}] * 3 + [{"t": "b"}] * 3 + [{"t": "c"}] * 4
    detector2 = HoneypotDetector(duplicate_threshold=5, duplicate_ratio_threshold=0.4)
    report = detector2.detect_duplicate_burst(records, key_fn=lambda r: r["t"])
    assert report.is_suspicious

    # Clean page -> no findings
    clean = '<html><body><a href="/x">x</a><input type="hidden" name="csrf" value="v"></body></html>'
    assert not detector.detect_in_html(clean).is_suspicious

    print("Honeypot detector OK.")