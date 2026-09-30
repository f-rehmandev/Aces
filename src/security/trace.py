"""
Security trace — records what every guard did during one fetch+extract.

This is what makes the security pipeline *observable*: at the end of a
run, we can say exactly how many elements were stripped, which honeypot
signals fired, whether the injection guard nulled any fields, and which
extraction rung produced the data.

Attached optionally to `_scrape_source` calls (see src/assistant.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SecurityTrace:
    url: str = ""

    # SSRF gate
    ssrf_allowed: bool = True
    ssrf_reason: str = ""

    # Honeypot detector
    honeypot_findings: list[str] = field(default_factory=list)

    # DOM sanitizer
    sanitizer_removed: int = 0
    sanitizer_reasons: dict[str, int] = field(default_factory=dict)

    # Extraction + injection guard
    records_extracted: int = 0
    records_after_guard: int = 0
    injection_findings: list[str] = field(default_factory=list)

    # Which extraction rung produced the data (spec §17.1):
    #   "rung1_jsonld" | "rung1_meta" | "rung2_llm" | "rung3_vision" | ""
    extraction_rung: str = ""

        # Fetch-layer provenance
    provider_used: str = ""       # "playwright" | "scraperapi" | "direct"
    fallback_used: bool = False
    block_reason: str = ""

    # Convenience
    @property
    def was_guarded(self) -> bool:
        return (
            not self.ssrf_allowed
            or bool(self.honeypot_findings)
            or self.sanitizer_removed > 0
            or bool(self.injection_findings)
        )

    def summary(self) -> str:
        parts = [f"url={self.url!r}"]
        if not self.ssrf_allowed:
            parts.append(f"SSRF_BLOCKED({self.ssrf_reason})")
        if self.sanitizer_removed:
            parts.append(f"stripped={self.sanitizer_removed}")
        if self.honeypot_findings:
            parts.append(f"honeypot={','.join(self.honeypot_findings)}")
        if self.injection_findings:
            parts.append(f"injection={len(self.injection_findings)}")
        if self.provider_used:
            parts.append(f"provider={self.provider_used}")
        if self.fallback_used:
            parts.append("fallback=yes")
        if self.extraction_rung:
            parts.append(f"rung={self.extraction_rung}")
        if self.block_reason and self.records_after_guard == 0:
            parts.append(f"blocked({self.block_reason})")
        parts.append(f"records={self.records_after_guard}/{self.records_extracted}")
        return " | ".join(parts)