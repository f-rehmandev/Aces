"""
Natural-language input resolver — wraps the existing command parser so the
NL path conforms to the same InputResolver protocol as every other mode
(spec §10).

The parser itself lives in src/assistant_core/command_parser.py and is
unchanged — this is purely an adapter. Injection of `router` lets tests
run without touching a real LLM.
"""

from __future__ import annotations
from typing import Optional

from src.assistant_core.command_parser import parse_command
from src.intake.resolver import InputResolution


class NaturalLanguageResolver:
    def __init__(self, router=None):
        self.router = router

    def resolve(self, raw: str) -> InputResolution:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("NaturalLanguageResolver requires a non-empty string")

        spec = parse_command(raw, router=self.router)

        # Snippet: keep first 80 chars so the description is readable
        snippet = raw.strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."

        warnings = []
        ambiguity = getattr(spec, "ambiguity_note", "")
        if ambiguity:
            warnings.append(f"ambiguous request: {ambiguity}")

        return InputResolution(
            spec=spec,
            source_description=f"natural language: {snippet}",
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Smoke test — uses a fake router so no LLM is called
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    class FakeRouter:
        def call(self, prompt, timeout=None):
            # The parser only needs the JSON shape back.
            return {
                "text": json.dumps({
                    "search_query": "wireless mouse prices",
                    "objective": "compare",
                    "min_records": 5,
                    "fields": ["title", "price"],
                    "filters": "",
                    "location": "",
                    "entity_name": "wireless mouse",
                    "ambiguity_note": "",
                }),
                "provider": "fake",
            }

    r = NaturalLanguageResolver(router=FakeRouter()).resolve(
        "find the best 5 wireless mouse prices"
    )
    print("objective:", r.spec.objective)
    print("field_names:", r.spec.field_names)
    print("source:", r.source_description)
    assert r.spec.objective == "compare"
    assert r.spec.field_names == ["title", "price"]
    assert r.warnings == []

    print("NaturalLanguageResolver OK.")