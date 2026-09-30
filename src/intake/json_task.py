"""
JSON task-file loader — spec §10 ("Imported scraper configuration") and
§55 (malicious-import defense).

Loads a JSON file, string, or dict and produces a TaskSpec. Strict schema:
    - Top-level keys must be a subset of the known TaskSpec groups.
    - Unknown keys are rejected unless `strict=False`.
    - Never evaluates expressions, never resolves external refs (§55).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Union

from src.core.task_spec import TaskSpec
from src.intake.resolver import InputResolution


# Keys the TaskSpec dataclass expects. Anything else is unknown.
_KNOWN_TOP_LEVEL_KEYS = set(TaskSpec.__dataclass_fields__.keys())


class MalformedTaskFile(ValueError):
    """The JSON is syntactically valid but doesn't match the schema."""


def _coerce(data: Any) -> dict:
    if not isinstance(data, dict):
        raise MalformedTaskFile(
            f"top-level JSON must be an object, got {type(data).__name__}"
        )
    return data


class JsonTaskFileResolver:
    """
    Loads a TaskSpec from JSON (str, Path, or already-parsed dict).

    `strict=True` (default) rejects any key not part of the TaskSpec schema,
    per §55.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict

    def resolve(self, raw: Union[str, Path, dict]) -> InputResolution:
        warnings: list[str] = []
        source_description = ""

        # --- already-parsed dict: skip the JSON string step ---
        if isinstance(raw, dict):
            data = _coerce(raw)
            source_description = "inline JSON (dict)"

        else:
            # --- obtain a JSON string ---
            if isinstance(raw, Path) or (
                isinstance(raw, str)
                and not raw.lstrip().startswith(("{", "["))
                and Path(raw).exists()
            ):
                path = Path(raw)
                text = path.read_text(encoding="utf-8")
                source_description = f"JSON task file: {path}"
            elif isinstance(raw, str):
                text = raw
                source_description = "inline JSON"
            else:
                raise TypeError(
                    f"JsonTaskFileResolver cannot handle {type(raw).__name__}"
                )

            # --- parse ---
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise MalformedTaskFile(f"invalid JSON: {e}") from e

            data = _coerce(parsed)

        # --- unknown-key check (§55) ---
        unknown = set(data.keys()) - _KNOWN_TOP_LEVEL_KEYS
        if unknown:
            if self.strict:
                raise MalformedTaskFile(
                    f"unknown top-level keys (strict mode): {sorted(unknown)}"
                )
            warnings.append(f"ignored unknown keys: {sorted(unknown)}")

        # --- construct TaskSpec ---
        try:
            spec = TaskSpec.from_dict(data)
        except TypeError as e:
            raise MalformedTaskFile(f"schema mismatch: {e}") from e

        return InputResolution(
            spec=spec,
            source_description=source_description,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    good = json.dumps({
        "natural_language_prompt": "track laptop prices",
        "objective": "monitor",
        "target": {"start_urls": ["https://x.com/a"], "source_hint": "laptops"},
        "fields": [{"name": "title", "type": "text"}],
        "budget": {"max_usd": 1.50},
    })
    r = JsonTaskFileResolver().resolve(good)
    assert r.spec.objective == "monitor"
    assert r.spec.budget.max_usd == 1.50
    print("Loaded spec OK")

    # dict path
    r = JsonTaskFileResolver().resolve({"objective": "monitor"})
    assert r.spec.objective == "monitor"
    assert "dict" in r.source_description.lower()
    print("Dict path OK")

    # unknown key rejected in strict mode
    try:
        JsonTaskFileResolver(strict=True).resolve('{"evil_key": "x"}')
        raise AssertionError("expected MalformedTaskFile")
    except MalformedTaskFile as e:
        print(f"strict reject OK: {e}")

    # unknown key warned in non-strict mode
    r = JsonTaskFileResolver(strict=False).resolve('{"evil_key": "x"}')
    assert any("ignored" in w for w in r.warnings)
    print("non-strict warn OK")

    print("JsonTaskFileResolver OK.")