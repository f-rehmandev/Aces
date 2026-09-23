"""
TaskSpec — the single internal representation every input mode resolves to.
Per project knowledge Section 11.2. Deliberately a plain dataclass (not a
framework model) so it stays simple, debuggable, and JSON-serializable for
Supabase storage.
"""

from dataclasses import dataclass, field, asdict
import uuid
import json


@dataclass
class TaskSpec:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1.0"
    natural_language_prompt: str = ""

    # target
    start_urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    source_hint: str = ""

    objective: str = "extract"  # extract | monitor | compare | lead_gen

    # entities
    entity_name: str = ""
    record_boundary_hint: str = ""
    identity_hint: str = ""

    # fields
    fields: list[str] = field(default_factory=list)

    # constraints
    filters: str = ""
    location: str = ""

    # navigation
    pagination: str = "auto"
    max_pages: int = 5

    # quality
    min_records: int = 1

    # output
    output_format: str = "xlsx"

    # client scoping (our own addition, needed for multi-tenant isolation — Part X)
    client_id: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


if __name__ == "__main__":
    spec = TaskSpec(
        natural_language_prompt="find best 3 prices of wireless mouse",
        start_urls=[],
        objective="compare",
        fields=["title", "price"],
    )
    print(spec.to_json())