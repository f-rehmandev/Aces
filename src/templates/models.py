"""
Template data model — spec §10A.

A Template is a reusable, versioned task preset: it carries a name,
description, category, an ordered list of input parameters, default
TaskSpec values, and governance metadata.

Templates are NOT arbitrary code. They cannot execute anything — the
only thing they produce is a TaskSpec, which the existing pipeline
already validates before running anything. §10A.2 is explicit that
imported templates can't run custom code; the safest way to enforce
that is to make a template structurally incapable of doing so.

This module defines WHAT a template is. The registry (in-memory store
with version pinning) lives in `registry.py`. Validation lives in
`validation.py`. Status transitions live in `publishing.py`.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TemplateStatus(str, Enum):
    """
    Lifecycle of a template (§10A.1).

        draft      — created, not yet reviewed or tested
        tested     — passed its test fixtures at least once
        approved   — reviewed and cleared for use
        published  — immutable, available in the shared library
        retired    — no longer selectable for new tasks

    Published versions are IMMUTABLE (§10A.2). To change a published
    template you create a new version; the old one stays frozen.
    """
    DRAFT = "draft"
    TESTED = "tested"
    APPROVED = "approved"
    PUBLISHED = "published"
    RETIRED = "retired"


class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"


class TaskArchetype(str, Enum):
    """
    Broad category of the extraction work. Used for filtering and for
    the UI to group templates.
    """
    ECOMMERCE_LISTING = "ecommerce_listing"
    DIRECTORY = "directory"
    JOB_BOARD = "job_board"
    LEAD_GENERATION = "lead_generation"
    NEWS_ARTICLE = "news_article"
    DOCUMENT_LIST = "document_list"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Input parameter
# ---------------------------------------------------------------------------

_PARAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass
class InputParameter:
    """
    One user-supplied value a template needs before it can build a
    TaskSpec. Example: a "city" parameter for "find pizza shops in
    {city} without websites".
    """
    name: str
    type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = True
    default: Any = None
    choices: list[str] = field(default_factory=list)   # only for ENUM

    def validates(self) -> tuple[bool, str]:
        if not _PARAM_NAME_RE.match(self.name):
            return False, (
                f"parameter name {self.name!r} must be lowercase "
                f"snake_case (a-z, 0-9, _)"
            )
        if self.type == ParameterType.ENUM and not self.choices:
            return False, f"enum parameter {self.name!r} has no choices"
        return True, "ok"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "InputParameter":
        d = dict(data)
        d["type"] = ParameterType(d.get("type", "string"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

@dataclass
class ValidationRules:
    """
    Optional pre-publication requirements for a template. All are
    advisory at task-run time — the real quality gate lives in
    `src/quality/`. These exist so a template can be rejected for
    publication if it doesn't hold up against its test fixtures.
    """
    require_min_fields: int = 1
    require_authorization_declared: bool = False
    forbid_login_walls: bool = True
    max_records_per_run: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationRules":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

@dataclass
class Template:
    """
    A reusable task definition (§10A.1).

    The `task_spec_defaults` dict is merged into a TaskSpec (via
    `TaskSpec.from_dict`) after parameter substitution. Anything a
    template can't express — the user's own natural-language override,
    dynamic URLs, etc. — is applied on top.
    """
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    archetype: TaskArchetype = TaskArchetype.GENERIC

    # Schema compatibility (bumped when TaskSpec changes shape)
    task_schema_version: str = "2.0"

    # Ordered parameters that must be filled before building a TaskSpec
    input_parameters: list[InputParameter] = field(default_factory=list)

    # Default TaskSpec values, e.g.:
    #   {"objective": "lead_gen",
    #    "fields": [{"name": "business_name", "type": "text"}, ...],
    #    "quality": {"min_records": 10},
    #    "constraints": {"geography": "{city}"}}
    task_spec_defaults: dict = field(default_factory=dict)

    # Archetype tags used for discovery/filtering
    supported_archetypes: list[TaskArchetype] = field(default_factory=list)

    # Field names this template guarantees to produce
    required_fields: list[str] = field(default_factory=list)

    # Optional additional rules
    validation_rules: ValidationRules = field(default_factory=ValidationRules)

    # Named security profile ("standard" | "strict" | "lead_gen" | ...)
    security_profile: str = "standard"

    # Fixture IDs used to test the template before approval
    test_fixture_ids: list[str] = field(default_factory=list)

    # Ownership / governance
    author: str = ""                # user id / "system"
    version: int = 1
    status: TemplateStatus = TemplateStatus.DRAFT

    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    # Optional: clone-of pointer for versioned templates
    parent_template_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "archetype": self.archetype.value,
            "task_schema_version": self.task_schema_version,
            "input_parameters": [p.to_dict() for p in self.input_parameters],
            "task_spec_defaults": dict(self.task_spec_defaults),
            "supported_archetypes": [a.value for a in self.supported_archetypes],
            "required_fields": list(self.required_fields),
            "validation_rules": self.validation_rules.to_dict(),
            "security_profile": self.security_profile,
            "test_fixture_ids": list(self.test_fixture_ids),
            "author": self.author,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_template_id": self.parent_template_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Template":
        d = dict(data)

        d["archetype"] = TaskArchetype(d.get("archetype", "generic"))
        d["status"] = TemplateStatus(d.get("status", "draft"))

        if isinstance(d.get("input_parameters"), list):
            d["input_parameters"] = [
                p if isinstance(p, InputParameter) else InputParameter.from_dict(p)
                for p in d["input_parameters"]
            ]

        if isinstance(d.get("supported_archetypes"), list):
            d["supported_archetypes"] = [
                a if isinstance(a, TaskArchetype) else TaskArchetype(a)
                for a in d["supported_archetypes"]
            ]

        if isinstance(d.get("validation_rules"), dict):
            d["validation_rules"] = ValidationRules.from_dict(d["validation_rules"])

        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def is_immutable(self) -> bool:
        """Published templates can't be edited in place (§10A.2)."""
        return self.status == TemplateStatus.PUBLISHED

    def is_selectable(self) -> bool:
        """Published and approved templates can be picked by users."""
        return self.status in (TemplateStatus.APPROVED, TemplateStatus.PUBLISHED)

    def get_parameter(self, name: str) -> Optional[InputParameter]:
        for p in self.input_parameters:
            if p.name == name:
                return p
        return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parameter validation
    ok, _ = InputParameter(name="city").validates()
    assert ok

    ok, reason = InputParameter(name="City").validates()
    assert not ok and "snake_case" in reason

    ok, reason = InputParameter(
        name="sort", type=ParameterType.ENUM,
    ).validates()
    assert not ok and "choices" in reason

    # Empty template
    t = Template()
    assert t.status == TemplateStatus.DRAFT
    assert t.version == 1
    assert not t.is_immutable()
    assert not t.is_selectable()

    # Published templates are immutable + selectable
    t.status = TemplateStatus.PUBLISHED
    assert t.is_immutable()
    assert t.is_selectable()

    # Approved is selectable but not immutable
    t2 = Template(status=TemplateStatus.APPROVED)
    assert t2.is_selectable()
    assert not t2.is_immutable()

    # Round trip with nested structures
    t3 = Template(
        name="pizza-lead-gen",
        description="Find pizza shops without a website",
        category="lead_generation",
        archetype=TaskArchetype.LEAD_GENERATION,
        input_parameters=[
            InputParameter(name="city", description="Target city"),
            InputParameter(name="count", type=ParameterType.INTEGER, default=20),
        ],
        task_spec_defaults={
            "objective": "lead_gen",
            "fields": [{"name": "business_name"}, {"name": "phone"}],
            "constraints": {"geography": "{city}"},
        },
        supported_archetypes=[TaskArchetype.LEAD_GENERATION],
        required_fields=["business_name", "phone"],
        validation_rules=ValidationRules(require_min_fields=2),
        test_fixture_ids=["fixture-1"],
        author="system",
    )
    d = t3.to_dict()
    assert d["status"] == "draft"
    assert d["archetype"] == "lead_generation"
    assert len(d["input_parameters"]) == 2
    assert d["validation_rules"]["require_min_fields"] == 2

    t4 = Template.from_dict(d)
    assert t4.name == "pizza-lead-gen"
    assert t4.archetype == TaskArchetype.LEAD_GENERATION
    assert t4.status == TemplateStatus.DRAFT
    assert len(t4.input_parameters) == 2
    assert t4.input_parameters[1].type == ParameterType.INTEGER
    assert t4.input_parameters[1].default == 20
    assert t4.supported_archetypes == [TaskArchetype.LEAD_GENERATION]
    assert t4.validation_rules.require_min_fields == 2
    assert t4.get_parameter("city") is not None
    assert t4.get_parameter("nope") is None

    print("Template models OK.")