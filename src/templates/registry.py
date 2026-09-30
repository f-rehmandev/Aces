"""
Template registry — spec §10A.2.

In-memory store for templates, indexed by both `template_id` and
`(name, version)`. Enforces:

    - Versioning: publishing creates an immutable snapshot; edits
      produce a new version via `new_version()`.
    - Status transitions: only the documented moves are allowed
      (draft → tested → approved → published → retired).
    - Immutability: a published template's fields can't be mutated
      in place. `update()` raises if the target is published; the
      caller is expected to call `new_version()` instead.

Persistence is deliberately not part of this module. The Supabase
schema for templates is a separate migration; when it lands, a thin
adapter can wrap this registry the same way `LeadMemoryStore` and
`CheckpointStore` are wrapped today.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.templates.models import (
    TaskArchetype, Template, TemplateStatus,
)


logger = logging.getLogger("templates.registry")


def _utc_now_iso() -> str:
    # Millisecond resolution — second-resolution timestamps made
    # rapid transitions within the same second indistinguishable,
    # which broke both audit trail ordering and update-detection.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Allowed status transitions
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[TemplateStatus, set[TemplateStatus]] = {
    TemplateStatus.DRAFT:     {TemplateStatus.TESTED, TemplateStatus.RETIRED},
    TemplateStatus.TESTED:    {TemplateStatus.APPROVED, TemplateStatus.DRAFT,
                                TemplateStatus.RETIRED},
    TemplateStatus.APPROVED:  {TemplateStatus.PUBLISHED, TemplateStatus.RETIRED},
    TemplateStatus.PUBLISHED: {TemplateStatus.RETIRED},
    TemplateStatus.RETIRED:   set(),
}


def can_transition(from_status: TemplateStatus, to_status: TemplateStatus) -> bool:
    if from_status == to_status:
        return True
    return to_status in _ALLOWED_TRANSITIONS.get(from_status, set())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TemplateRegistryError(Exception):
    """Base class for registry errors."""


class DuplicateTemplateError(TemplateRegistryError):
    """A template with this (name, version) already exists."""


class TemplateNotFoundError(TemplateRegistryError):
    """No template matches the given id or (name, version)."""


class ImmutableTemplateError(TemplateRegistryError):
    """Attempted to mutate a published template."""


class InvalidTransitionError(TemplateRegistryError):
    """Attempted a status transition that isn't allowed."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TemplateRegistry:
    """
    Holds every registered template, in memory.

    Two indexes:
        _by_id            : template_id   -> Template
        _versions_by_name : name          -> {version: template_id}
    """

    def __init__(self):
        self._by_id: dict[str, Template] = {}
        # name -> {version_int: template_id}
        self._versions_by_name: dict[str, dict[int, str]] = {}
        # template_id -> (name, version, status)
        #
        # Because register() stores the caller's Template object directly
        # (so `reg.get(id) is t` holds), a caller who mutates `t.name` or
        # `t.status` mutates the stored object too. This snapshot of the
        # identity fields lets update() detect that the caller violated
        # the contract even though the two objects are identical.
        self._identity_snapshots: dict[str, tuple[str, int, TemplateStatus]] = {}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def register(self, template: Template) -> None:
        """
        Add a new template. Rejects duplicate (name, version).
        Does not allow silent overwrite — use `new_version()` for that.
        """
        if template.template_id in self._by_id:
            raise DuplicateTemplateError(
                f"template_id {template.template_id!r} already registered"
            )

        name_versions = self._versions_by_name.setdefault(template.name, {})
        if template.version in name_versions:
            existing_id = name_versions[template.version]
            raise DuplicateTemplateError(
                f"template {template.name!r} version {template.version} "
                f"already exists (id={existing_id!r})"
            )

        # Validate every input parameter (fast, catches author mistakes)
        for p in template.input_parameters:
            ok, reason = p.validates()
            if not ok:
                raise TemplateRegistryError(
                    f"invalid parameter on template {template.name!r}: {reason}"
                )

        self._by_id[template.template_id] = template
        name_versions[template.version] = template.template_id
        self._identity_snapshots[template.template_id] = (
            template.name, template.version, template.status,
        )

    def update(self, template: Template) -> None:
        """
        Replace a non-published template's fields in place.

        Raises if the stored template is published, or if any identity
        field (name, version, status) has changed since the template was
        registered. Identity changes go through `new_version()` and
        `transition_status()`.
        """
        existing = self._by_id.get(template.template_id)
        if existing is None:
            raise TemplateNotFoundError(
                f"template_id {template.template_id!r} not registered"
            )

        snap = self._identity_snapshots.get(template.template_id)
        if snap is None:
            # Shouldn't be possible given register() always writes one,
            # but be defensive rather than crashing later.
            raise TemplateRegistryError(
                f"template {template.template_id!r} has no identity snapshot"
            )
        snap_name, snap_version, snap_status = snap

        if snap_status == TemplateStatus.PUBLISHED:
            raise ImmutableTemplateError(
                f"template {snap_name!r} v{snap_version} is published and "
                f"cannot be edited in place; use new_version() to create "
                f"an editable copy"
            )
        if template.status != snap_status:
            raise TemplateRegistryError(
                f"use transition_status() to change status "
                f"({snap_status.value} → {template.status.value})"
            )
        if template.name != snap_name or template.version != snap_version:
            raise TemplateRegistryError(
                "name and version are identity fields; use new_version() "
                "to change them"
            )

        template.updated_at = _utc_now_iso()
        self._by_id[template.template_id] = template
        # Refresh the snapshot — updated_at has changed but identity hasn't.
        self._identity_snapshots[template.template_id] = (
            template.name, template.version, template.status,
        )

    def transition_status(
        self,
        template_id: str,
        new_status: TemplateStatus,
    ) -> Template:
        """
        Move a template between lifecycle states. Only the allowed
        transitions in `_ALLOWED_TRANSITIONS` are accepted.
        """
        template = self._by_id.get(template_id)
        if template is None:
            raise TemplateNotFoundError(f"template_id {template_id!r} not found")

        if not can_transition(template.status, new_status):
            raise InvalidTransitionError(
                f"cannot transition {template.name!r} from "
                f"{template.status.value} to {new_status.value}"
            )

        template.status = new_status
        template.updated_at = _utc_now_iso()
        self._identity_snapshots[template.template_id] = (
            template.name, template.version, template.status,
        )
        return template

    def new_version(self, template_id: str) -> Template:
        """
        Create a fresh DRAFT clone of a template with `version + 1`.

        The original stays untouched (including its status). The clone
        carries a `parent_template_id` pointer back to the original.
        """
        original = self._by_id.get(template_id)
        if original is None:
            raise TemplateNotFoundError(f"template_id {template_id!r} not found")

        next_version = original.version + 1
        name_versions = self._versions_by_name.get(original.name, {})
        if next_version in name_versions:
            raise DuplicateTemplateError(
                f"template {original.name!r} already has a v{next_version}"
            )

        clone_dict = original.to_dict()
        # Reset identity and lifecycle
        import uuid
        clone_dict["template_id"] = str(uuid.uuid4())
        clone_dict["version"] = next_version
        clone_dict["status"] = TemplateStatus.DRAFT.value
        clone_dict["parent_template_id"] = original.template_id
        clone_dict["created_at"] = _utc_now_iso()
        clone_dict["updated_at"] = clone_dict["created_at"]

        clone = Template.from_dict(clone_dict)
        self.register(clone)
        return clone

    def delete(self, template_id: str) -> bool:
        """
        Remove a template. Refuses to delete published templates
        (retire them first). Returns True if removed, False otherwise.
        """
        template = self._by_id.get(template_id)
        if template is None:
            return False
        if template.is_immutable():
            raise ImmutableTemplateError(
                f"cannot delete published template {template.name!r} "
                f"v{template.version}; retire it first"
            )

        del self._by_id[template_id]
        self._identity_snapshots.pop(template_id, None)
        versions = self._versions_by_name.get(template.name, {})
        versions.pop(template.version, None)
        if not versions:
            self._versions_by_name.pop(template.name, None)
        return True
    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get(self, template_id: str) -> Optional[Template]:
        return self._by_id.get(template_id)

    def require(self, template_id: str) -> Template:
        t = self.get(template_id)
        if t is None:
            raise TemplateNotFoundError(f"template_id {template_id!r} not found")
        return t

    def get_by_name(
        self,
        name: str,
        version: Optional[int] = None,
    ) -> Optional[Template]:
        """
        Look up by name.

        If `version` is given, return that exact version (or None).
        Otherwise return the highest-priority version: prefer the
        highest PUBLISHED, then highest APPROVED, then highest overall.
        """
        versions = self._versions_by_name.get(name)
        if not versions:
            return None

        if version is not None:
            tid = versions.get(version)
            return self._by_id.get(tid) if tid else None

        def _highest_in(statuses: set[TemplateStatus]) -> Optional[Template]:
            candidates = [
                self._by_id[tid]
                for tid in versions.values()
                if self._by_id[tid].status in statuses
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda t: t.version)

        for tier in (
            {TemplateStatus.PUBLISHED},
            {TemplateStatus.APPROVED},
            {TemplateStatus.TESTED},
            {TemplateStatus.DRAFT},
            {TemplateStatus.RETIRED},
        ):
            hit = _highest_in(tier)
            if hit is not None:
                return hit

        return None

    def list_all(self) -> list[Template]:
        return list(self._by_id.values())

    def list_by_status(self, status: TemplateStatus) -> list[Template]:
        return [t for t in self._by_id.values() if t.status == status]

    def list_by_archetype(self, archetype: TaskArchetype) -> list[Template]:
        return [t for t in self._by_id.values() if t.archetype == archetype]

    def list_selectable(self) -> list[Template]:
        """Approved or published templates — the ones users can pick."""
        return [t for t in self._by_id.values() if t.is_selectable()]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._by_id


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.templates.models import InputParameter, ParameterType

    reg = TemplateRegistry()

    # Empty
    assert len(reg) == 0
    assert reg.get("nothing") is None
    assert reg.get_by_name("nope") is None

    # Register a draft
    t = Template(
        name="pizza-lead-gen",
        description="Find pizza shops",
        archetype=TaskArchetype.LEAD_GENERATION,
        input_parameters=[InputParameter(name="city")],
    )
    reg.register(t)
    assert len(reg) == 1
    assert reg.get(t.template_id) is t
    assert t.template_id in reg

    # Duplicate id rejected
    try:
        reg.register(t)
        raise AssertionError("expected DuplicateTemplateError")
    except DuplicateTemplateError:
        pass

    # Duplicate (name, version) rejected
    t_dup = Template(name="pizza-lead-gen", version=1)
    try:
        reg.register(t_dup)
        raise AssertionError("expected DuplicateTemplateError")
    except DuplicateTemplateError:
        pass

    # get_by_name picks it up
    assert reg.get_by_name("pizza-lead-gen") is t
    assert reg.get_by_name("pizza-lead-gen", version=1) is t
    assert reg.get_by_name("pizza-lead-gen", version=99) is None

    # List by status and archetype
    assert reg.list_by_status(TemplateStatus.DRAFT) == [t]
    assert reg.list_by_archetype(TaskArchetype.LEAD_GENERATION) == [t]
    assert reg.list_selectable() == []

    # Valid transitions: draft → tested → approved → published
    reg.transition_status(t.template_id, TemplateStatus.TESTED)
    assert t.status == TemplateStatus.TESTED
    reg.transition_status(t.template_id, TemplateStatus.APPROVED)
    assert t.status == TemplateStatus.APPROVED
    reg.transition_status(t.template_id, TemplateStatus.PUBLISHED)
    assert t.status == TemplateStatus.PUBLISHED
    assert t.is_immutable()
    assert reg.list_selectable() == [t]

    # Invalid transition: published → draft
    try:
        reg.transition_status(t.template_id, TemplateStatus.DRAFT)
        raise AssertionError("expected InvalidTransitionError")
    except InvalidTransitionError:
        pass

    # Mutating a published template via update() raises
    try:
        reg.update(t)
        raise AssertionError("expected ImmutableTemplateError")
    except ImmutableTemplateError:
        pass

    # Creating a new version
    v2 = reg.new_version(t.template_id)
    assert v2.version == 2
    assert v2.status == TemplateStatus.DRAFT
    assert v2.parent_template_id == t.template_id
    assert v2.name == t.name
    assert v2.template_id != t.template_id
    assert len(reg) == 2

    # get_by_name without version prefers published
    assert reg.get_by_name("pizza-lead-gen") is t   # v1 is published
    # get_by_name with version returns exact
    assert reg.get_by_name("pizza-lead-gen", version=2) is v2

    # Deleting published raises; deleting draft works
    try:
        reg.delete(t.template_id)
        raise AssertionError("expected ImmutableTemplateError")
    except ImmutableTemplateError:
        pass
    assert reg.delete(v2.template_id) is True
    assert reg.delete(v2.template_id) is False
    assert len(reg) == 1

    # Invalid input parameter rejected at register time
    bad = Template(
        name="bad",
        input_parameters=[InputParameter(name="UPPER")],
    )
    try:
        reg.register(bad)
        raise AssertionError("expected TemplateRegistryError")
    except TemplateRegistryError:
        pass

    print("Template registry OK.")