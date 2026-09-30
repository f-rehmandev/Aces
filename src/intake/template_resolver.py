"""
Template input resolver — spec §10A.3.

Bridges the Template registry to the intake dispatcher. A user (or API
caller) submits a dict naming a template plus parameter values; this
resolver looks the template up, builds a TaskSpec, and returns it in
the same `InputResolution` shape every other input mode uses.

Input dict shape:

    {
        # One of these identifies the template:
        "template_id": "<uuid>",             # exact id  OR
        "template_name": "pizza-lead-gen",   # by name (uses best version)
        "template_version": 2,               # optional, only with name

        # Optional user-supplied values for the template's parameters:
        "parameters": {"city": "Lahore", "count": 20},

        # Optional partial TaskSpec overrides applied on top:
        "overrides": {"quality": {"min_records": 100}},

        # Optional verbatim prompt preserved for audit:
        "prompt": "find pizza shops in Lahore without websites",
    }

Design notes:
    - Templates are looked up by name using the registry's tier
      preference (published > approved > tested > draft > retired).
      Pass `template_version` for an exact pin.
    - By default only approved/published templates resolve. Set
      `require_selectable=False` on the resolver for authoring flows
      where drafts are legitimate.
    - The resolver has no I/O and no side effects — everything is
      delegated to the registry + builder.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.intake.resolver import InputResolution
from src.templates.builder import (
    TemplateBuildError, TemplateBuilder,
)
from src.templates.models import Template
from src.templates.registry import TemplateRegistry


logger = logging.getLogger("intake.template_resolver")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TemplateResolutionError(ValueError):
    """Raised when a template request can't be turned into a TaskSpec."""


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TemplateResolver:
    """
    Resolves a template reference dict into an InputResolution.

    See module docstring for the input shape.
    """

    def __init__(
        self,
        registry: TemplateRegistry,
        require_selectable: bool = True,
    ):
        self.registry = registry
        self.builder = TemplateBuilder(require_selectable=require_selectable)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, raw: Any) -> InputResolution:
        if not isinstance(raw, dict):
            raise TemplateResolutionError(
                f"template input must be a dict, "
                f"got {type(raw).__name__}"
            )

        template = self._lookup_template(raw)

        parameters = raw.get("parameters") or {}
        overrides = raw.get("overrides") or {}
        prompt = raw.get("prompt") or ""

        if not isinstance(parameters, dict):
            raise TemplateResolutionError(
                f"'parameters' must be a dict, "
                f"got {type(parameters).__name__}"
            )
        if not isinstance(overrides, dict):
            raise TemplateResolutionError(
                f"'overrides' must be a dict, "
                f"got {type(overrides).__name__}"
            )
        if not isinstance(prompt, str):
            raise TemplateResolutionError(
                f"'prompt' must be a string, got {type(prompt).__name__}"
            )

        try:
            spec = self.builder.build(
                template,
                parameters=parameters,
                overrides=overrides,
                natural_language_prompt=prompt,
            )
        except TemplateBuildError as e:
            raise TemplateResolutionError(
                f"could not build TaskSpec from template "
                f"{template.name!r} v{template.version}: {e}"
            ) from e

        description = (
            f"template: {template.name!r} v{template.version} "
            f"({template.status.value})"
        )
        warnings: list[str] = []
        if not prompt:
            warnings.append(
                "no natural-language prompt provided; the run's audit "
                "trail will show an empty prompt field"
            )

        return InputResolution(
            spec=spec,
            source_description=description,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def _lookup_template(self, raw: dict) -> Template:
        template_id = raw.get("template_id")
        if template_id:
            if not isinstance(template_id, str):
                raise TemplateResolutionError(
                    f"'template_id' must be a string, "
                    f"got {type(template_id).__name__}"
                )
            template = self.registry.get(template_id)
            if template is None:
                raise TemplateResolutionError(
                    f"no template with id {template_id!r}"
                )
            return template

        name = raw.get("template_name")
        if not name:
            raise TemplateResolutionError(
                "template input must include either 'template_id' or "
                "'template_name'"
            )
        if not isinstance(name, str):
            raise TemplateResolutionError(
                f"'template_name' must be a string, "
                f"got {type(name).__name__}"
            )

        version = raw.get("template_version")
        if version is not None and not isinstance(version, int):
            raise TemplateResolutionError(
                f"'template_version' must be an int, "
                f"got {type(version).__name__}"
            )

        template = self.registry.get_by_name(name, version=version)
        if template is None:
            if version is not None:
                raise TemplateResolutionError(
                    f"no template named {name!r} at version {version}"
                )
            raise TemplateResolutionError(
                f"no template named {name!r}"
            )
        return template


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def resolve_template(
    raw: Any,
    registry: TemplateRegistry,
    require_selectable: bool = True,
) -> InputResolution:
    return TemplateResolver(
        registry, require_selectable=require_selectable,
    ).resolve(raw)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.templates.models import (
        InputParameter, ParameterType, TaskArchetype, TemplateStatus,
    )

    reg = TemplateRegistry()

    # Registered published template
    t = Template(
        name="pizza-lead-gen",
        archetype=TaskArchetype.LEAD_GENERATION,
        status=TemplateStatus.PUBLISHED,
        input_parameters=[
            InputParameter(name="city", required=True),
            InputParameter(
                name="count", type=ParameterType.INTEGER, default=20,
            ),
        ],
        task_spec_defaults={
            "objective": "lead_gen",
            "target": {"source_hint": "pizza shops in {city}"},
            "constraints": {"geography": "{city}"},
            "quality": {"min_records": "{count}"},
        },
    )
    reg.register(t)
    # Transition through the lifecycle to satisfy the transition rules
    # (register leaves it as draft; manually walk it forward).
    # Note: register() stores the object by reference, so mutating the
    # status field directly is fine for a smoke test.
    t.status = TemplateStatus.PUBLISHED

    resolver = TemplateResolver(reg)

    # ---- Lookup by id ----
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
        "prompt": "find pizza shops in Lahore",
    })
    assert r.spec.objective == "lead_gen"
    assert r.spec.constraints.geography == "Lahore"
    assert r.spec.quality.min_records == 20
    assert r.spec.natural_language_prompt == "find pizza shops in Lahore"
    assert "pizza-lead-gen" in r.source_description
    assert r.warnings == []

    # ---- Lookup by name ----
    r = resolver.resolve({
        "template_name": "pizza-lead-gen",
        "parameters": {"city": "Karachi", "count": 50},
    })
    assert r.spec.constraints.geography == "Karachi"
    assert r.spec.quality.min_records == 50
    # No prompt → warning
    assert any("prompt" in w.lower() for w in r.warnings)

    # ---- Lookup by name + version ----
    r = resolver.resolve({
        "template_name": "pizza-lead-gen",
        "template_version": 1,
        "parameters": {"city": "Quetta"},
    })
    assert r.spec.constraints.geography == "Quetta"

    # ---- Overrides ----
    r = resolver.resolve({
        "template_id": t.template_id,
        "parameters": {"city": "Lahore"},
        "overrides": {"quality": {"min_records": 999}},
    })
    assert r.spec.quality.min_records == 999

    # ---- Missing identifier ----
    try:
        resolver.resolve({"parameters": {}})
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "template_id" in str(e) or "template_name" in str(e)

    # ---- Unknown id ----
    try:
        resolver.resolve({"template_id": "nope"})
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "nope" in str(e)

    # ---- Unknown name ----
    try:
        resolver.resolve({"template_name": "ghost"})
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "ghost" in str(e)

    # ---- Unknown name + version ----
    try:
        resolver.resolve({"template_name": "ghost", "template_version": 5})
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "ghost" in str(e) and "5" in str(e)

    # ---- Non-dict input ----
    try:
        resolver.resolve("not a dict")
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError:
        pass

    # ---- Wrong-typed parameters field ----
    try:
        resolver.resolve({
            "template_id": t.template_id,
            "parameters": "not a dict",
        })
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "parameters" in str(e)

    # ---- Missing required parameter surfaces as build error ----
    try:
        resolver.resolve({"template_id": t.template_id, "parameters": {}})
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "city" in str(e)

    # ---- Draft template rejected by default ----
    draft = Template(
        name="draft", status=TemplateStatus.DRAFT,
        task_spec_defaults={"objective": "extract"},
    )
    reg.register(draft)
    try:
        resolver.resolve({"template_id": draft.template_id})
        raise AssertionError("expected TemplateResolutionError")
    except TemplateResolutionError as e:
        assert "approved" in str(e) or "published" in str(e)

    # ---- Draft allowed when require_selectable=False ----
    lenient = TemplateResolver(reg, require_selectable=False)
    r = lenient.resolve({"template_id": draft.template_id})
    assert r.spec.objective == "extract"

    print("TemplateResolver OK.")