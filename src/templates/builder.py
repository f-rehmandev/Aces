"""
Template → TaskSpec builder — spec §10A.3.

Turns a Template plus a dict of user-supplied parameter values into a
concrete TaskSpec, ready to feed into the existing pipeline.

Parameter substitution:
    Wherever a string in the template's `task_spec_defaults` contains
    "{param_name}", the value from the user's parameter dict is
    substituted in. Lists and dicts recurse. Non-string scalars pass
    through unchanged.

    Unfilled required parameters are a hard error. Unfilled optional
    parameters with a default use the default. Optional without a
    default and without a provided value remain as literal
    "{param_name}" strings, which the caller is expected to resolve
    (or reject) before running.

Overrides:
    After substitution, `overrides` (a partial TaskSpec dict) is
    deep-merged on top. This lets a caller say "use this template but
    with min_records = 50" without editing the template.

Security:
    The builder is deliberately dumb — it produces data only. It never
    executes template content, never touches the filesystem, and never
    makes network calls. Anything a TaskSpec can't express is out of
    scope by construction.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional

from src.core.task_spec import TaskSpec
from src.templates.models import (
    InputParameter, ParameterType, Template, TemplateStatus,
)


logger = logging.getLogger("templates.builder")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TemplateBuildError(Exception):
    """A template could not be turned into a TaskSpec."""


class MissingParameterError(TemplateBuildError):
    """A required parameter wasn't provided and has no default."""


class InvalidParameterError(TemplateBuildError):
    """A provided parameter value doesn't match its declared type/choices."""


class UnpublishedTemplateError(TemplateBuildError):
    """Only approved or published templates can be built into TaskSpecs."""


# ---------------------------------------------------------------------------
# Parameter substitution
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def _substitute(value: Any, params: dict[str, Any]) -> Any:
    """
    Recursively replace {param_name} placeholders with values from
    `params`. Placeholders without a matching key are left intact.

    Type preservation:
        A string that is ENTIRELY one placeholder — e.g. "{count}" —
        substitutes the raw value, preserving its Python type. So
        integer parameters stay int, booleans stay bool, etc.

        A string that embeds a placeholder inside other text — e.g.
        "pizza shops in {city}" — substitutes the string form, since
        the result is unavoidably a string.
    """
    if isinstance(value, str):
        # Whole-string placeholder → preserve the raw type
        full = _PLACEHOLDER_RE.fullmatch(value)
        if full is not None and full.group(1) in params:
            return params[full.group(1)]

        # Otherwise regex-substitute with string coercion
        def repl(match: re.Match) -> str:
            name = match.group(1)
            if name in params:
                return str(params[name])
            return match.group(0)
        return _PLACEHOLDER_RE.sub(repl, value)
    if isinstance(value, list):
        return [_substitute(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, params) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _coerce_parameter_value(
    parameter: InputParameter,
    raw_value: Any,
) -> Any:
    """Validate and coerce a single user-supplied value."""
    if parameter.type == ParameterType.STRING:
        if not isinstance(raw_value, str):
            raise InvalidParameterError(
                f"parameter {parameter.name!r} expects a string, "
                f"got {type(raw_value).__name__}"
            )
        return raw_value

    if parameter.type == ParameterType.INTEGER:
        if isinstance(raw_value, bool):
            # bool is a subclass of int; reject it explicitly
            raise InvalidParameterError(
                f"parameter {parameter.name!r} expects an integer, got bool"
            )
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, str) and raw_value.strip().lstrip("-").isdigit():
            return int(raw_value)
        raise InvalidParameterError(
            f"parameter {parameter.name!r} expects an integer, "
            f"got {raw_value!r}"
        )

    if parameter.type == ParameterType.FLOAT:
        if isinstance(raw_value, bool):
            raise InvalidParameterError(
                f"parameter {parameter.name!r} expects a number, got bool"
            )
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value)
            except ValueError:
                pass
        raise InvalidParameterError(
            f"parameter {parameter.name!r} expects a number, got {raw_value!r}"
        )

    if parameter.type == ParameterType.BOOLEAN:
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str) and raw_value.lower() in ("true", "false"):
            return raw_value.lower() == "true"
        raise InvalidParameterError(
            f"parameter {parameter.name!r} expects a boolean, got {raw_value!r}"
        )

    if parameter.type == ParameterType.ENUM:
        if not isinstance(raw_value, str):
            raise InvalidParameterError(
                f"parameter {parameter.name!r} expects a string, "
                f"got {type(raw_value).__name__}"
            )
        if raw_value not in parameter.choices:
            raise InvalidParameterError(
                f"parameter {parameter.name!r} must be one of "
                f"{parameter.choices}, got {raw_value!r}"
            )
        return raw_value

    raise InvalidParameterError(
        f"parameter {parameter.name!r} has unknown type {parameter.type!r}"
    )


def _resolve_parameters(
    template: Template,
    provided: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate provided values and fill in defaults. Raise on missing
    required params. Unknown keys in `provided` are ignored (the caller
    may be applying the same parameter dict to several templates).
    """
    resolved: dict[str, Any] = {}

    for p in template.input_parameters:
        if p.name in provided and provided[p.name] is not None:
            resolved[p.name] = _coerce_parameter_value(p, provided[p.name])
            continue

        if p.default is not None:
            resolved[p.name] = p.default
            continue

        if p.required:
            raise MissingParameterError(
                f"parameter {p.name!r} is required by template "
                f"{template.name!r} but no value was provided"
            )

        # Optional, no default — leave as-is; substitution will keep the
        # literal {name} placeholder, and the caller can decide what to
        # do with that.

    return resolved


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay onto base. Dicts merge recursively; everything else replaces."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# Spec-shape validation
# ---------------------------------------------------------------------------

# Keys in TaskSpec that must be dicts (they map to a nested dataclass).
_NESTED_DICT_KEYS = (
    "target", "constraints", "navigation", "source_requirements",
    "quality", "output", "schedule", "compliance", "budget",
)

# Keys in TaskSpec that must be lists of dicts (they map to nested dataclasses).
_NESTED_LIST_KEYS = ("fields", "entities")


def _validate_spec_dict(spec_dict: dict) -> None:
    """
    Fail fast if the pre-from_dict shape is wrong.

    TaskSpec.from_dict tolerates a surprising amount of malformed input
    because it only wraps nested dataclasses when the value is already
    a dict. Passing e.g. {"target": "oops"} produces a TaskSpec whose
    `target` attribute is a string — which explodes later, far from the
    cause. Catching it here gives the template author a precise error.
    """
    if not isinstance(spec_dict, dict):
        raise TemplateBuildError(
            f"template produced a non-dict defaults object: "
            f"{type(spec_dict).__name__}"
        )

    for key in _NESTED_DICT_KEYS:
        if key in spec_dict and not isinstance(spec_dict[key], dict):
            raise TemplateBuildError(
                f"template field {key!r} must be a dict "
                f"(it maps to a nested spec group), "
                f"got {type(spec_dict[key]).__name__}"
            )

    for key in _NESTED_LIST_KEYS:
        if key not in spec_dict:
            continue
        value = spec_dict[key]
        if not isinstance(value, list):
            raise TemplateBuildError(
                f"template field {key!r} must be a list, "
                f"got {type(value).__name__}"
            )
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise TemplateBuildError(
                    f"template field {key!r}[{i}] must be a dict, "
                    f"got {type(item).__name__}"
                )



# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TemplateBuilder:
    """
    Builds TaskSpecs from Templates.

    `require_selectable` — when True (default), refuses to build from
    templates that aren't approved or published. Set to False for
    authoring/testing workflows where draft templates are legitimate.
    """

    def __init__(self, require_selectable: bool = True):
        self.require_selectable = require_selectable

    def build(
        self,
        template: Template,
        parameters: Optional[dict[str, Any]] = None,
        overrides: Optional[dict] = None,
        natural_language_prompt: str = "",
    ) -> TaskSpec:
        """
        Produce a TaskSpec from a template.

        `parameters`     — user-supplied values for the template's input
                           parameters. Missing required → error.
        `overrides`      — partial TaskSpec dict applied on top of the
                           substituted defaults. Useful for one-off tweaks.
        `natural_language_prompt` — preserved verbatim for audit; the
                           template cannot fabricate this on its own.
        """
        if self.require_selectable and not template.is_selectable():
            raise UnpublishedTemplateError(
                f"template {template.name!r} v{template.version} has "
                f"status {template.status.value!r}; only approved or "
                f"published templates can be built"
            )

        resolved = _resolve_parameters(template, parameters or {})

        # Deep-copy so we never mutate the template's defaults
        spec_dict = copy.deepcopy(template.task_spec_defaults)
        spec_dict = _substitute(spec_dict, resolved)

        if overrides:
            spec_dict = _deep_merge(spec_dict, overrides)

        if natural_language_prompt:
            spec_dict["natural_language_prompt"] = natural_language_prompt

        _validate_spec_dict(spec_dict)

        try:
            spec = TaskSpec.from_dict(spec_dict)
        except TypeError as e:
            raise TemplateBuildError(
                f"template {template.name!r} produced an invalid TaskSpec: {e}"
            ) from e

        return spec


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_from_template(
    template: Template,
    parameters: Optional[dict[str, Any]] = None,
    overrides: Optional[dict] = None,
    natural_language_prompt: str = "",
) -> TaskSpec:
    return TemplateBuilder().build(
        template,
        parameters=parameters,
        overrides=overrides,
        natural_language_prompt=natural_language_prompt,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.templates.models import TaskArchetype

    t = Template(
        name="pizza-lead-gen",
        archetype=TaskArchetype.LEAD_GENERATION,
        status=TemplateStatus.PUBLISHED,
        input_parameters=[
            InputParameter(name="city", required=True),
            InputParameter(name="count", type=ParameterType.INTEGER, default=10),
            InputParameter(
                name="sort",
                type=ParameterType.ENUM,
                required=False,
                choices=["rating", "name"],
            ),
        ],
        task_spec_defaults={
            "objective": "lead_gen",
            "target": {"source_hint": "pizza shops in {city}"},
            "fields": [
                {"name": "business_name", "type": "text"},
                {"name": "phone", "type": "phone"},
            ],
            "constraints": {"geography": "{city}"},
            "quality": {"min_records": "{count}"},
        },
    )

    # Happy path
    spec = build_from_template(
        t,
        parameters={"city": "Lahore"},
        natural_language_prompt="find pizza shops in Lahore",
    )
    assert spec.objective == "lead_gen"
    assert spec.target.source_hint == "pizza shops in Lahore"
    assert spec.constraints.geography == "Lahore"
    assert spec.quality.min_records == 10           # default
    assert spec.natural_language_prompt == "find pizza shops in Lahore"
    assert spec.field_names == ["business_name", "phone"]

    # Explicit count overrides the default
    spec2 = build_from_template(
        t, parameters={"city": "Karachi", "count": 50},
    )
    assert spec2.quality.min_records == 50
    assert spec2.constraints.geography == "Karachi"

    # String "50" coerces to int
    spec3 = build_from_template(
        t, parameters={"city": "Lahore", "count": "50"},
    )
    assert spec3.quality.min_records == 50

    # Overrides merge on top
    spec4 = build_from_template(
        t,
        parameters={"city": "Lahore"},
        overrides={"quality": {"min_records": 99}, "output": {"format": "csv"}},
    )
    assert spec4.quality.min_records == 99
    assert spec4.output.format == "csv"
    # Unrelated quality keys preserved
    assert spec4.quality.max_failed_page_pct == 0.3

    # Enum validation
    try:
        build_from_template(
            t, parameters={"city": "Lahore", "sort": "bogus"},
        )
        raise AssertionError("expected InvalidParameterError")
    except InvalidParameterError:
        pass

    # Valid enum
    spec5 = build_from_template(
        t, parameters={"city": "Lahore", "sort": "rating"},
    )
    assert spec5 is not None

    # Missing required parameter
    try:
        build_from_template(t, parameters={})
        raise AssertionError("expected MissingParameterError")
    except MissingParameterError:
        pass

    # Wrong type
    try:
        build_from_template(t, parameters={"city": "Lahore", "count": "many"})
        raise AssertionError("expected InvalidParameterError")
    except InvalidParameterError:
        pass

    # Bool rejected as integer
    try:
        build_from_template(t, parameters={"city": "Lahore", "count": True})
        raise AssertionError("expected InvalidParameterError")
    except InvalidParameterError:
        pass

    # Unpublished template rejected by default
    draft = Template(
        name="draft",
        status=TemplateStatus.DRAFT,
        task_spec_defaults={"objective": "extract"},
    )
    try:
        build_from_template(draft)
        raise AssertionError("expected UnpublishedTemplateError")
    except UnpublishedTemplateError:
        pass

    # But allowed when require_selectable=False
    spec6 = build_from_template(
        draft,
    ) if False else TemplateBuilder(require_selectable=False).build(draft)
    assert spec6.objective == "extract"

    # Template defaults not mutated
    assert "{city}" in t.task_spec_defaults["constraints"]["geography"]

    print("Template builder OK.")