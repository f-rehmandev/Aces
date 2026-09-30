"""
Selector data model — spec §13.2 (per-selector configuration) and §13.4
(multiple-match modes).

This module defines WHAT a selector is. The engine that EXECUTES selectors
lives in engine.py and consumes these dataclasses.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json
import uuid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SelectorType(str, Enum):
    """§13.1 — the type of extraction this selector performs."""
    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    LINK = "link"
    IMAGE = "image"
    ATTRIBUTE = "attribute"
    HTML = "html"
    TABLE = "table"
    JSONLD = "jsonld"
    EMBEDDED_JSON = "embedded_json"
    META = "meta"
    REGEX = "regex"
    URL = "url"
    PAGINATION = "pagination"
    INTERACTION = "interaction"


class MatchMode(str, Enum):
    """§13.4 — how many matches to keep and how."""
    FIRST = "first"
    ALL = "all"
    NTH = "nth"
    JOIN = "join"


class WaitCondition(str, Enum):
    """§13.2 — the wait_condition field."""
    VISIBLE = "visible"
    ATTACHED = "attached"
    NETWORK_IDLE = "network_idle"
    CUSTOM_JS = "custom_js"
    NONE = "none"


# ---------------------------------------------------------------------------
# Main spec
# ---------------------------------------------------------------------------

@dataclass
class SelectorSpec:
    """
    One node in the extraction tree. Mirrors §13.2 exactly, plus a few
    optional fields the engine needs to record results.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # --- identification ---
    name: str = ""                     # e.g. "price", "title" — what field this selector fills
    parent: Optional[str] = None       # id of the parent selector (for tree context, §13.3)
    type: SelectorType = SelectorType.CSS

    # --- expression ---
    expression: str = ""               # e.g. ".product-card .price" or "//span[@data-price]/text()"
    attribute: Optional[str] = None    # for ATTRIBUTE type: which attribute to read (e.g. "href")

    # --- match behavior (§13.4) ---
    multiple: MatchMode = MatchMode.FIRST
    nth: Optional[int] = None          # used when multiple == NTH
    join_separator: str = " "          # used when multiple == JOIN

    # --- timing / waiting ---
    delay_ms: int = 0
    wait_condition: WaitCondition = WaitCondition.NONE
    timeout_ms: int = 5000
    execution_order: int = 0

    # --- validation & requiredness ---
    required: bool = False
    validation_rule: Optional[str] = None   # free-form for now; structured validators come later
    normalizer: Optional[str] = None        # name of a registered normalizer
    confidence_threshold: float = 0.0

    # --- fallback semantics (§13.5) ---
    fallback_selectors: list["SelectorSpec"] = field(default_factory=list)

    # --- human notes ---
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        d["multiple"] = self.multiple.value
        d["wait_condition"] = self.wait_condition.value
        d["fallback_selectors"] = [f.to_dict() for f in self.fallback_selectors]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Result of running a selector against an HTML document
# ---------------------------------------------------------------------------

@dataclass
class SelectorResult:
    """The outcome of executing one SelectorSpec against a page."""
    selector_id: str
    field_name: str
    success: bool
    value: object = None                       # str | list[str] | None
    match_count: int = 0
    rung_used: int = 1                         # 1 = direct, 2 = first fallback, ...
    selector_used: str = ""                    # the expression that actually matched
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    price = SelectorSpec(
        name="price",
        type=SelectorType.CSS,
        expression=".product-card .price",
        required=True,
        multiple=MatchMode.FIRST,
    )
    fallback = SelectorSpec(
        name="price",
        type=SelectorType.ATTRIBUTE,
        expression="[data-price]",
        attribute="data-price",
    )
    price.fallback_selectors.append(fallback)

    print(price.to_json())
    print("\nSelectorSpec OK.")