"""
Selector tree — spec §13.3 (parent-child context).

Composes SelectorSpec nodes into a Web-Scraper-style tree.

Two modes, chosen automatically at extract time:

    LIST MODE
        Exactly one top-level node, whose multiple == ALL and which has
        children. Its matches become records; its children fill fields.
        Nested ALL-children become nested lists (e.g. reviews in §13.3).

    SINGLE RECORD MODE
        Anything else. One record is produced. Top-level nodes are either
        leaves (fields) or FIRST-match parents whose children fill fields.

Public API:
    tree = SelectorTree()
    tree.add(SelectorSpec(name="product", expression=".card", multiple=MatchMode.ALL))
    tree.add(SelectorSpec(name="title",  expression=".title::text"), parent="product")
    tree.add(SelectorSpec(name="price",  expression=".price::text"), parent="product")
    records = tree.extract(html)     # -> [{"title": ..., "price": ...}, ...]
"""

from __future__ import annotations
from typing import Optional
from bs4 import Tag

from src.selectors.spec import SelectorSpec, MatchMode
from src.selectors.engine import SelectorEngine


class SelectorNode:
    """One node in the tree: a SelectorSpec plus its children and parent link."""

    def __init__(self, spec: SelectorSpec, parent_name: Optional[str] = None):
        self.spec = spec
        self.parent_name = parent_name
        self.children: list["SelectorNode"] = []

    def __repr__(self) -> str:
        return f"<SelectorNode {self.spec.name!r} children={len(self.children)}>"


class SelectorTree:
    """See module docstring for the model."""

    def __init__(self):
        self._top: list[SelectorNode] = []
        self._by_name: dict[str, SelectorNode] = {}
        self._all: list[SelectorNode] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add(
        self,
        spec: SelectorSpec,
        parent: Optional[str] = None,
    ) -> SelectorNode:
        """
        Add a node. `parent` is the name of an existing node, or None for
        a top-level node.
        """
        if not spec.name:
            raise ValueError("SelectorSpec.name is required to add it to a tree")
        if spec.name in self._by_name:
            raise ValueError(f"Duplicate node name: {spec.name!r}")

        node = SelectorNode(spec, parent_name=parent)

        if parent is None:
            self._top.append(node)
        else:
            parent_node = self._by_name.get(parent)
            if parent_node is None:
                raise KeyError(f"No parent named {parent!r}")
            parent_node.children.append(node)

        self._by_name[spec.name] = node
        self._all.append(node)
        return node

    def get(self, name: str) -> Optional[SelectorNode]:
        return self._by_name.get(name)

    def top_level_nodes(self) -> list[SelectorNode]:
        return list(self._top)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract(self, html: str) -> list[dict]:
        """Run the tree against HTML. Returns a list of records."""
        if not self._top:
            return []

        engine = SelectorEngine(html)
        soup = engine.soup

        # Choose mode (§13.3)
        single_top = self._top[0] if len(self._top) == 1 else None
        is_list_mode = (
            single_top is not None
            and single_top.children
            and single_top.spec.multiple == MatchMode.ALL
        )

        if is_list_mode:
            return self._extract_list(single_top, soup, engine)

        return [self._extract_single_record(self._top, soup, engine)]

    # ------------------------------------------------------------------
    # List-mode helper
    # ------------------------------------------------------------------
    def _extract_list(
        self,
        node: SelectorNode,
        scope: Tag,
        engine: SelectorEngine,
    ) -> list[dict]:
        """Run `node` against `scope`, produce one dict per match."""
        result = engine.run(node.spec, scope=scope)
        if not result.success:
            return []

        # Normalize to a list (FIRST returns a single value)
        raw = result.value
        cards = raw if isinstance(raw, list) else [raw]

        records: list[dict] = []
        for card in cards:
            record: dict = {}
            for child in node.children:
                record[child.spec.name] = self._value_for_child(child, card, engine)
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Single-record-mode helper
    # ------------------------------------------------------------------
    def _extract_single_record(
        self,
        nodes: list[SelectorNode],
        scope: Tag,
        engine: SelectorEngine,
    ) -> dict:
        """Merge the top-level nodes into a single record."""
        record: dict = {}
        for node in nodes:
            record[node.spec.name] = self._value_for_child(node, scope, engine)
        return record

    # ------------------------------------------------------------------
    # One child → one value (or nested list, or merged sub-record)
    # ------------------------------------------------------------------
    def _value_for_child(
        self,
        node: SelectorNode,
        scope: Tag,
        engine: SelectorEngine,
    ):
        """
        Compute the value this child contributes to the record built at `scope`.

        - leaf node (no children)                → scalar or list (from engine)
        - ALL-mode inner node                    → list of dicts (nested list)
        - FIRST-mode inner node                  → its own children merged into
                                                   the parent record's dict
        """
        # --- leaf ---
        if not node.children:
            r = engine.run(node.spec, scope=scope)
            return r.value if r.success else None

        # --- inner node ---
        if node.spec.multiple == MatchMode.ALL:
            # Nested list (e.g. reviews inside a product)
            return self._extract_list(node, scope, engine)

        # --- inner FIRST → merge its children into the parent record ---
        r = engine.run(node.spec, scope=scope)
        if not r.success or r.value is None:
            return None

        # r.value is a Tag (or list of one, if a fallback widened it)
        sub_scope = r.value[0] if isinstance(r.value, list) and r.value else r.value
        if not isinstance(sub_scope, Tag):
            return None

        merged: dict = {}
        for child in node.children:
            merged[child.spec.name] = self._value_for_child(child, sub_scope, engine)
        return merged

    # ------------------------------------------------------------------
    # Serialization (for persistence / plan review UI)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"top": [self._node_to_dict(n) for n in self._top]}

    def _node_to_dict(self, node: SelectorNode) -> dict:
        return {
            "spec": node.spec.to_dict(),
            "children": [self._node_to_dict(c) for c in node.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectorTree":
        tree = cls()

        def build(node_data: dict, parent_name: Optional[str]):
            spec = _spec_from_dict(node_data["spec"])
            tree.add(spec, parent=parent_name)
            for child in node_data.get("children", []):
                build(child, parent_name=spec.name)

        for top in data.get("top", []):
            build(top, parent_name=None)
        return tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec_from_dict(d: dict) -> SelectorSpec:
    """Rebuild a SelectorSpec from a plain dict (e.g. loaded from JSON)."""
    from src.selectors.spec import SelectorType, WaitCondition

    spec = SelectorSpec(
        id=d.get("id", SelectorSpec().id),
        name=d.get("name", ""),
        parent=d.get("parent"),
        type=SelectorType(d.get("type", "css")),
        expression=d.get("expression", ""),
        attribute=d.get("attribute"),
        multiple=MatchMode(d.get("multiple", "first")),
        nth=d.get("nth"),
        join_separator=d.get("join_separator", " "),
        delay_ms=d.get("delay_ms", 0),
        wait_condition=WaitCondition(d.get("wait_condition", "none")),
        timeout_ms=d.get("timeout_ms", 5000),
        execution_order=d.get("execution_order", 0),
        required=d.get("required", False),
        validation_rule=d.get("validation_rule"),
        normalizer=d.get("normalizer"),
        confidence_threshold=d.get("confidence_threshold", 0.0),
        notes=d.get("notes", ""),
    )
    spec.fallback_selectors = [_spec_from_dict(f) for f in d.get("fallback_selectors", [])]
    return spec


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    <html><body>
      <div class="card">
        <h2 class="title">Wireless Mouse</h2>
        <span class="price">$24.99</span>
        <a class="detail" href="/p/mouse">View</a>
      </div>
      <div class="card">
        <h2 class="title">Mechanical Keyboard</h2>
        <span class="price">$75.00</span>
        <a class="detail" href="/p/keyboard">View</a>
      </div>
      <div class="card">
        <h2 class="title">USB-C Hub</h2>
        <span class="price">$39.50</span>
        <a class="detail" href="/p/hub">View</a>
      </div>
    </body></html>
    """

    from src.selectors.spec import SelectorType

    tree = SelectorTree()
    tree.add(SelectorSpec(
        name="product", expression=".card", multiple=MatchMode.ALL,
    ))
    tree.add(SelectorSpec(
        name="title", type=SelectorType.CSS,
        expression=".title::text",
    ), parent="product")
    tree.add(SelectorSpec(
        name="price", type=SelectorType.CSS,
        expression=".price::text",
    ), parent="product")
    tree.add(SelectorSpec(
        name="url", type=SelectorType.LINK, expression="a.detail",
    ), parent="product")

    records = tree.extract(sample)
    for r in records:
        print(r)

    assert len(records) == 3
    assert records[0]["title"] == "Wireless Mouse"
    assert records[0]["price"] == "$24.99"
    assert records[0]["url"] == "/p/mouse"

    # Round-trip via dict
    tree2 = SelectorTree.from_dict(tree.to_dict())
    records2 = tree2.extract(sample)
    assert records2 == records

    print("\nSelectorTree OK.")