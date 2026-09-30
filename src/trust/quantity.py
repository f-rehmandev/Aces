"""
Quantity parsing — extract package size and dose from text like:
    "Panadol 500mg x 200 tablets"
    "100ml bottle"
    "1kg pack"
    "Panadol Extra 24 caplets"

Then compute single-unit price so a "big" pack and a "small" pack can be
compared fairly. A separate `unit_price_basis` column records what the
single-unit price is measured in ("per tablet", "per ml", "per g").

Design:
    - Pure parsing, no LLM. Deterministic and testable.
    - Only returns QuantityInfo when the parse is unambiguous.
    - Never invents data: returns None if the text doesn't say.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Units we recognize
# ---------------------------------------------------------------------------

_COUNT_UNITS = {
    "tablet": "tablet", "tablets": "tablet", "tab": "tablet", "tabs": "tablet",
    "caplet": "caplet", "caplets": "caplet",
    "capsule": "capsule", "capsules": "capsule", "cap": "capsule", "caps": "capsule",
    "pill": "pill", "pills": "pill",
    "sachet": "sachet", "sachets": "sachet",
    "strip": "strip", "strips": "strip",
    "vial": "vial", "vials": "vial",
    "bottle": "bottle", "bottles": "bottle",
    "pack": "pack", "packs": "pack", "packet": "pack",
    "piece": "piece", "pieces": "piece", "pc": "piece", "pcs": "piece",
    "unit": "unit", "units": "unit",
}

_VOLUME_UNITS = {
    "ml": ("ml", 1.0), "milliliter": ("ml", 1.0), "milliliters": ("ml", 1.0),
    "cl": ("ml", 10.0),
    "l": ("ml", 1000.0), "liter": ("ml", 1000.0), "liters": ("ml", 1000.0),
    "litre": ("ml", 1000.0), "litres": ("ml", 1000.0),
    "fl oz": ("ml", 29.5735), "floz": ("ml", 29.5735),
}

_WEIGHT_UNITS = {
    "mg": ("g", 0.001), "milligram": ("g", 0.001), "milligrams": ("g", 0.001),
    "g": ("g", 1.0), "gram": ("g", 1.0), "grams": ("g", 1.0),
    "kg": ("g", 1000.0), "kilogram": ("g", 1000.0), "kilograms": ("g", 1000.0),
    "oz": ("g", 28.3495), "ounce": ("g", 28.3495), "ounces": ("g", 28.3495),
    "lb": ("g", 453.592), "pound": ("g", 453.592), "pounds": ("g", 453.592),
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class QuantityInfo:
    count: float                # how many units (e.g. 200 tablets, 500 ml)
    unit: str                   # "tablet", "ml", "g", "capsule"
    basis: str                  # human-readable: "per tablet", "per ml"
    strength: Optional[str] = None      # "500mg" if present (informational)
    source: str = ""            # which parsing pass found this

    def to_dict(self) -> dict:
        return {
            "package_quantity": self.count,
            "package_unit": self.unit,
            "unit_price_basis": self.basis,
            "dose_strength": self.strength,
        }


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# "500mg" — strength only, no count
_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|ug|iu)\b", re.IGNORECASE
)

# "200 tablets" / "200 x 500mg" / "24 caplets" / "x200"
_COUNT_BEFORE_UNIT_RE = re.compile(
    r"(\d{1,6}(?:[,]\d{3})*)\s*[x×\u00d7]?\s*"
    r"(tablets?|tabs?|caplets?|capsules?|caps?|pills?|sachets?|strips?|"
    r"vials?|bottles?|pack(?:ets?|s)?|pieces?|pcs?|units?)"
    r"\b",
    re.IGNORECASE,
)

# "x 200" / "× 200" after a strength
_MULTIPLIER_RE = re.compile(r"[x×\u00d7]\s*(\d{1,6})", re.IGNORECASE)

# Volume units — always a package quantity.
_VOLUME_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(ml|milliliters?|cl|liters?|litres?|l)"
    r"\b",
    re.IGNORECASE,
)

# Weight units — always a package quantity.
# "g" is listed after longer forms AND without a preceding "m" so that
# "500mg" cannot be matched as "00g". mg is a *strength*, not a package.
_WEIGHT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(kg|kilograms?|grams?|g|oz|ounces?|lb|pounds?)"
    r"\b",
    re.IGNORECASE,
)

# Package word alone with a number, e.g. "pack of 24"
_PACK_OF_RE = re.compile(r"\b(?:pack|box|bottle|jar)\s+of\s+(\d{1,6})", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_quantity(text: Optional[str]) -> Optional[QuantityInfo]:
    """
    Best-effort quantity extraction. Returns None if the text has no
    unambiguous quantity statement.
    """
    if not text or not isinstance(text, str):
        return None

    s = text.strip()

    # Pull the strength out first so "500mg x 200 tablets" doesn't get
    # mis-parsed as "500 x 200".
    strength_match = _STRENGTH_RE.search(s)
    strength = None
    if strength_match:
        strength = f"{strength_match.group(1)}{strength_match.group(2).lower()}"

    # Highest priority: "N <count unit>"
    m = _COUNT_BEFORE_UNIT_RE.search(s)
    if m:
        try:
            count = float(m.group(1).replace(",", ""))
        except ValueError:
            count = None
        if count and count > 0:
            unit_raw = m.group(2).lower()
            unit = _COUNT_UNITS.get(unit_raw, unit_raw)
            return QuantityInfo(
                count=count,
                unit=unit,
                basis=f"per {unit}",
                strength=strength,
                source="count_unit",
            )

    # Next: "pack of N"
    m = _PACK_OF_RE.search(s)
    if m:
        try:
            count = float(m.group(1))
        except ValueError:
            count = None
        if count and count > 0:
            return QuantityInfo(
                count=count, unit="unit",
                basis="per unit", strength=strength, source="pack_of",
            )

    # Next: volume (liquids)
    m = _VOLUME_RE.search(s)
    if m:
        try:
            amount = float(m.group(1))
        except ValueError:
            amount = None
        if amount and amount > 0:
            unit_raw = m.group(2).lower()
            if unit_raw in _VOLUME_UNITS:
                canonical, factor = _VOLUME_UNITS[unit_raw]
                return QuantityInfo(
                    count=amount * factor, unit=canonical,
                    basis=f"per {canonical}", strength=strength,
                    source="volume",
                )

    # Next: weight (solids, powders)
    m = _WEIGHT_RE.search(s)
    if m:
        try:
            amount = float(m.group(1))
        except ValueError:
            amount = None
        if amount and amount > 0:
            unit_raw = m.group(2).lower()
            if unit_raw in _WEIGHT_UNITS:
                canonical, factor = _WEIGHT_UNITS[unit_raw]
                return QuantityInfo(
                    count=amount * factor, unit=canonical,
                    basis=f"per {canonical}", strength=strength,
                    source="weight",
                )

    # Weakest: "x N" alone after a strength ("500mg x 200")
    if strength_match:
        m = _MULTIPLIER_RE.search(s)
        if m:
            try:
                count = float(m.group(1))
            except ValueError:
                count = None
            if count and count > 0:
                return QuantityInfo(
                    count=count, unit="unit",
                    basis="per unit", strength=strength, source="multiplier",
                )

    return None


def attach_unit_price(
    record: dict,
    price_field: str = "price",
    quantity_field: str = "quantity",
    title_fields: tuple[str, ...] = ("title", "product_name", "name"),
) -> dict:
    """
    Mutate `record` in place: try to extract quantity (from `quantity_field`
    first, then from any title field), then compute
    `single_unit_price = normalized_price / package_quantity`.

    Safe: if anything is missing, the record is returned unchanged except
    for missing keys being left as-is.
    """
    if not isinstance(record, dict):
        return record

    # Find the best candidate text for quantity parsing
    candidates = []
    if quantity_field in record and record[quantity_field]:
        candidates.append(str(record[quantity_field]))
    for tf in title_fields:
        if record.get(tf):
            candidates.append(str(record[tf]))

    info: Optional[QuantityInfo] = None
    for text in candidates:
        info = parse_quantity(text)
        if info is not None:
            break

    if info is None:
        return record

    record["package_quantity"] = info.count
    record["package_unit"] = info.unit
    record["unit_price_basis"] = info.basis
    if info.strength:
        record["dose_strength"] = info.strength

    # Try to compute single-unit price from whatever normalized price
    # fields the cleaner produced
    price = None
    for key in (f"{price_field}_normalized", price_field):
        v = record.get(key)
        if v in (None, ""):
            continue
        try:
            price = float(v)
            break
        except (TypeError, ValueError):
            continue

    if price is not None and info.count > 0:
        record["single_unit_price"] = round(price / info.count, 4)

    return record


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Direct count units
    q = parse_quantity("Panadol 500mg x 200 tablets")
    assert q is not None
    assert q.count == 200
    assert q.unit == "tablet"
    assert q.strength == "500mg"

    q = parse_quantity("Paracetamol 24 caplets")
    assert q and q.count == 24 and q.unit == "caplet"

    q = parse_quantity("Panadol Extra 100 tablets")
    assert q and q.count == 100 and q.unit == "tablet"

    # Volume
    q = parse_quantity("Cough Syrup 100ml bottle")
    assert q and q.count == 100.0 and q.unit == "ml"

    q = parse_quantity("1L solution")
    assert q and q.count == 1000.0 and q.unit == "ml"

    # Weight
    q = parse_quantity("Glucose 500g")
    assert q and q.count == 500.0 and q.unit == "g"

    q = parse_quantity("Protein 1kg")
    assert q and q.count == 1000.0 and q.unit == "g"

    # Pack of N
    q = parse_quantity("Vitamin C pack of 60")
    assert q and q.count == 60 and q.unit == "unit"

    # Multiplier only
    q = parse_quantity("Panadol 500mg x 24")
    assert q and q.count == 24 and q.unit == "unit" and q.strength == "500mg"

    # No quantity -> None
    assert parse_quantity("Panadol") is None
    assert parse_quantity("") is None
    assert parse_quantity(None) is None

    # Unit price attached
    rec = {
        "title": "Panadol 500mg x 200 tablets",
        "price": "$400",
        "price_normalized": 400.0,
    }
    attach_unit_price(rec)
    assert rec["package_quantity"] == 200
    assert rec["package_unit"] == "tablet"
    assert rec["single_unit_price"] == 2.0

    # No price -> no single_unit_price
    rec2 = {"title": "Panadol 500mg x 200 tablets"}
    attach_unit_price(rec2)
    assert "single_unit_price" not in rec2

    # Nothing parseable -> unchanged
    rec3 = {"title": "random thing", "price_normalized": 5.0}
    attach_unit_price(rec3)
    assert "package_quantity" not in rec3

    print("Quantity parsing OK.")