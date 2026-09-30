"""Unit tests for quantity parsing + unit price derivation."""
import pytest

from src.trust.quantity import parse_quantity, attach_unit_price, QuantityInfo


# ---------------------------------------------------------------------------
# Tablets / caplets / capsules
# ---------------------------------------------------------------------------

def test_basic_tablets():
    q = parse_quantity("Panadol 500mg x 200 tablets")
    assert q is not None
    assert q.count == 200
    assert q.unit == "tablet"
    assert q.strength == "500mg"


def test_caplets():
    q = parse_quantity("Tylenol 24 caplets")
    assert q and q.count == 24 and q.unit == "caplet"


def test_capsules():
    q = parse_quantity("Amoxicillin 20 capsules")
    assert q and q.count == 20 and q.unit == "capsule"


def test_tabs_abbreviation():
    q = parse_quantity("Aspirin 30 tabs")
    assert q and q.count == 30 and q.unit == "tablet"


def test_strip():
    q = parse_quantity("Panadol 5 strips")
    assert q and q.count == 5 and q.unit == "strip"


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def test_ml():
    q = parse_quantity("Cough syrup 100 ml")
    assert q and q.count == 100 and q.unit == "ml"


def test_liters_normalized_to_ml():
    q = parse_quantity("Saline 1 L")
    assert q and q.count == 1000.0 and q.unit == "ml"


def test_liter_variants():
    q = parse_quantity("2 litres")
    assert q and q.count == 2000.0 and q.unit == "ml"


# ---------------------------------------------------------------------------
# Weight
# ---------------------------------------------------------------------------

def test_grams():
    q = parse_quantity("Glucose 500 g")
    assert q and q.count == 500.0 and q.unit == "g"


def test_kilograms_normalized_to_grams():
    q = parse_quantity("Whey protein 2 kg")
    assert q and q.count == 2000.0 and q.unit == "g"


def test_mg_is_not_a_package_quantity_by_itself():
    # "500mg" is a *strength*, not a package count.
    q = parse_quantity("Panadol 500mg")
    # No count unit found, no volume/weight with a plural, no pack-of,
    # no multiplier -> None
    assert q is None


# ---------------------------------------------------------------------------
# Ambiguous / edge
# ---------------------------------------------------------------------------

def test_pack_of_phrase():
    q = parse_quantity("Vitamin C pack of 60")
    assert q and q.count == 60 and q.unit == "unit"


def test_multiplier_only_after_strength():
    q = parse_quantity("Panadol 500mg x 24")
    assert q and q.count == 24 and q.unit == "unit" and q.strength == "500mg"


def test_comma_separated_count():
    q = parse_quantity("Bulk 1,000 tablets")
    assert q and q.count == 1000 and q.unit == "tablet"


def test_empty_and_none():
    assert parse_quantity("") is None
    assert parse_quantity(None) is None
    assert parse_quantity("no numbers here") is None


def test_non_string():
    assert parse_quantity(123) is None


# ---------------------------------------------------------------------------
# attach_unit_price
# ---------------------------------------------------------------------------

def test_attach_computes_single_unit_price():
    rec = {
        "title": "Panadol 500mg x 200 tablets",
        "price": "$400",
        "price_normalized": 400.0,
    }
    attach_unit_price(rec)
    assert rec["package_quantity"] == 200
    assert rec["package_unit"] == "tablet"
    assert rec["single_unit_price"] == 2.0


def test_attach_prefers_dedicated_quantity_field():
    rec = {
        "title": "Panadol pack",
        "quantity": "24 tablets",
        "price_normalized": 48.0,
    }
    attach_unit_price(rec)
    assert rec["package_quantity"] == 24
    assert rec["single_unit_price"] == 2.0


def test_attach_no_price_leaves_no_unit_price():
    rec = {"title": "Panadol 500mg x 200 tablets"}
    attach_unit_price(rec)
    assert rec["package_quantity"] == 200
    assert "single_unit_price" not in rec


def test_attach_unparseable_title_unchanged():
    rec = {"title": "asdf", "price_normalized": 9.99}
    attach_unit_price(rec)
    assert "package_quantity" not in rec
    assert "single_unit_price" not in rec


def test_attach_uses_raw_price_as_fallback():
    rec = {"title": "Cough syrup 100ml", "price": "$25.00"}
    attach_unit_price(rec)
    # "$25.00" isn't a float; no normalized price -> no single_unit_price
    assert rec["package_quantity"] == 100
    assert "single_unit_price" not in rec


def test_attach_non_dict_returned_unchanged():
    assert attach_unit_price(None) is None
    assert attach_unit_price("string") == "string"