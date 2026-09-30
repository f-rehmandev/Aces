"""Unit tests for API keys (spec §45.1)."""
from src.api.keys import (
    ApiKeyStore, ApiKey, generate_key, hash_key, key_prefix,
)


def test_generate_key_prefix():
    k = generate_key()
    assert k.startswith("aces_")
    assert len(k) > 20


def test_key_prefix_display():
    k = "aces_abcdefghijklmnop"
    p = key_prefix(k)
    assert p.endswith("…")
    assert len(p) == 13   # 12 chars + ellipsis


def test_hash_key_stable():
    assert hash_key("x", b"p") == hash_key("x", b"p")


def test_hash_key_pepper_changes_output():
    assert hash_key("x", b"p1") != hash_key("x", b"p2")


def test_create_returns_raw_key_once():
    store = ApiKeyStore()
    rec, raw = store.create("acme", note="dev")
    assert raw.startswith("aces_")
    assert rec.client_id == "acme"
    assert rec.note == "dev"
    # Hash never exposed in serialized output
    assert "key_hash" not in rec.to_dict()


def test_verify_valid_key():
    store = ApiKeyStore(pepper=b"pep")
    rec, raw = store.create("acme")
    found = store.verify(raw)
    assert found is not None
    assert found.key_id == rec.key_id
    assert found.last_used_at is not None


def test_verify_wrong_key_returns_none():
    store = ApiKeyStore()
    store.create("acme")
    assert store.verify("aces_wrong") is None


def test_verify_wrong_pepper_returns_none():
    store = ApiKeyStore(pepper=b"pep1")
    _, raw = store.create("acme")
    other = ApiKeyStore(pepper=b"pep2")
    other._by_hash.update(store._by_hash)
    other._by_id.update(store._by_id)
    assert other.verify(raw) is None


def test_revoke_makes_key_invalid():
    store = ApiKeyStore()
    rec, raw = store.create("acme")
    assert store.revoke(rec.key_id)
    assert store.verify(raw) is None


def test_revoke_is_idempotent():
    store = ApiKeyStore()
    rec, _ = store.create("acme")
    assert store.revoke(rec.key_id)
    assert not store.revoke(rec.key_id)


def test_list_for_client():
    store = ApiKeyStore()
    store.create("acme")
    store.create("acme")
    store.create("other")
    assert len(store.list_for_client("acme")) == 2
    assert len(store.list_for_client("other")) == 1


def test_persistence_roundtrip():
    store = ApiKeyStore(pepper=b"pep")
    rec, raw = store.create("acme", note="survives reload")
    rows = store.to_rows()
    store2 = ApiKeyStore.from_rows(rows, pepper=b"pep")
    assert store2.verify(raw) is not None
    assert store2.get(rec.key_id).note == "survives reload"


def test_from_rows_no_raw_keys_possible():
    # Rows don't carry the raw key — verify can't be called without it.
    store = ApiKeyStore(pepper=b"p")
    store.create("acme")
    rows = store.to_rows()
    for row in rows:
        assert "raw" not in row
        assert "key_hash" in row   # hash is stored, raw is not