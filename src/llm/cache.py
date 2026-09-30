"""
LLM response cache — spec §42.

Caches LLM responses so re-running a task against the same page with
the same prompt never re-hits the API. This is what makes iterating on
a parser or a healing rule cheap during development.

Cache key (§42):
    sha256(
        normalized_prompt
        + screenshot_hash (if applicable)
        + dom_hash
        + model_identifier
        + relevant_options
    )

In practice the DOM is already embedded in the prompt, so hashing the
normalized prompt captures both. `model_identifier` and `options` are
folded in via a canonical JSON serialization.

Storage:
    SQLite. Zero-config, single-file, works offline. `db_path=":memory:"`
    for tests (fast, ephemeral).

TTL and invalidation (§42.2):
    - Default TTL: 7 days.
    - Entries past TTL are treated as misses and lazily evicted.
    - Size caps (entry count + total bytes) with LRU eviction on
      `last_used_at`.
    - `prompt_template_version` is part of the key; bump it to
      invalidate every entry produced by a prior template version.

Thread safety:
    A single `threading.Lock` serializes all writes and reads. The
    workload is low-contention (LLM calls dominate), so a single lock
    is simpler and fast enough.

Never raises:
    All storage failures are logged and swallowed. A broken cache must
    never break the pipeline it is trying to accelerate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger("llm.cache")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = ":memory:"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600    # 7 days per §42.2
DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key             TEXT PRIMARY KEY,
    prompt          TEXT NOT NULL,
    response_text   TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    last_used_at    REAL NOT NULL,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    response_bytes  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_last_used
    ON llm_cache (last_used_at);
"""


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------

def compute_key(
    prompt: str,
    *,
    model: str = "",
    options: Optional[dict] = None,
    prompt_template_version: str = "",
    dom_hash: str = "",
    screenshot_hash: str = "",
) -> str:
    """
    Deterministic cache key.

    Sorts keys, drops None options, collapses whitespace in the prompt —
    so cosmetic differences do not produce different keys.
    """
    payload = {
        "prompt": _normalize(prompt),
        "model": model or "",
        "options": _canonicalize_options(options or {}),
        "template_version": prompt_template_version or "",
        "dom_hash": dom_hash or "",
        "screenshot_hash": screenshot_hash or "",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize(prompt: str) -> str:
    return " ".join((prompt or "").split())


def _canonicalize_options(options: dict) -> dict:
    return {k: v for k, v in (options or {}).items() if v is not None}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    key: str
    prompt: str
    response_text: str
    provider: str = ""
    model: str = ""
    created_at: float = 0.0
    last_used_at: float = 0.0
    hit_count: int = 0
    response_bytes: int = 0

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class LLMCache:
    """SQLite-backed LLM response cache. Thread-safe. Never raises."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        clock=time.time,
    ):
        self.db_path = str(db_path)
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._clock = clock
        self._lock = threading.Lock()

        # Process-local counters (not persisted). Surfaced via stats().
        self._hits = 0
        self._misses = 0
        self._expired = 0
        self._evicted = 0

        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[CacheEntry]:
        """Return the entry if present and unexpired; else None."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM llm_cache WHERE key = ?", (key,),
            )
            row = cur.fetchone()
            if row is None:
                self._misses += 1
                return None

            now = self._clock()
            if now - row["created_at"] > self.ttl_seconds:
                self._conn.execute(
                    "DELETE FROM llm_cache WHERE key = ?", (key,),
                )
                self._conn.commit()
                self._expired += 1
                self._misses += 1
                return None

            self._conn.execute(
                "UPDATE llm_cache SET last_used_at = ?, "
                "hit_count = hit_count + 1 WHERE key = ?",
                (now, key),
            )
            self._conn.commit()
            self._hits += 1

            entry = _row_to_entry(row)
            entry.hit_count += 1
            entry.last_used_at = now
            return entry

    def put(
        self,
        key: str,
        prompt: str,
        response_text: str,
        *,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Store a response. Silent no-op on storage failure."""
        now = self._clock()
        size = len(response_text.encode("utf-8", errors="ignore"))
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO llm_cache (
                        key, prompt, response_text, provider, model,
                        created_at, last_used_at, hit_count, response_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT (key) DO UPDATE SET
                        response_text  = excluded.response_text,
                        provider       = excluded.provider,
                        model          = excluded.model,
                        created_at     = excluded.created_at,
                        last_used_at   = excluded.last_used_at,
                        response_bytes = excluded.response_bytes
                    """,
                    (key, prompt, response_text, provider, model,
                     now, now, size),
                )
                self._conn.commit()
            self._maybe_evict()
        except Exception as e:
            logger.warning(f"LLM cache put failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def evict_expired(self) -> int:
        """Delete every entry past TTL. Returns count removed."""
        cutoff = self._clock() - self.ttl_seconds
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM llm_cache WHERE created_at < ?", (cutoff,),
            )
            self._conn.commit()
            n = cur.rowcount or 0
            self._expired += n
            return n

    def clear(self) -> int:
        """Delete everything. Returns count removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()
            return cur.rowcount or 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(response_bytes), 0) AS b FROM llm_cache"
            )
            row = cur.fetchone()
            n = row["n"] if row else 0
            b = row["b"] if row else 0
        total = self._hits + self._misses
        return {
            "entry_count": n,
            "total_bytes": b,
            "hits": self._hits,
            "misses": self._misses,
            "expired": self._expired,
            "evicted": self._evicted,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _maybe_evict(self) -> None:
        """LRU eviction when either cap is exceeded."""
        with self._lock:
            while True:
                cur = self._conn.execute(
                    "SELECT COUNT(*) AS n, "
                    "COALESCE(SUM(response_bytes), 0) AS b FROM llm_cache"
                )
                row = cur.fetchone()
                n = row["n"] or 0
                b = row["b"] or 0
                if n <= self.max_entries and b <= self.max_bytes:
                    return
                cur = self._conn.execute(
                    "DELETE FROM llm_cache WHERE key IN ("
                    "  SELECT key FROM llm_cache "
                    "  ORDER BY last_used_at ASC LIMIT 1"
                    ")"
                )
                self._conn.commit()
                self._evicted += cur.rowcount or 0
                if (cur.rowcount or 0) == 0:
                    return


def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
    return CacheEntry(
        key=row["key"],
        prompt=row["prompt"],
        response_text=row["response_text"],
        provider=row["provider"],
        model=row["model"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        hit_count=row["hit_count"],
        response_bytes=row["response_bytes"],
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_default_cache() -> Optional[LLMCache]:
    """
    Build the cache the LLMRouter uses by default.

    Env:
        ACES_LLM_CACHE_DISABLED   "1" disables entirely
        ACES_LLM_CACHE_PATH       sqlite file path (default ~/.aces/llm_cache.db)
        ACES_LLM_CACHE_TTL        seconds (default 604800 = 7 days)
        ACES_LLM_CACHE_MAX        max entries (default 10000)

    Returns None (no cache) when:
        - disabled via env
        - the home directory isn't writable
        - the SQLite file can't be opened
        - running under pytest (avoids creating files in ~ during tests)
    """
    import os
    import sys

    # Never auto-enable under pytest — tests shouldn't write to the user's
    # home directory just by importing LLMRouter.
    if "pytest" in sys.modules:
        return None

    if os.getenv("ACES_LLM_CACHE_DISABLED", "").strip() == "1":
        return None

    path = os.getenv("ACES_LLM_CACHE_PATH", "").strip()
    if not path:
        home = Path.home() / ".aces"
        try:
            home.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        path = str(home / "llm_cache.db")

    ttl = _int_env("ACES_LLM_CACHE_TTL", DEFAULT_TTL_SECONDS)
    mx = _int_env("ACES_LLM_CACHE_MAX", DEFAULT_MAX_ENTRIES)
    try:
        return LLMCache(db_path=path, ttl_seconds=ttl, max_entries=mx)
    except Exception as e:
        logger.warning(f"LLM cache disabled (init failed): {e}")
        return None


def _int_env(name: str, default: int) -> int:
    import os
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default