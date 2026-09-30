"""
Human-like jitter — spec §15.4.

A static 1-second delay between requests is the single biggest
automation tell. Real users pause to read, scroll, and think. This
module produces unpredictable, right-skewed delays that resemble real
user interaction rather than a metronome.

Design:
    - Delays are drawn from a right-skewed distribution (log-normal by
      default) so short pauses dominate but long "reading" pauses
      occasionally appear.
    - The distribution is per-domain seeded so a given domain gets a
      consistent pacing profile within a session, which prevents
      "every request takes exactly 1.2 seconds" patterns.
    - An optional burst-vs-rest pattern occasionally injects a much
      longer pause (default 5% of requests get 4-8x the median).
    - All values are clamped to a hard [min, max] window so a rare
      log-normal outlier can't hang a crawl for 60 seconds.

Never sleeps more than `hard_max` seconds. Never sleeps less than
`hard_min` seconds. Caller-injectable `rng` and `sleep` for tests.
"""
from __future__ import annotations

import logging
import math
import random
import time
from typing import Awaitable, Callable, Optional


logger = logging.getLogger("network.jitter")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MEDIAN_SECONDS = 1.2
DEFAULT_SIGMA = 0.5           # log-normal shape parameter
DEFAULT_HARD_MIN = 0.4
DEFAULT_HARD_MAX = 6.0
DEFAULT_BURST_PROBABILITY = 0.05
DEFAULT_BURST_MULTIPLIER = (4.0, 8.0)   # uniform range


class Jitter:
    """
    Per-domain human-like delay generator.

    Usage:
        jit = Jitter()
        delay = jit.next_delay("example.com")   # → float seconds
        await jit.sleep_for("example.com")      # → actually sleeps

    `sleep` is injectable so tests can pass a no-op.
    """

    def __init__(
        self,
        median_seconds: float = DEFAULT_MEDIAN_SECONDS,
        sigma: float = DEFAULT_SIGMA,
        hard_min: float = DEFAULT_HARD_MIN,
        hard_max: float = DEFAULT_HARD_MAX,
        burst_probability: float = DEFAULT_BURST_PROBABILITY,
        burst_multiplier_range: tuple[float, float] = DEFAULT_BURST_MULTIPLIER,
        rng: Optional[random.Random] = None,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ):
        if median_seconds <= 0:
            raise ValueError("median_seconds must be > 0")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        if hard_min < 0 or hard_max < hard_min:
            raise ValueError("must have 0 <= hard_min <= hard_max")
        if not 0.0 <= burst_probability <= 1.0:
            raise ValueError("burst_probability must be in [0, 1]")

        self.median_seconds = float(median_seconds)
        self.sigma = float(sigma)
        self.hard_min = float(hard_min)
        self.hard_max = float(hard_max)
        self.burst_probability = float(burst_probability)
        self.burst_multiplier_range = tuple(burst_multiplier_range)

        # Seed per-domain to keep a single session's pacing coherent.
        # We seed from the domain string's hash plus a random offset so
        # two different Jitter instances don't collide.
        self._rng = rng or random.Random()
        self._session_seed = self._rng.randint(0, 2**32 - 1)
        self._domain_rngs: dict[str, random.Random] = {}

        if sleep is None:
            import asyncio
            sleep = asyncio.sleep
        self._sleep = sleep

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _rng_for(self, domain: str) -> random.Random:
        d = (domain or "").lower()
        if d not in self._domain_rngs:
            seed = (hash(d) ^ self._session_seed) & 0xFFFFFFFF
            self._domain_rngs[d] = random.Random(seed)
        return self._domain_rngs[d]

    def next_delay(self, domain: str = "") -> float:
        """
        Sample a delay in seconds for one request to `domain`.
        """
        rng = self._rng_for(domain)

        # Log-normal around the median
        # median = exp(mu) → mu = ln(median)
        mu = math.log(self.median_seconds)
        delay = rng.lognormvariate(mu, self.sigma)

        # Occasional burst: reading a page, filling a form, etc.
        if rng.random() < self.burst_probability:
            lo, hi = self.burst_multiplier_range
            delay *= rng.uniform(lo, hi)

        # Clamp to hard bounds so a rare outlier can't hang the crawl
        return max(self.hard_min, min(self.hard_max, delay))

    async def sleep_for(self, domain: str = "") -> float:
        """
        Sleep for the sampled delay. Returns the actual number of
        seconds slept (useful for tests and telemetry).
        """
        delay = self.next_delay(domain)
        logger.debug(f"jitter: sleeping {delay:.2f}s before {domain!r}")
        await self._sleep(delay)
        return delay

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def sample(self, domain: str = "", n: int = 100) -> list[float]:
        """Return n sampled delays — useful for tests / distribution checks."""
        return [self.next_delay(domain) for _ in range(n)]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Deterministic RNG
    j = Jitter(rng=random.Random(42))

    # Basic sampling
    samples = j.sample("example.com", n=1000)
    assert all(0.4 <= s <= 6.0 for s in samples)
    avg = sum(samples) / len(samples)
    # Median around 1.2; average slightly higher due to right skew
    assert 0.8 < avg < 3.0, f"average {avg} out of expected range"

    # Minimum floor respected
    floor_j = Jitter(
        median_seconds=0.01, sigma=0.1,
        hard_min=0.5, hard_max=1.0, rng=random.Random(1),
    )
    samples = floor_j.sample(n=200)
    assert all(s >= 0.5 for s in samples)

    # Ceiling respected
    ceil_j = Jitter(
        median_seconds=10.0, sigma=0.5,
        hard_min=0.1, hard_max=3.0, rng=random.Random(2),
    )
    samples = ceil_j.sample(n=200)
    assert all(s <= 3.0 for s in samples)

    # Per-domain determinism across instances: two Jitter objects
    # built with the same seed produce the same sequence for the
    # same domain.
    j_a = Jitter(rng=random.Random(99))
    j_b = Jitter(rng=random.Random(99))
    assert j_a.sample("domain-a.com", n=20) == j_b.sample("domain-a.com", n=20)

    # Within one instance, the RNG advances state between calls —
    # consecutive samples differ (this is what prevents metronome
    # pacing patterns).
    j_c = Jitter(rng=random.Random(99))
    first = j_c.sample("domain-a.com", n=20)
    second = j_c.sample("domain-a.com", n=20)
    assert first != second

    # Different domains within the same instance get different
    # sequences (their RNG seeds are domain-derived).
    j_d = Jitter(rng=random.Random(99))
    a = j_d.sample("domain-a.com", n=20)
    b = j_d.sample("domain-b.com", n=20)
    assert a != b

    # sleep_for uses the injected sleep
    slept: list[float] = []
    async def fake_sleep(s): slept.append(s)
    j4 = Jitter(rng=random.Random(7), sleep=fake_sleep)
    async def run():
        got = await j4.sleep_for("example.com")
        return got
    got = asyncio.run(run())
    assert slept == [got]
    assert 0.4 <= got <= 6.0

    # Bad args rejected
    for bad in [
        dict(median_seconds=0),
        dict(sigma=-1),
        dict(hard_min=-1),
        dict(hard_min=2, hard_max=1),
        dict(burst_probability=2.0),
    ]:
        try:
            Jitter(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass

    print("Jitter OK.")