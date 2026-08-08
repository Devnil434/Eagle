"""
tests/test_baseline.py — Unit tests for ZoneBaseline (Welford's algorithm).

Reference values computed independently with Python's statistics module.
"""
import math
import statistics

import pytest
import fakeredis

from services.memory.baseline import ZoneBaseline, MIN_COUNT_FOR_ANOMALY


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def baseline(redis_client):
    return ZoneBaseline(redis_client, "test_zone")


# ── Welford math ──────────────────────────────────────────────────────────────

SAMPLES = [10.0, 15.0, 12.0, 30.0, 11.0, 14.0, 13.0, 9.0, 16.0, 20.0]


def test_welford_mean_matches_reference(baseline):
    for s in SAMPLES:
        baseline.update(s)
    stats = baseline.get_stats()
    assert math.isclose(stats["mean"], statistics.mean(SAMPLES), rel_tol=1e-6)


def test_welford_std_matches_reference(baseline):
    for s in SAMPLES:
        baseline.update(s)
    stats = baseline.get_stats()
    # get_stats() rounds to 4dp; compare with abs_tol matching that precision
    assert math.isclose(stats["std"], statistics.stdev(SAMPLES), abs_tol=1e-4)


def test_welford_variance_matches_reference(baseline):
    for s in SAMPLES:
        baseline.update(s)
    stats = baseline.get_stats()
    assert math.isclose(stats["variance"], statistics.variance(SAMPLES), rel_tol=1e-6)


def test_welford_count(baseline):
    for s in SAMPLES:
        baseline.update(s)
    assert baseline.get_stats()["count"] == len(SAMPLES)


# ── Redis persistence ─────────────────────────────────────────────────────────

def test_stats_persist_across_instances(redis_client):
    """A new ZoneBaseline instance reading the same Redis key sees prior data."""
    b1 = ZoneBaseline(redis_client, "persist_zone")
    for s in SAMPLES:
        b1.update(s)

    b2 = ZoneBaseline(redis_client, "persist_zone")
    stats = b2.get_stats()
    assert stats["count"] == len(SAMPLES)
    assert math.isclose(stats["mean"], statistics.mean(SAMPLES), rel_tol=1e-6)


def test_in_memory_cache_used_after_first_load(redis_client):
    """Second call to _load() returns cached object without hitting Redis."""
    b = ZoneBaseline(redis_client, "cache_zone")
    b.update(10.0)
    first = b._load()
    second = b._load()
    assert first is second


def test_different_zones_are_independent(redis_client):
    za = ZoneBaseline(redis_client, "zone_a")
    zb = ZoneBaseline(redis_client, "zone_b")

    for s in SAMPLES:
        za.update(s)
    zb.update(999.0)

    assert za.get_stats()["count"] == len(SAMPLES)
    assert zb.get_stats()["count"] == 1


# ── Anomaly detection ─────────────────────────────────────────────────────────

def test_zero_std_not_flagged(baseline):
    """When std==0 (all identical values) nothing should be anomalous."""
    for _ in range(MIN_COUNT_FOR_ANOMALY):
        baseline.update(10.0)
    assert baseline.is_anomalous(10.001) is False
    assert baseline.is_anomalous(9999.0) is False


def test_detect_before_update_does_not_contaminate(redis_client):
    """Anomaly detection must happen before baseline update."""
    b = ZoneBaseline(redis_client, "order_zone")
    # Use varied samples so std > 0, enabling anomaly detection
    for v in [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3]:
        b.update(v)
    mean_before = b.get_stats()["mean"]
    outlier = 9999.0
    # std > 0 now, so outlier is correctly flagged
    assert b.is_anomalous(outlier) is True
    b.update(outlier)
    # mean shifts only after detection — correct order confirmed
    assert b.get_stats()["mean"] > mean_before


def test_no_anomaly_before_min_count(baseline):
    """Anomaly detection is suppressed until MIN_COUNT_FOR_ANOMALY samples."""
    for _ in range(MIN_COUNT_FOR_ANOMALY - 1):
        baseline.update(10.0)
    assert baseline.is_anomalous(9999.0) is False


def test_anomaly_flagged_for_outlier(baseline):
    # Use varied samples so std > 0, then test a clear outlier
    for v in [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3]:
        baseline.update(v)
    # mean ~10.1, std ~0.45 → threshold ~10.1 + 2.5*0.45 ~11.2 → 50.0 is anomalous
    assert baseline.is_anomalous(50.0) is True


def test_normal_dwell_not_flagged(baseline):
    for s in SAMPLES:
        baseline.update(s)
    mean = statistics.mean(SAMPLES)
    assert baseline.is_anomalous(mean) is False


def test_empty_stats_returns_zero(baseline):
    stats = baseline.get_stats()
    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0


# ── Input validation ──────────────────────────────────────────────────────────

def test_invalid_zone_name_raises(redis_client):
    import pytest
    with pytest.raises(ValueError, match="Invalid zone name"):
        ZoneBaseline(redis_client, "../../etc/passwd")


def test_zone_name_with_special_chars_raises(redis_client):
    import pytest
    with pytest.raises(ValueError):
        ZoneBaseline(redis_client, "zone<script>alert(1)</script>")
