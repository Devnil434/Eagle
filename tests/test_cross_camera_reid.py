"""
tests/test_cross_camera_reid.py — Unit tests for cross-camera ReID, MemoryService,
and the /cameras API routes.

All tests are offline — no real Redis, no network, no GPU.
We use fakeredis for Redis and FastAPI TestClient for route tests.

Run with:
    pytest tests/test_cross_camera_reid.py -v
"""
from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import time

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_redis():
    """In-memory Redis replacement (no server required)."""
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeRedis()


@pytest.fixture()
def reid(fake_redis):
    from services.tracking.cross_camera_reid import CrossCameraReID
    return CrossCameraReID(fake_redis, cosine_threshold=0.85, window_seconds=5.0)


@pytest.fixture()
def memory_service(fake_redis, reid):
    from services.memory.memory import MemoryService
    return MemoryService(fake_redis, reid)


# ── Helper factories ──────────────────────────────────────────────────────────

def _random_embed(seed: int = 0, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _make_event(
    event_type,
    camera_id: str = "cam_01",
    track_id:  int = 1,
    frame_id:  int = 0,
):
    from libs.schemas.tracking import TrackLifecycleEvent, TrackState
    return TrackLifecycleEvent(
        event              = event_type,
        track_id           = track_id,
        frame_id           = frame_id,
        camera_id          = camera_id,
        zones_present      = [],
        dwell_time_seconds = 0.0,
        timestamp_ms       = time.time() * 1000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CrossCameraReID unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossCameraReID:

    def test_new_track_gets_unique_global_id(self, reid):
        embed = _random_embed(seed=1)
        r1    = reid.match_or_create("cam_01", 1, embed)
        r2    = reid.match_or_create("cam_02", 2, embed * 0)  # zero vector, different embed
        # Both are novel entries on first call — they should get ids
        assert r1.global_id
        assert r2.global_id
        # They are different tracks with very different embeddings
        # (r2 embed is zeroed so similarity will be ~0)
        assert r2.global_id != r1.global_id or r2.is_new

    def test_same_person_across_cameras_matched(self, reid):
        """Identical embedding on cam_01 then cam_02 should share a global_id."""
        embed = _random_embed(seed=42)

        # Person LOST on cam_01 → store embedding
        reid.store_embedding("cam_01", 3, embed, global_id=None)

        # Small sleep NOT needed — fakeredis doesn't age entries in unit time
        result = reid.match_or_create("cam_02", 7, embed)

        assert not result.is_new, "Should match existing embedding, not create new id"
        assert result.similarity >= 0.85
        assert result.matched_cam   == "cam_01"
        assert result.matched_track == 3

    def test_different_persons_not_matched(self, reid):
        """Orthogonal embeddings should not be merged."""
        e1 = np.zeros(128, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(128, dtype=np.float32); e2[1] = 1.0   # orthogonal

        reid.store_embedding("cam_01", 1, e1)
        result = reid.match_or_create("cam_02", 2, e2)

        # cosine(e1, e2) = 0  < 0.85 threshold
        assert result.is_new, "Orthogonal embeddings must not be merged"
        assert result.similarity < 0.85

    def test_identity_map_updated(self, reid):
        embed = _random_embed(seed=99)
        reid.store_embedding("cam_01", 5, embed, global_id=None)
        result = reid.match_or_create("cam_02", 8, embed)

        tokens = reid.get_identity(result.global_id)
        assert "cam_02:8" in tokens

    def test_old_embedding_outside_window_ignored(self, reid, fake_redis):
        """Embeddings older than WINDOW_SECONDS must not be matched."""
        embed = _random_embed(seed=7)

        # Manually write a record with an old timestamp
        old_ms = (time.time() - 10) * 1000.0   # 10 seconds ago > 5 s window
        record = {
            "camera_id":    "cam_01",
            "track_id":     99,
            "embedding":    embed.tolist(),
            "timestamp_ms": old_ms,
            "global_id":    None,
        }
        fake_redis.setex("embed:cam_01:99", 60, json.dumps(record))

        result = reid.match_or_create("cam_02", 10, embed)
        assert result.is_new, "Stale embedding must not trigger a match"

    def test_same_camera_embeddings_not_matched(self, reid):
        """Must not match a track against embeddings from the same camera."""
        embed = _random_embed(seed=3)
        reid.store_embedding("cam_01", 1, embed)
        result = reid.match_or_create("cam_01", 2, embed)
        # Match must be excluded (same camera)
        assert result.is_new

    def test_global_id_is_valid_uuid(self, reid):
        import uuid
        embed  = _random_embed(seed=11)
        result = reid.match_or_create("cam_01", 1, embed)
        parsed = uuid.UUID(result.global_id)   # raises if invalid
        assert str(parsed) == result.global_id

    def test_false_positive_rate_below_10_percent(self, reid):
        """
        Acceptance criterion: FPR < 10 % on randomly paired embeddings.

        We create 50 random pairs of unrelated embeddings (different seeds).
        Each pair is treated as cam_01 → cam_02.  We expect < 5 false positives.
        """
        fp = 0
        n  = 50
        for i in range(n):
            # Clear redis between rounds to isolate pairs
            reid._r.flushall()
            e1 = _random_embed(seed=i * 2)
            e2 = _random_embed(seed=i * 2 + 1)
            reid.store_embedding("cam_01", i, e1)
            result = reid.match_or_create("cam_02", i + 1000, e2)
            if not result.is_new:
                fp += 1
        fpr = fp / n
        assert fpr < 0.10, f"False positive rate {fpr:.0%} exceeds 10 % threshold"


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryService unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryService:

    def test_born_event_creates_track_record(self, memory_service, fake_redis):
        from libs.schemas.tracking import TrackState
        event  = _make_event(TrackState.BORN, camera_id="cam_01", track_id=1)
        gid    = memory_service.handle_lifecycle_event(event, embedding=_random_embed())
        record = memory_service.get_track_record("cam_01", 1)

        assert record is not None
        assert record["global_id"] == gid
        assert record["state"] == "ACTIVE"

    def test_lost_event_updates_state(self, memory_service):
        from libs.schemas.tracking import TrackState
        born = _make_event(TrackState.BORN, camera_id="cam_01", track_id=2)
        memory_service.handle_lifecycle_event(born, embedding=_random_embed(seed=5))

        lost = _make_event(TrackState.LOST, camera_id="cam_01", track_id=2)
        memory_service.handle_lifecycle_event(lost, embedding=_random_embed(seed=5))

        record = memory_service.get_track_record("cam_01", 2)
        assert record["state"] == "LOST"

    def test_dead_event_updates_state(self, memory_service):
        from libs.schemas.tracking import TrackState
        born = _make_event(TrackState.BORN, camera_id="cam_01", track_id=3)
        memory_service.handle_lifecycle_event(born, embedding=_random_embed(seed=6))

        dead = _make_event(TrackState.DEAD, camera_id="cam_01", track_id=3)
        memory_service.handle_lifecycle_event(dead)

        record = memory_service.get_track_record("cam_01", 3)
        assert record["state"] == "DEAD"

    def test_born_without_embedding_still_assigns_global_id(self, memory_service):
        from libs.schemas.tracking import TrackState
        event = _make_event(TrackState.BORN, camera_id="cam_01", track_id=10)
        gid   = memory_service.handle_lifecycle_event(event, embedding=None)
        assert gid is not None and len(gid) > 0

    def test_cross_camera_global_id_consistent(self, memory_service, reid, fake_redis):
        """
        Same person: BORN on cam_01 → LOST on cam_01 → BORN on cam_02.
        The global_id returned for cam_02 should match cam_01's.
        """
        from libs.schemas.tracking import TrackState
        embed = _random_embed(seed=77)

        born1 = _make_event(TrackState.BORN, camera_id="cam_01", track_id=3)
        gid1  = memory_service.handle_lifecycle_event(born1, embedding=embed)

        lost1 = _make_event(TrackState.LOST, camera_id="cam_01", track_id=3)
        memory_service.handle_lifecycle_event(lost1, embedding=embed)

        born2 = _make_event(TrackState.BORN, camera_id="cam_02", track_id=7)
        gid2  = memory_service.handle_lifecycle_event(born2, embedding=embed)

        assert gid1 == gid2, (
            f"Same person should get same global_id across cameras, "
            f"got cam_01={gid1}  cam_02={gid2}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI route tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def test_client(fake_redis, reid, memory_service):
    """Minimal FastAPI app wired with the camera & identity routers."""
    fastapi = pytest.importorskip("fastapi")
    httpx   = pytest.importorskip("httpx")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from apps.backend.routes.cameras import router, identity_router

    app = FastAPI()
    app.state.redis   = fake_redis
    app.state.reid    = reid
    app.state.memory  = memory_service
    app.include_router(router)
    app.include_router(identity_router)

    return TestClient(app)


class TestCameraRoutes:

    def test_list_tracks_empty(self, test_client):
        resp = test_client.get("/cameras/cam_01/tracks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_track_not_found(self, test_client):
        resp = test_client.get("/cameras/cam_01/tracks/999")
        assert resp.status_code == 404

    def test_push_embedding_born_returns_global_id(self, test_client):
        embed = _random_embed(seed=20).tolist()
        resp  = test_client.post(
            "/cameras/cam_01/tracks/1/embedding",
            json={"embedding": embed, "event_type": "BORN"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "global_id" in data
        assert len(data["global_id"]) > 0

    def test_push_embedding_lost_returns_200(self, test_client):
        embed = _random_embed(seed=21).tolist()
        resp  = test_client.post(
            "/cameras/cam_01/tracks/2/embedding",
            json={"embedding": embed, "event_type": "LOST"},
        )
        assert resp.status_code == 200

    def test_cross_camera_reid_via_api(self, test_client):
        """
        LOST on cam_01 via API → BORN on cam_02 via API → same global_id.
        """
        embed = _random_embed(seed=50).tolist()

        # cam_01 LOST
        test_client.post(
            "/cameras/cam_01/tracks/3/embedding",
            json={"embedding": embed, "event_type": "LOST"},
        )

        # cam_02 BORN
        resp = test_client.post(
            "/cameras/cam_02/tracks/7/embedding",
            json={"embedding": embed, "event_type": "BORN"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_cam"]   == "cam_01"
        assert data["matched_track"] == 3
        assert data["similarity"]    >= 0.85

    def test_identity_endpoint_returns_tokens(self, test_client):
        embed = _random_embed(seed=60).tolist()

        r = test_client.post(
            "/cameras/cam_01/tracks/5/embedding",
            json={"embedding": embed, "event_type": "BORN"},
        )
        gid  = r.json()["global_id"]
        resp = test_client.get(f"/identities/{gid}")
        assert resp.status_code == 200
        assert "cam_01:5" in resp.json()["tokens"]

    def test_identity_not_found(self, test_client):
        resp = test_client.get("/identities/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_invalid_event_type_rejected(self, test_client):
        embed = _random_embed(seed=70).tolist()
        resp  = test_client.post(
            "/cameras/cam_01/tracks/1/embedding",
            json={"embedding": embed, "event_type": "UNKNOWN"},
        )
        assert resp.status_code == 422

    def test_track_record_includes_global_id(self, test_client, memory_service, fake_redis):
        """After writing a track record, GET /cameras/.../tracks/:id should return global_id."""
        from libs.schemas.tracking import TrackState
        embed = _random_embed(seed=80)
        event = _make_event(TrackState.BORN, camera_id="cam_01", track_id=42)
        memory_service.handle_lifecycle_event(event, embedding=embed)

        resp = test_client.get("/cameras/cam_01/tracks/42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["track_id"]  == 42
        assert data["global_id"] is not None
