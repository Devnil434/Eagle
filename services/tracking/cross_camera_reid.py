"""
cross_camera_reid.py — Cross-camera person re-identification using appearance embeddings.

Algorithm
---------
1. Each Tracker emits appearance embeddings (deep features) from DeepSort when a
   track transitions to LOST.  The embedding is stored in Redis under a per-camera
   key with a TTL of WINDOW_SECONDS.

2. On every BORN event in *any* camera we retrieve embeddings from *all other*
   cameras that appeared within the last WINDOW_SECONDS seconds.

3. We compute cosine similarity between the new embedding and every stored candidate.
   If the best match exceeds COSINE_THRESHOLD (default 0.85) we assign the same
   ``global_id``; otherwise we mint a new one.

4. The global_id → [cam:track] mapping is persisted in Redis under the key
   ``identity:{global_id}`` so the API layer can expose it.

Redis key schema
----------------
- ``embed:{camera_id}:{track_id}``  → JSON blob  {embedding, timestamp_ms, global_id}
  TTL: WINDOW_SECONDS
- ``identity:{global_id}``          → JSON list   ["cam1:3", "cam2:7", …]
  TTL: IDENTITY_TTL_SECONDS

Usage
-----
    reid = CrossCameraReID(redis_client)
    # on LOST in cam1, call:
    reid.store_embedding(camera_id="cam_01", track_id=3, embedding=vec)
    # on BORN in cam2, call:
    result = reid.match_or_create(camera_id="cam_02", track_id=7, embedding=vec2)
    print(result.global_id)   # same id as cam1:3 if similarity > threshold
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────
COSINE_THRESHOLD    = 0.85   # minimum similarity to consider same identity
WINDOW_SECONDS      = 5.0    # max gap between disappearance and re-appearance
IDENTITY_TTL_SECONDS = 3600  # how long global identity mapping lives in Redis
EMBED_TTL_SECONDS   = int(WINDOW_SECONDS) + 2   # small buffer over window


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class EmbeddingRecord:
    camera_id:    str
    track_id:     int
    embedding:    list[float]          # unit-normalised appearance vector
    timestamp_ms: float
    global_id:    Optional[str] = None


@dataclass
class ReidResult:
    global_id:     str
    is_new:        bool                # True → new identity minted
    matched_cam:   Optional[str] = None
    matched_track: Optional[int] = None
    similarity:    float = 0.0


# ── Core class ───────────────────────────────────────────────────────────────

class CrossCameraReID:
    """
    Stateless helper that reads/writes embeddings and identities via Redis.

    Parameters
    ----------
    redis_client:
        A ``redis.Redis`` (or compatible) client with ``get``, ``setex``,
        ``append``, ``keys`` methods.  Pass a ``FakeRedis`` in tests.
    cosine_threshold:
        Similarity cutoff for a positive match.
    window_seconds:
        Time window within which a disappearance on one camera can be linked
        to an appearance on another.
    """

    def __init__(
        self,
        redis_client,
        cosine_threshold: float = COSINE_THRESHOLD,
        window_seconds:   float = WINDOW_SECONDS,
    ) -> None:
        self._r         = redis_client
        self._threshold = cosine_threshold
        self._window_ms = window_seconds * 1000.0

    # ── Public API ────────────────────────────────────────────────────────────

    def store_embedding(
        self,
        camera_id:  str,
        track_id:   int,
        embedding:  np.ndarray,
        global_id:  Optional[str] = None,
    ) -> None:
        """
        Persist a track's appearance embedding when it goes LOST.

        Call this from the memory service whenever a LOST lifecycle event arrives
        and the tracker can supply the embedding vector.

        Args:
            camera_id:  e.g. ``"cam_01"``
            track_id:   integer track id from DeepSort
            embedding:  raw float32 feature vector (will be L2-normalised here)
            global_id:  existing global id if already known, else None
        """
        norm_vec = self._normalise(embedding)
        record = {
            "camera_id":    camera_id,
            "track_id":     track_id,
            "embedding":    norm_vec.tolist(),
            "timestamp_ms": time.time() * 1000.0,
            "global_id":    global_id,
        }
        key = self._embed_key(camera_id, track_id)
        self._r.setex(key, EMBED_TTL_SECONDS, json.dumps(record))
        logger.debug("Stored embedding %s  global_id=%s", key, global_id)

    def match_or_create(
        self,
        camera_id:  str,
        track_id:   int,
        embedding:  np.ndarray,
    ) -> ReidResult:
        """
        Try to match a new track against recently-lost tracks on other cameras.

        Call this from the memory service on a BORN lifecycle event.

        Returns a ``ReidResult`` with the assigned ``global_id`` (existing or new).
        The result is also written into Redis so the API can retrieve it.
        """
        norm_vec   = self._normalise(embedding)
        now_ms     = time.time() * 1000.0
        candidates = self._fetch_candidates(camera_id, now_ms)

        best_sim   = -1.0
        best_rec: Optional[EmbeddingRecord] = None

        for rec in candidates:
            sim = float(np.dot(norm_vec, np.array(rec.embedding)))
            if sim > best_sim:
                best_sim = sim
                best_rec = rec

        if best_rec is not None and best_sim >= self._threshold:
            # Re-use the matched global_id (or create one if the match itself is new)
            global_id = best_rec.global_id or str(uuid.uuid4())
            result = ReidResult(
                global_id     = global_id,
                is_new        = False,
                matched_cam   = best_rec.camera_id,
                matched_track = best_rec.track_id,
                similarity    = best_sim,
            )
            logger.info(
                "ReID match  cam=%s track=%d  ←→  cam=%s track=%d  sim=%.3f  gid=%s",
                camera_id, track_id,
                best_rec.camera_id, best_rec.track_id,
                best_sim, global_id,
            )
        else:
            global_id = str(uuid.uuid4())
            result = ReidResult(global_id=global_id, is_new=True, similarity=best_sim)
            logger.info(
                "ReID new identity  cam=%s track=%d  gid=%s  (best_sim=%.3f)",
                camera_id, track_id, global_id, best_sim,
            )

        # Persist this track's embedding so future BORN events can match against it
        self.store_embedding(camera_id, track_id, norm_vec, global_id=global_id)

        # Update the identity → [cam:track] list in Redis
        self._update_identity_map(global_id, camera_id, track_id)

        return result

    def get_identity(self, global_id: str) -> list[str]:
        """
        Return all ``cam:track`` tokens linked to this global_id.

        Returns [] if the identity has expired or never existed.
        """
        raw = self._r.get(self._identity_key(global_id))
        if raw is None:
            return []
        return json.loads(raw)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        vec = np.array(vec, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    @staticmethod
    def _embed_key(camera_id: str, track_id: int) -> str:
        return f"embed:{camera_id}:{track_id}"

    @staticmethod
    def _identity_key(global_id: str) -> str:
        return f"identity:{global_id}"

    def _fetch_candidates(
        self,
        exclude_camera: str,
        now_ms: float,
    ) -> list[EmbeddingRecord]:
        """Return all live embedding records that belong to other cameras."""
        pattern = "embed:*"
        # KEYS is fine for small deployments; use SCAN for production scale
        all_keys = []
        cursor = 0
        while True:
            cursor, batch = self._r.scan(cursor, match=pattern, count=100)
            all_keys.extend(batch)
            if cursor == 0:
                break
        records: list[EmbeddingRecord] = []

        for raw_key in all_keys:
            key_str = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            parts   = key_str.split(":")           # embed : camera_id : track_id
            if len(parts) < 3:
                continue
            cam = parts[1]
            if cam == exclude_camera:
                continue

            raw = self._r.get(key_str)
            if raw is None:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            age_ms = now_ms - data.get("timestamp_ms", 0)
            if age_ms > self._window_ms:
                continue   # outside the 5-second re-id window

            records.append(EmbeddingRecord(
                camera_id    = data["camera_id"],
                track_id     = int(data["track_id"]),
                embedding    = data["embedding"],
                timestamp_ms = data["timestamp_ms"],
                global_id    = data.get("global_id"),
            ))

        return records

    def _update_identity_map(
        self,
        global_id: str,
        camera_id: str,
        track_id:  int,
    ) -> None:
        token = f"{camera_id}:{track_id}"
        key   = self._identity_key(global_id)
        raw   = self._r.get(key)
        tokens: list[str] = json.loads(raw) if raw else []
        if token not in tokens:
            tokens.append(token)
        self._r.setex(key, IDENTITY_TTL_SECONDS, json.dumps(tokens))
