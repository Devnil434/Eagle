"""
VLM captioning layer.  Event-triggered — NEVER called every frame.

Providers:
  mock   → deterministic from ActionHint (CI, no GPU needed)
  ollama → LLaVA-Next via local Ollama server
  qwen   → Qwen-VL-Chat via HuggingFace (stub, requires GPU)
"""
from __future__ import annotations

import abc
import base64
import logging
import os

import cv2
import numpy as np

from libs.schemas.memory import ActionHint
from services.reasoning.prompts import build_captioning_prompt

logger = logging.getLogger(__name__)

VLM_PROVIDER  = os.getenv("VLM_PROVIDER",  "mock")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",   "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llava:latest")
VLM_TIMEOUT   = float(os.getenv("VLM_TIMEOUT", "30"))


# ── Exceptions ────────────────────────────────────────────────────────────────

class VLMTimeoutError(RuntimeError):
    pass

class VLMUnavailableError(RuntimeError):
    pass


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseCaptioner(abc.ABC):
    @abc.abstractmethod
    def caption(
        self,
        frame:       np.ndarray,
        action_hint: ActionHint = ActionHint.UNKNOWN,
        allowed_labels: list[str] | None = None,
    ) -> str:
        ...


# ── Mock (deterministic, no network) ─────────────────────────────────────────

class MockVLMCaptioner(BaseCaptioner):
    """Returns scripted captions based on ActionHint. Used in CI and tests."""

    CAPTION_MAP: dict[ActionHint, str] = {
        ActionHint.ZONE_ENTRY:
            "Person steps into restricted area near the door.",
        ActionHint.LINGERING:
            "Person is standing still near the access control point.",
        ActionHint.NEAR_KEYPAD:
            "Person appears to be interacting with the wall-mounted keypad.",
        ActionHint.REPEATED_APPROACH:
            "Person approaches the keypad area for the second time.",
        ActionHint.WALKING:
            "Person is walking through the corridor at normal pace.",
        ActionHint.STANDING:
            "Person is standing still in the hallway.",
    }

    def caption(
        self,
        frame:          np.ndarray,
        action_hint:    ActionHint = ActionHint.UNKNOWN,
        allowed_labels: list[str] | None = None,
    ) -> str:
        return self.CAPTION_MAP.get(
            action_hint,
            "Person is visible in the surveillance frame.",
        )


# ── Ollama (LLaVA-Next) ─────────────────────────────────────────────────────

class OllamaVLMCaptioner(BaseCaptioner):
    """
    Sends a JPEG frame to Ollama's /api/generate endpoint (LLaVA model).
    Requires: `ollama serve` running + `ollama pull llava:latest`.
    """

    def __init__(
        self,
        base_url: str  = OLLAMA_HOST,
        model:    str  = OLLAMA_MODEL,
        timeout:  float = VLM_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def caption(
        self,
        frame:          np.ndarray,
        action_hint:    ActionHint = ActionHint.UNKNOWN,
        allowed_labels: list[str] | None = None,
    ) -> str:
        try:
            import httpx
        except ImportError:
            raise VLMUnavailableError("httpx not installed: pip install httpx")

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")
        jpg_b64 = base64.b64encode(buf).decode()

        prompt = build_captioning_prompt(allowed_labels)

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "images": [jpg_b64],
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except httpx.TimeoutException as e:
            raise VLMTimeoutError(f"Ollama VLM timed out after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise VLMUnavailableError(f"Ollama VLM HTTP error: {e}") from e


# ── Qwen-VL stub ────────────────────────────────────────────────────────────

class QwenVLCaptioner(BaseCaptioner):
    """
    Qwen-VL-Chat via HuggingFace transformers with 4-bit quantisation.
    Requires: pip install transformers accelerate bitsandbytes
    Falls back to MockVLMCaptioner if transformers not installed.
    """

    def __init__(self, model_id: str = "Qwen/Qwen-VL-Chat") -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, load_in_4bit=True
            )
            self._available = True
        except ImportError:
            logger.warning("transformers not installed; falling back to MockVLMCaptioner")
            self._fallback   = MockVLMCaptioner()
            self._available  = False

    def caption(
        self,
        frame:          np.ndarray,
        action_hint:    ActionHint = ActionHint.UNKNOWN,
        allowed_labels: list[str] | None = None,
    ) -> str:
        if not self._available:
            return self._fallback.caption(frame, action_hint, allowed_labels)

        import tempfile
        import pathlib
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, frame)
            tmp_path = f.name

        prompt = build_captioning_prompt(allowed_labels)
        query   = self._tokenizer.from_list_format([
            {"image": tmp_path},
            {"text":  prompt},
        ])
        response, _ = self._model.chat(self._tokenizer, query=query, history=None)
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        return response.strip()


# ── Factory ───────────────────────────────────────────────────────────────────

def get_captioner(provider: str | None = None) -> BaseCaptioner:
    """
    Return the right captioner based on VLM_PROVIDER env var.

    VLM_PROVIDER=mock    → MockVLMCaptioner   (default, CI-safe)
    VLM_PROVIDER=ollama  → OllamaVLMCaptioner
    VLM_PROVIDER=qwen    → QwenVLCaptioner
    """
    p = (provider or VLM_PROVIDER).lower()
    if p == "mock":
        return MockVLMCaptioner()
    if p == "ollama":
        return OllamaVLMCaptioner()
    if p == "qwen":
        return QwenVLCaptioner()
    raise ValueError(f"Unknown VLM_PROVIDER: '{p}'. "
                     f"Choose from: mock, ollama, qwen")
