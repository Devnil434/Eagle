"""
Detection Overlay Theme Management API

Endpoints:
  GET    /themes                    - List all themes (presets + custom)
  GET    /themes/{name}             - Get a specific theme
  POST   /themes                    - Save a custom theme
  DELETE /themes/{name}             - Delete a custom theme
"""
from __future__ import annotations
import logging
import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/themes", tags=["themes"])

THEME_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config", "themes")
THEMES_FILE = os.path.join(THEME_DIR, "custom_themes.json")

os.makedirs(THEME_DIR, exist_ok=True)

PRESETS = {
  "default": {
    "name": "Default",
    "description": "Standard surveillance overlay",
    "boundingBoxColors": {"palette": ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7"], "opacity": 0.8, "borderWidth": 4},
    "label": {"visible": True, "fontSize": 12, "fontFamily": "monospace", "backgroundColor": "rgba(0,0,0,0.6)", "textColor": "#ffffff"},
    "confidence": {"visible": True, "mode": "percentage", "color": "#ffffff"},
    "trackingTrails": {"enabled": True, "maxLength": 20, "trailOpacity": 0.5, "trailWidth": 2},
    "accessibility": {"highContrast": False, "reducedMotion": False, "largeText": False},
  },
  "professional": {
    "name": "Professional",
    "description": "Clean, muted palette for long shifts",
    "boundingBoxColors": {"palette": ["#f87171", "#60a5fa", "#4ade80", "#fbbf24", "#c084fc"], "opacity": 0.6, "borderWidth": 3},
    "label": {"visible": True, "fontSize": 11, "fontFamily": "sans-serif", "backgroundColor": "rgba(30,30,30,0.7)", "textColor": "#e0e0e0"},
    "confidence": {"visible": True, "mode": "percentage", "color": "#d1d5db"},
    "trackingTrails": {"enabled": True, "maxLength": 15, "trailOpacity": 0.4, "trailWidth": 2},
    "accessibility": {"highContrast": False, "reducedMotion": False, "largeText": False},
  },
  "highContrast": {
    "name": "High Contrast",
    "description": "Maximum visibility for accessibility",
    "boundingBoxColors": {"palette": ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff"], "opacity": 1.0, "borderWidth": 5},
    "label": {"visible": True, "fontSize": 14, "fontFamily": "sans-serif", "backgroundColor": "rgba(0,0,0,0.9)", "textColor": "#ffffff", "fontWeight": "bold"},
    "confidence": {"visible": True, "mode": "bar", "color": "#00ff00"},
    "trackingTrails": {"enabled": True, "maxLength": 25, "trailOpacity": 0.8, "trailWidth": 3},
    "accessibility": {"highContrast": True, "reducedMotion": False, "largeText": True},
  },
  "minimal": {
    "name": "Minimal",
    "description": "Subtle overlay with reduced visual clutter",
    "boundingBoxColors": {"palette": ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#9333ea"], "opacity": 0.4, "borderWidth": 2},
    "label": {"visible": True, "fontSize": 10, "fontFamily": "monospace", "backgroundColor": "transparent", "textColor": "#ffffff"},
    "confidence": {"visible": False, "mode": "off", "color": "#ffffff"},
    "trackingTrails": {"enabled": False, "maxLength": 10, "trailOpacity": 0.3, "trailWidth": 1},
    "accessibility": {"highContrast": False, "reducedMotion": True, "largeText": False},
  },
  "nightMode": {
    "name": "Night Mode",
    "description": "Dimmed palette for dark environments",
    "boundingBoxColors": {"palette": ["#ff6b6b", "#74b9ff", "#55efc4", "#ffeaa7", "#a29bfe"], "opacity": 0.5, "borderWidth": 3},
    "label": {"visible": True, "fontSize": 12, "fontFamily": "monospace", "backgroundColor": "rgba(10,10,20,0.8)", "textColor": "#a0a0b0"},
    "confidence": {"visible": True, "mode": "percentage", "color": "#a0a0b0"},
    "trackingTrails": {"enabled": True, "maxLength": 15, "trailOpacity": 0.3, "trailWidth": 2},
    "accessibility": {"highContrast": False, "reducedMotion": False, "largeText": False},
  },
  "colorBlind": {
    "name": "Color Blind Friendly",
    "description": "Pattern-based cues for color vision deficiency",
    "boundingBoxColors": {"palette": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2"], "opacity": 0.85, "borderWidth": 4},
    "label": {"visible": True, "fontSize": 13, "fontFamily": "sans-serif", "backgroundColor": "rgba(0,0,0,0.7)", "textColor": "#ffffff"},
    "confidence": {"visible": True, "mode": "bar", "color": "#56b4e9"},
    "trackingTrails": {"enabled": True, "maxLength": 20, "trailOpacity": 0.6, "trailWidth": 3},
    "accessibility": {"highContrast": True, "reducedMotion": False, "largeText": True},
  },
}


class ThemeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    boundingBoxColors: dict
    label: dict
    confidence: dict
    trackingTrails: dict
    accessibility: dict


class ThemeResponse(BaseModel):
    name: str
    boundingBoxColors: dict
    label: dict
    confidence: dict
    trackingTrails: dict
    accessibility: dict
    isCustom: bool = False


def _load_custom_themes() -> dict:
    if not os.path.exists(THEMES_FILE):
        return {}
    try:
        with open(THEMES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_custom_themes(themes: dict):
    os.makedirs(os.path.dirname(THEMES_FILE), exist_ok=True)
    with open(THEMES_FILE, "w", encoding="utf-8") as f:
        json.dump(themes, f, indent=2)


@router.get("", response_model=dict[str, ThemeResponse])
def list_themes():
    customs = _load_custom_themes()
    result = {}
    for k, v in PRESETS.items():
        result[k] = ThemeResponse(name=v.get("name", k), boundingBoxColors=v["boundingBoxColors"], label=v["label"], confidence=v["confidence"], trackingTrails=v["trackingTrails"], accessibility=v["accessibility"], isCustom=False)
    for k, v in customs.items():
        result[k] = ThemeResponse(name=v.get("name", k), boundingBoxColors=v["boundingBoxColors"], label=v["label"], confidence=v["confidence"], trackingTrails=v["trackingTrails"], accessibility=v["accessibility"], isCustom=True)
    return result


@router.get("/{theme_name}", response_model=ThemeResponse)
def get_theme(theme_name: str):
    all_themes = {**PRESETS, **_load_custom_themes()}
    if theme_name not in all_themes:
        raise HTTPException(status_code=404, detail=f"Theme {theme_name!r} not found")
    t = all_themes[theme_name]
    is_custom = theme_name in _load_custom_themes()
    return ThemeResponse(name=t.get("name", theme_name), boundingBoxColors=t["boundingBoxColors"], label=t["label"], confidence=t["confidence"], trackingTrails=t["trackingTrails"], accessibility=t["accessibility"], isCustom=is_custom)


@router.post("", response_model=ThemeResponse, status_code=201)
def save_theme(body: ThemeRequest):
    customs = _load_custom_themes()
    customs[body.name] = {
        "name": body.name,
        "boundingBoxColors": body.boundingBoxColors,
        "label": body.label,
        "confidence": body.confidence,
        "trackingTrails": body.trackingTrails,
        "accessibility": body.accessibility,
    }
    _save_custom_themes(customs)
    logger.info("Custom theme saved  name=%s", body.name)
    return ThemeResponse(name=body.name, boundingBoxColors=body.boundingBoxColors, label=body.label, confidence=body.confidence, trackingTrails=body.trackingTrails, accessibility=body.accessibility, isCustom=True)


@router.delete("/{theme_name}")
def delete_theme(theme_name: str):
    customs = _load_custom_themes()
    if theme_name not in customs:
        raise HTTPException(status_code=404, detail=f"Custom theme {theme_name!r} not found")
    del customs[theme_name]
    _save_custom_themes(customs)
    logger.info("Custom theme deleted  name=%s", theme_name)
    return {"deleted": theme_name}
