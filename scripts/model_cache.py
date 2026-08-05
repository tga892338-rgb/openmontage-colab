"""Lightweight adapter/model availability checks used by Colab runners.

This module intentionally does not download models. It reports whether the
optional Python packages and local adapter modules needed by each configured
role are importable.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict


def _check_import(module: str) -> Dict[str, Any]:
    try:
        importlib.import_module(module)
        return {"available": True, "module": module}
    except Exception as exc:
        return {"available": False, "module": module, "error": f"{type(exc).__name__}: {exc}"}


def health_check() -> Dict[str, Dict[str, Any]]:
    """Return non-destructive availability checks for the project's adapters."""
    checks = {
        "planner": "src.adapters.planner_qwen3",
        "tts": "src.adapters.tts_kokoro",
        "image": "src.adapters.image_qwen",
        "video": "src.adapters.video_diffusion",
        "music": "src.adapters.music_musicgen",
        "montage": "src.adapters.montage_ffmpeg",
    }
    results: Dict[str, Dict[str, Any]] = {}
    for role, module in checks.items():
        results[role] = _check_import(module)
    return results
