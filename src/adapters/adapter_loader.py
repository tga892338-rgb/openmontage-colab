"""Loads adapters for roles based on the model registry and selector."""
from typing import Optional
from importlib import import_module
from pathlib import Path
import logging
import os
from ..registry import load_registry
from ..selector import ModelSelector

log = logging.getLogger(__name__)

# mapping from model identifiers to adapter module names (snake_case)
# Include common aliases so the registry can use expressive ids while the
# adapter loader maps them to actual adapter modules already present.
MODEL_TO_ADAPTER = {
    "qwen3": ("planner_qwen3", "PlannerQwen3"),
    "qwen-3": ("planner_qwen3", "PlannerQwen3"),
    "flan-t5-small": ("planner_transformers", "PlannerTransformers"),
    "faster-whisper": ("transcriber_faster_whisper", "FasterWhisperTranscriber"),
    "faster_whisper": ("transcriber_faster_whisper", "FasterWhisperTranscriber"),
    "whisper": ("transcriber_whisper", "WhisperTranscriber"),
    "cosyvoice-3.0": ("tts_cosyvoice", "CosyVoiceTTS"),
    "kokoro-82m": ("tts_cosyvoice", "CosyVoiceTTS"),
    "qwen-image": ("image_qwen", "QwenImageGenerator"),
    "flux.2-klein-4b": ("adapter_stub", "ImageStub"),
    "flux.2-dev": ("adapter_stub", "ImageStub"),
    "wan-2.2": ("video_wan", "WanVideoGenerator"),
    "wan2.2-ti2v-5b": ("video_wan", "WanVideoGenerator"),
    "ltx-video": ("adapter_stub", "VideoStub"),
    "ffmpeg": ("montage_ffmpeg", "FFmpegMontageEditor"),
    "moviepy": ("montage_ffmpeg", "MoviePyMontageEditor"),
    "remotion": ("montage_ffmpeg", "MoviePyMontageEditor"),
    "real-esrgan": ("enhancement_esrgan", "RealESRGANEnhancer"),
    "rife": ("enhancement_esrgan", "RIFEInterpolator"),
    "musicgen": ("music_musicgen", "MusicGenMusicGenerator"),
    "stable-audio": ("music_musicgen", "StableAudioMusicGenerator"),
    "stable-audio-open-small": ("music_musicgen", "StableAudioMusicGenerator"),
    "stable-audio-3-medium": ("music_musicgen", "StableAudioMusicGenerator"),
    "comfyui": ("workflow_comfyui", "ComfyUIWorkflow"),
    # planner/transcriber/tts aliases mapped to best-known local adapters or fallbacks
    "deepseek-r1-distill": ("planner_qwen3", "PlannerQwen3"),
    "distil-whisper": ("transcriber_faster_whisper", "FasterWhisperTranscriber"),
    "gemma3-27b-it": ("planner_qwen3", "PlannerQwen3"),
    "moss-tts": ("tts_cosyvoice", "CosyVoiceTTS"),
    "parler-tts": ("tts_cosyvoice", "CosyVoiceTTS"),
    "hunyuanvideo-1.5": ("video_wan", "WanVideoGenerator"),
    "seamless-m4t-v2": ("transcriber_faster_whisper", "FasterWhisperTranscriber"),
}


def get_component(role: str, profile: str = "balanced"):
    reg = load_registry().data
    sel = ModelSelector(reg)
    pick = sel.pick(role, profile=profile)
    choice = pick.get("choice")

    if not choice:
        log.debug("No choice for role %s", role)
        return None

    # translate choice to adapter module
    entry = MODEL_TO_ADAPTER.get(choice)
    if not entry:
        log.debug("No adapter mapping for choice %s (role=%s)", choice, role)
        return None

    module_name, class_name = entry
    full_module = f"src.adapters.{module_name}"
    # Avoid importing adapters by default to prevent heavy model installs/downloads during discovery.
    # Set OM_LOAD_ADAPTERS=1 in the environment to allow importing and instantiating adapters.
    if os.environ.get("OM_LOAD_ADAPTERS") != "1":
        log.debug("OM_LOAD_ADAPTERS not set; skipping adapter import for %s", full_module)
        return None
    try:
        mod = import_module(full_module)
        cls = getattr(mod, class_name)
        # pass adapter-specific config from registry if present
        reg = load_registry().data
        adapters_conf = reg.get("adapters", {})
        model_conf = adapters_conf.get(choice, {})
        try:
            inst = cls(choice, config=model_conf or {})
        except Exception as ie:
            # Log and persist instantiation error for diagnosis
            import traceback
            tb = traceback.format_exc()
            log.info("Adapter %s instantiation failed: %s", full_module, ie)
            # Append to projects/smoke-report/artifacts/real_init.log for easier debugging
            try:
                art = Path('projects') / 'smoke-report' / 'artifacts'
                art.mkdir(parents=True, exist_ok=True)
                with open(art / 'real_init.log', 'a', encoding='utf-8') as lf:
                    lf.write(f"=== Adapter instantiation error: {full_module} ===\n")
                    lf.write(tb + "\n")
            except Exception:
                pass
            return None
        if getattr(inst, "available", True):
            return inst
        else:
            log.info("Adapter %s available=False; will fallback", full_module)
            return None
    except Exception as e:
        log.info("Failed to load adapter %s: %s", full_module, e)
        return None
