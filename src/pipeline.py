"""Simple modular pipeline runner with mock mode for OpenMontage."""
from pathlib import Path
from .registry import load_registry
from .selector import ModelSelector
from .mock import (
    MockPlanner,
    MockTranscriber,
    MockVoice,
    MockImageGen,
    MockVideoGen,
    MockEditor,
    MockEnhancer,
    MockMusic,
)
import json
import logging
import time
from typing import Dict, Any

log = logging.getLogger(__name__)

PROJECTS_ROOT = Path("projects")


class Pipeline:
    def __init__(self, mock: bool = True):
        self.registry = load_registry().data
        self.selector = ModelSelector(self.registry)
        self.mock = mock

        # Try to load real adapters via adapter_loader. When not in mock mode,
        # do NOT silently return a mock if an adapter is missing. Instead return None
        # so the caller can explicitly handle/announce missing capabilities.
        from .adapters.adapter_loader import get_component

        # helper to choose adapter or mock
        def _get(role: str, mock_cls):
            if self.mock:
                return mock_cls("mock")
            comp = get_component(role, profile=("mock" if self.mock else "balanced"))
            if comp is None:
                # explicit: adapter absent in this environment
                log.error("No adapter available for role '%s' under profile '%s'", role, ("mock" if self.mock else "balanced"))
                return None
            return comp

        self.planner = _get("planner", MockPlanner)
        self.transcriber = _get("transcriber", MockTranscriber)
        self.voice = _get("tts", MockVoice)
        self.image = _get("image_generation", MockImageGen)
        self.video = _get("video_generation", MockVideoGen)
        self.editor = _get("montage", MockEditor)
        self.enhancer = _get("enhancement", MockEnhancer)
        self.music = _get("music", MockMusic)

        # If not in mock mode, handle missing adapters.
        # When OM_REAL_STRICT=1 is set in the environment, treat missing adapters as a hard failure
        # and raise an error so the caller (real Stage 1 run) stops. Otherwise substitute mocks
        # to allow local architecture/testing runs to continue.
        if not self.mock:
            missing = [name for name, comp in (
                ("planner", self.planner),
                ("transcriber", self.transcriber),
                ("tts", self.voice),
                ("image_generation", self.image),
                ("video_generation", self.video),
                ("montage", self.editor),
                ("enhancement", self.enhancer),
                ("music", self.music),
            ) if comp is None]
            if missing:
                # If strict real-mode requested, fail fast and report which adapters are missing.
                import os
                if os.environ.get("OM_REAL_STRICT") == "1":
                    raise RuntimeError(f"Required adapters unavailable in strict real mode: {missing}")

                log.warning("Required adapters unavailable: %s. Substituting mock implementations to allow the pipeline to run.", missing)
                mock_map = {
                    "planner": MockPlanner,
                    "transcriber": MockTranscriber,
                    "tts": MockVoice,
                    "image_generation": MockImageGen,
                    "video_generation": MockVideoGen,
                    "montage": MockEditor,
                    "enhancement": MockEnhancer,
                    "music": MockMusic,
                }
                attr_map = {
                    "planner": "planner",
                    "transcriber": "transcriber",
                    "tts": "voice",
                    "image_generation": "image",
                    "video_generation": "video",
                    "montage": "editor",
                    "enhancement": "enhancer",
                    "music": "music",
                }
                for name in missing:
                    mock_cls = mock_map.get(name)
                    attr = attr_map.get(name)
                    if mock_cls and attr:
                        setattr(self, attr, mock_cls("mock"))
                # continue without raising

    def run(self, title: str = "sample-project") -> Dict[str, Any]:
        project = PROJECTS_ROOT / title
        project.mkdir(parents=True, exist_ok=True)
        artifacts = project / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        log.info("Running planner (mock)...")
        plan = self.planner.run("[mock subtitles]")
        (artifacts / "scene_plan.json").write_text(json.dumps(plan, indent=2))

        log.info("Generating voice (mock)...")
        voice = self.voice.run("This is a mocked narration.", output=str(project / "assets" / "audio" / "voice.wav"))

        log.info("Generating images (mock)...")
        imgs = self.image.run("A cinematic still", count=2)

        log.info("Generating video clips (mock)...")
        clip = self.video.run({"prompt": "motion clip"})

        log.info("Composing (mock)...")
        final = self.editor.run(str(project), assets={"voice": voice, "images": imgs, "clip": clip})

        duration = time.perf_counter() - start
        summary = {
            "project": title,
            "plan": plan,
            "voice": voice,
            "images": imgs,
            "clip": clip,
            "final": final,
            "duration_s": duration,
        }
        (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def create_sample_run(output_dir: str = "projects/sample-project") -> Dict[str, Any]:
    p = Pipeline(mock=True)
    return p.run("sample-project")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = create_sample_run()
    print("Created sample run:", s.get("final"))
