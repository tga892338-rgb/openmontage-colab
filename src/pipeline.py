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
import os
from typing import Dict, Any

log = logging.getLogger(__name__)

PROJECTS_ROOT = Path("projects")


class Pipeline:
    def __init__(self, mock: bool = True):
        self.registry = load_registry().data
        self.selector = ModelSelector(self.registry)
        self.mock = mock
        # If strict real mode is requested via environment, override any caller-specified
        # mock=True so required stages are resolved against real adapters and missing
        # adapters produce an error rather than silently using mocks.
        if os.environ.get("OM_REAL_STRICT") == "1":
            log.info("OM_REAL_STRICT=1 detected; forcing real adapter resolution (mock=False) for required stages")
            self.mock = False

        # Try to load real adapters via adapter_loader. When not in mock mode,
        # do NOT silently return a mock if an adapter is missing. Instead return None
        # so the caller can explicitly handle/announce missing capabilities.
        from .adapters.adapter_loader import get_component

        # Required components for REAL Stage 1 (must be real models in strict mode)
        required_roles = ("planner", "image_generation", "montage")

        # helper to choose adapter or mock; in OM_REAL_STRICT=1, required roles must
        # be resolved to real adapters even if mock mode was requested by the caller.
        def _get(role: str, mock_cls):
            # If caller explicitly requested mock and we're NOT in strict real mode for this role,
            # return a mock immediately to preserve test/mock behavior.
            if self.mock and not (os.environ.get("OM_REAL_STRICT") == "1" and role in required_roles):
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
            # Required components for REAL Stage 1 (must be real models in strict mode)
            required_roles = ("planner", "image_generation", "montage")
            optional_roles = ("transcriber", "tts", "video_generation", "enhancement", "music")

            missing_required = [name for name, comp in (
                ("planner", self.planner),
                ("image_generation", self.image),
                ("montage", self.editor),
            ) if comp is None]

            missing_optional = [name for name, comp in (
                ("transcriber", self.transcriber),
                ("tts", self.voice),
                ("video_generation", self.video),
                ("enhancement", self.enhancer),
                ("music", self.music),
            ) if comp is None]

            if missing_required or missing_optional:
                if os.environ.get("OM_REAL_STRICT") == "1":
                    # In strict real mode, missing required adapters block the run.
                    if missing_required:
                        raise RuntimeError(f"Required adapters unavailable in strict real mode: {missing_required}")
                    # missing optional components are reported but do not block
                    if missing_optional:
                        log.warning("Optional adapters unavailable in strict real mode: %s", missing_optional)
                else:
                    # Non-strict: substitute mocks for any missing components to allow the pipeline to run
                    all_missing = missing_required + missing_optional
                    log.warning("Required/optional adapters unavailable: %s. Substituting mock implementations to allow the pipeline to run.", all_missing)
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
                    for name in all_missing:
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
        # Planner
        planner_label = getattr(self.planner, 'name', self.planner.__class__.__name__)
        is_mock_planner = isinstance(self.planner, MockPlanner)
        log.info("Running planner (%s): %s", 'mock' if is_mock_planner else 'real', planner_label)
        plan = self.planner.run("[mock subtitles]")
        (artifacts / "scene_plan.json").write_text(json.dumps(plan, indent=2))

        # Voice
        voice_label = getattr(self.voice, 'name', self.voice.__class__.__name__)
        is_mock_voice = isinstance(self.voice, MockVoice)
        log.info("Generating voice (%s): %s", 'mock' if is_mock_voice else 'real', voice_label)
        voice = self.voice.run("This is a mocked narration.", output=str(project / "assets" / "audio" / "voice.wav"))

        # Images
        image_label = getattr(self.image, 'name', self.image.__class__.__name__)
        is_mock_image = isinstance(self.image, MockImageGen)
        log.info("Generating images (%s): %s", 'mock' if is_mock_image else 'real', image_label)
        imgs = self.image.run("A cinematic still", count=2)

        # Video clips
        video_label = getattr(self.video, 'name', self.video.__class__.__name__)
        is_mock_video = isinstance(self.video, MockVideoGen)
        log.info("Generating video clips (%s): %s", 'mock' if is_mock_video else 'real', video_label)
        clip = self.video.run({"prompt": "motion clip"})

        # Compose
        editor_label = getattr(self.editor, 'name', self.editor.__class__.__name__)
        is_mock_editor = isinstance(self.editor, MockEditor)
        log.info("Composing (%s): %s", 'mock' if is_mock_editor else 'real', editor_label)
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
