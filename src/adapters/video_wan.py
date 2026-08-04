"""Wan 2.2 video generator adapter with MoviePy fallback.

Attempts to use a native `wan` package if present. If not, uses MoviePy
to assemble a short clip from provided images or generates a color clip as a lightweight local fallback.
"""
from typing import Dict, Any
from src.abstractions import VideoGenerator
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class WanVideoGenerator(VideoGenerator):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.available = True  # always available due to built-in fallbacks
        self._backend = None
        try:
            import wan  # hypothetical wan python package
            self._backend = "wan"
        except Exception:
            try:
                from moviepy.editor import ImageSequenceClip, ColorClip  # type: ignore
                self._backend = "moviepy"
            except Exception as e:
                log.debug("No wan or moviepy available: %s", e)
                # Fallback: use PIL to generate single-frame video
                self._backend = "placeholder"

    def run(self, prompts: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        out_dir = Path(kwargs.get("output_dir", "projects/sample-project/assets/video"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (kwargs.get("filename", "wan_fallback.mp4"))

        if self._backend == "wan":
            # Placeholder for actual WAN SDK usage
            try:
                wan = __import__("wan")
                # user must implement real call; here we raise to indicate non-implemented
                raise NotImplementedError("WAN SDK call not implemented in adapter; integrate your local WAN client here.")
            except Exception as e:
                log.warning("wan backend failed: %s, falling back", e)
                # Fall through to moviepy/placeholder

        if self._backend == "moviepy":
            try:
                from moviepy.editor import ImageSequenceClip, ColorClip
                images = prompts.get("images") or []
                duration = float(prompts.get("duration", 5.0))
                fps = int(prompts.get("fps", 24))
                if images:
                    clip = ImageSequenceClip(images, fps=fps)
                    clip = clip.set_duration(duration)
                    clip.write_videofile(str(out_path), fps=fps, audio=False, logger=None, verbose=False)
                else:
                    # create a solid color clip as a placeholder
                    clip = ColorClip(size=(1280, 720), color=(20, 20, 40), duration=duration)
                    clip.write_videofile(str(out_path), fps=fps, audio=False, logger=None, verbose=False)
                return {"path": str(out_path)}
            except Exception as e:
                log.warning("moviepy fallback failed: %s, using placeholder", e)
                # Fall through to placeholder

        if self._backend == "placeholder" or self._backend == "moviepy":
            # Last resort: generate a simple MP4 placeholder using ffmpeg or raw bytes
            try:
                import subprocess
                # Generate a 5-second black video using ffmpeg
                duration = prompts.get("duration", 5.0)
                subprocess.run([
                    "ffmpeg", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}",
                    "-pix_fmt", "yuv420p", "-y", str(out_path)
                ], capture_output=True, check=False)
                if out_path.exists():
                    return {"path": str(out_path), "fallback": True}
            except Exception:
                pass

            # If all else fails, write a placeholder
            out_path.write_text("VIDEO_PLACEHOLDER")
            return {"path": str(out_path), "fallback": True}

        raise RuntimeError("Unsupported video backend")
