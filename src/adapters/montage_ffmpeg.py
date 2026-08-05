"""FFmpeg-based montage/editing adapter."""

import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from src.validators import is_valid_video

logger = logging.getLogger(__name__)


class FFmpegMontageEditor:
    """Montage editor using FFmpeg. Exposes a `run(project_path, assets=...)` method
    compatible with Pipeline.run, delegating the heavy lifting to compose_montage.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize FFmpeg montage editor.
        
        Args:
            model_name: adapter model identifier
            config: Adapter config from models.yaml
        """
        self.model_name = model_name
        self.config = config or {}
        self.available = self._check_available()
        self.name = "FFmpegMontageEditor"
    
    def _check_available(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(self, project_path: str, assets: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Compatibility wrapper used by Pipeline.run.

        The pipeline expects an editor.run(project_path, assets=...) call. Internally
        this adapter implements compose_montage(...); this wrapper adapts that contract.
        """
        assets = assets or {}
        clips = []
        # If images are present, create very short image clips with a duration each
        images = None
        if isinstance(assets.get("images"), dict):
            images = assets.get("images").get("paths")
        elif isinstance(assets.get("images"), list):
            images = assets.get("images")
        if images:
            for p in images:
                clips.append({"path": p, "start": 0, "duration": 2})
        # If a direct video clip provided
        clip = assets.get("clip")
        if isinstance(clip, dict) and clip.get("path"):
            clips.append({"path": clip.get("path"), "start": 0, "duration": clip.get("duration", 5)})
        audio = (assets.get("voice") or {}).get("path") if isinstance(assets.get("voice"), dict) else None
        return self.compose_montage(clips, audio_path=audio)

    def compose_montage(
        self,
        clips: list,
        audio_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compose clips into final montage.
        
        Args:
            clips: List of clip dicts with path, start, duration
            audio_path: Optional audio track
            metadata: Optional metadata (subtitles, etc.)
        
        Returns:
            Dict with output_path and metadata
        """
        if not self.available:
            logger.warning("FFmpeg not available, returning mock result")
            return {
                "output_path": "mock_montage.mp4",
                "duration": 60.0,
                "clips_count": len(clips),
                "status": "mock"
            }
        
        try:
            output_path = "final_montage.mp4"
            
            logger.info(f"Composing {len(clips)} clips into montage")
            
            # Simple concat: write filelist
            filelist = "concat_list.txt"
            with open(filelist, "w") as f:
                for clip in clips:
                    clip_path = clip.get("path", "")
                    f.write(f"file '{clip_path}'\n")
            
            # FFmpeg concat filter
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", filelist,
                "-c", "copy",
                output_path
            ]
            
            # Add audio if provided
            if audio_path:
                cmd.extend(["-i", audio_path, "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0"])
            
            subprocess.run(cmd, check=True, capture_output=True)
            # Validate the produced artifact before claiming success
            validation = is_valid_video(output_path)
            if validation.get("valid"):
                return {
                    "output_path": output_path,
                    "clips_count": len(clips),
                    "audio_track": audio_path is not None,
                    "status": "success",
                    "validation": validation,
                }
            else:
                logger.error("Montage created but failed validation: %s", validation)
                return {
                    "output_path": output_path,
                    "clips_count": len(clips),
                    "audio_track": audio_path is not None,
                    "status": "failed_validation",
                    "validation": validation,
                }
        except Exception as e:
            logger.error(f"FFmpeg montage failed: {e}")
            return {
                "output_path": None,
                "error": str(e),
                "status": "failed"
            }


class MoviePyMontageEditor:
    """Montage editor using MoviePy (lightweight fallback).

    Provides a `run(project_path, assets=...)` entrypoint compatible with Pipeline.run
    and delegates composition work to compose_montage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MoviePy montage editor.
        
        Args:
            config: Adapter config from models.yaml
        """
        self.config = config or {}
        self.available = self._check_available()
        self.name = "MoviePyMontageEditor"
    
    def _check_available(self) -> bool:
        """Check if MoviePy is available."""
        try:
            import moviepy
            return True
        except ImportError:
            return False

    def run(self, project_path: str, assets: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        assets = assets or {}
        clips = []
        images = None
        if isinstance(assets.get("images"), dict):
            images = assets.get("images").get("paths")
        elif isinstance(assets.get("images"), list):
            images = assets.get("images")
        if images:
            for p in images:
                clips.append({"path": p, "start": 0, "duration": 2})
        clip = assets.get("clip")
        if isinstance(clip, dict) and clip.get("path"):
            clips.append({"path": clip.get("path"), "start": 0, "duration": clip.get("duration", 5)})
        audio = (assets.get("voice") or {}).get("path") if isinstance(assets.get("voice"), dict) else None
        return self.compose_montage(clips, audio_path=audio)

    def compose_montage(
        self,
        clips: list,
        audio_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compose clips into final montage using MoviePy.
        
        Args:
            clips: List of clip dicts with path, start, duration
            audio_path: Optional audio track
            metadata: Optional metadata
        
        Returns:
            Dict with output_path and metadata
        """
        if not self.available:
            logger.warning("MoviePy not available, returning mock result")
            return {
                "output_path": "mock_montage.mp4",
                "status": "mock"
            }
        
        try:
            from moviepy.editor import concatenate_videoclips, VideoFileClip, CompositeAudioClip
            
            logger.info(f"Composing {len(clips)} clips with MoviePy")
            
            # Load video clips
            video_clips = []
            for clip in clips:
                clip_path = clip.get("path")
                if clip_path:
                    vclip = VideoFileClip(clip_path)
                    video_clips.append(vclip)
            
            if not video_clips:
                return {"output_path": None, "error": "No clips provided", "status": "failed"}
            
            # Concatenate
            final = concatenate_videoclips(video_clips)
            
            # Add audio if provided
            if audio_path:
                from moviepy.audio.AudioFileClip import AudioFileClip
                audio = AudioFileClip(audio_path)
                if final.duration > audio.duration:
                    audio = audio.speedx(final.duration / audio.duration)
                final = final.set_audio(audio)
            
            output_path = "final_montage.mp4"
            final.write_videofile(output_path, verbose=False, logger=None)
            
            return {
                "output_path": output_path,
                "duration": final.duration,
                "fps": final.fps,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"MoviePy montage failed: {e}")
            return {
                "output_path": None,
                "error": str(e),
                "status": "failed"
            }
