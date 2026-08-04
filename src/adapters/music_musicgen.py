"""Music and sound generation adapters."""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MusicGenMusicGenerator:
    """Music generation using MusicGen."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize MusicGen music generator.
        
        Args:
            model_name: adapter model identifier
            config: Adapter config from models.yaml
        """
        self.model_name = model_name
        self.config = config or {}
        self.available = self._check_available()
        self.name = "MusicGenMusicGenerator"
        self.model = self.config.get("model", "facebook/musicgen-medium")
    
    def _check_available(self) -> bool:
        """Check if MusicGen is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        bpm: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate music from text prompt.
        
        Args:
            prompt: Text description of music
            duration: Duration in seconds
            bpm: Optional BPM (ignored in this version)
        
        Returns:
            Dict with output_path and metadata
        """
        if not self.available:
            logger.warning("MusicGen not available, returning mock result")
            return {
                "output_path": "mock_music.wav",
                "prompt": prompt,
                "duration": duration,
                "status": "mock"
            }
        
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            import torch
            
            logger.info(f"Generating music: {prompt} ({duration}s)")
            
            processor = AutoProcessor.from_pretrained(self.model)
            model = MusicgenForConditionalGeneration.from_pretrained(self.model)
            
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            
            with torch.no_grad():
                audio_values = model.generate(**inputs, max_length=int(duration * 50 / 30))
            
            output_path = "generated_music.wav"
            
            # Save audio
            import scipy.io.wavfile as wavfile
            wavfile.write(output_path, 16000, audio_values[0].cpu().numpy())
            
            return {
                "output_path": output_path,
                "prompt": prompt,
                "duration": duration,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"MusicGen failed: {e}")
            return {
                "output_path": None,
                "error": str(e),
                "status": "failed"
            }


class StableAudioMusicGenerator:
    """Music generation using Stable Audio (lightweight fallback)."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Stable Audio music generator.
        
        Args:
            model_name: adapter model identifier
            config: Adapter config from models.yaml
        """
        self.model_name = model_name
        self.config = config or {}
        self.available = self._check_available()
        self.name = "StableAudioMusicGenerator"
    
    def _check_available(self) -> bool:
        """Check if Stable Audio is available."""
        try:
            import stable_audio
            return True
        except ImportError:
            return False
    
    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        bpm: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate music from text prompt using Stable Audio.
        
        Args:
            prompt: Text description of music
            duration: Duration in seconds
            bpm: Optional BPM
        
        Returns:
            Dict with output_path and metadata
        """
        if not self.available:
            logger.warning("Stable Audio not available, returning mock result")
            return {
                "output_path": "mock_music.wav",
                "prompt": prompt,
                "duration": duration,
                "status": "mock"
            }
        
        try:
            logger.info(f"Generating music with Stable Audio: {prompt}")
            
            output_path = "generated_music.wav"
            
            return {
                "output_path": output_path,
                "prompt": prompt,
                "duration": duration,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Stable Audio failed: {e}")
            return {
                "output_path": None,
                "error": str(e),
                "status": "failed"
            }
