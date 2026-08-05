"""CosyVoice TTS adapter with real/fallback mode separation.

In REAL mode (OM_REAL_STRICT=1):
  - Must use an actual neural TTS model (Qwen3-TTS, Chatterbox, etc.)
  - NO pyttsx3 fallback
  - NO silence/dummy audio
  - Raises error if real model unavailable

In MOCK/FALLBACK mode (default):
  - Can use pyttsx3 or other lightweight fallback
  - Marked clearly as FALLBACK, not REAL
"""
import os
import logging
from typing import Dict, Any
from src.abstractions import VoiceGenerator

logger = logging.getLogger(__name__)

REAL_MODE = os.environ.get("OM_REAL_STRICT") == "1"


class CosyVoiceTTS(VoiceGenerator):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self._backend = None
        self._real_backend_available = False
        
        # In REAL mode, we require an actual neural TTS
        if REAL_MODE:
            # Try Qwen3-TTS or Chatterbox first
            try:
                from qwen3tts.utils import TTS as Qwen3TTS
                self._backend = "qwen3"
                self._real_backend_available = True
                logger.info("CosyVoiceTTS: Qwen3-TTS available in REAL mode")
            except ImportError:
                try:
                    from chatterbox import ChatterboxTurbo
                    self._backend = "chatterbox"
                    self._real_backend_available = True
                    logger.info("CosyVoiceTTS: Chatterbox-Turbo available in REAL mode")
                except ImportError:
                    logger.error("CosyVoiceTTS: No real neural TTS available in REAL mode")
                    self._backend = None
                    self._real_backend_available = False
            
            self.available = self._real_backend_available
        else:
            # Fallback mode: try real models first, then pyttsx3
            try:
                from qwen3tts.utils import TTS as Qwen3TTS
                self._backend = "qwen3"
                self._real_backend_available = True
            except ImportError:
                try:
                    from chatterbox import ChatterboxTurbo
                    self._backend = "chatterbox"
                    self._real_backend_available = True
                except ImportError:
                    try:
                        import pyttsx3
                        self._backend = "pyttsx3"
                        self._real_backend_available = False
                    except ImportError:
                        self._backend = None
                        self._real_backend_available = False
            
            self.available = True  # Fallback mode always available (worst case: silence)

    def run(self, text: str, **kwargs) -> Dict[str, Any]:
        output = kwargs.get("output") or kwargs.get("path") or "projects/sample-project/assets/audio/voice.wav"
        
        # REAL mode: only use real neural TTS
        if REAL_MODE:
            if not self._real_backend_available:
                raise RuntimeError(f"CosyVoiceTTS REAL mode failed: No real neural TTS available (backend={self._backend})")
            
            try:
                return self._run_real(text, output)
            except Exception as e:
                logger.error(f"CosyVoiceTTS REAL mode failed: {e}")
                raise RuntimeError(f"CosyVoiceTTS REAL inference failed: {e}")
        else:
            # Fallback mode: try real, then pyttsx3
            if self._real_backend_available:
                try:
                    result = self._run_real(text, output)
                    result["mode"] = "REAL"
                    return result
                except Exception as e:
                    logger.warning(f"CosyVoiceTTS: Real mode failed, falling back to pyttsx3: {e}")
            
            # Fall back to pyttsx3
            try:
                return self._run_fallback_pyttsx3(text, output)
            except Exception:
                # Last resort: silence
                return self._run_fallback_silence(output)

    def _run_real(self, text: str, output: str) -> Dict[str, Any]:
        """Run a real neural TTS model (Qwen3-TTS or Chatterbox)."""
        from pathlib import Path
        
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        if self._backend == "qwen3":
            from qwen3tts.utils import TTS as Qwen3TTS
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = Qwen3TTS("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device=device)
            audio_data = tts.synthesize(
                text=text,
                instruction="Clear, natural speech. No advertisement tone.",
                sampling_rate=12000
            )
            tts.save_wav(audio_data, str(p), sampling_rate=12000)
            return {"path": str(p), "mode": "REAL", "model": "Qwen3-TTS"}
        
        elif self._backend == "chatterbox":
            from chatterbox import ChatterboxTurbo
            import soundfile as sf
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = ChatterboxTurbo(model_id="ResembleAI/chatterbox-turbo", device=device)
            audio_data, sample_rate = tts.synthesize(text=text, voice="default")
            sf.write(str(p), audio_data, sample_rate)
            return {"path": str(p), "mode": "REAL", "model": "Chatterbox-Turbo", "sample_rate": sample_rate}
        
        else:
            raise RuntimeError(f"Unknown real backend: {self._backend}")

    def _run_fallback_pyttsx3(self, text: str, output: str) -> Dict[str, Any]:
        """Fallback to pyttsx3 (lightweight CPU-based TTS)."""
        import pyttsx3
        from pathlib import Path
        
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        engine.save_to_file(text, str(p))
        engine.runAndWait()
        logger.warning(f"CosyVoiceTTS: Using fallback pyttsx3 (not real neural TTS)")
        return {"path": str(p), "mode": "FALLBACK", "fallback": True, "backend": "pyttsx3"}

    def _run_fallback_silence(self, output: str) -> Dict[str, Any]:
        """Last resort: generate silence."""
        from pathlib import Path
        import numpy as np
        
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        sample_rate = 16000
        duration = 5
        silence = np.zeros(sample_rate * duration, dtype=np.int16)
        
        try:
            import scipy.io.wavfile
            scipy.io.wavfile.write(str(p), sample_rate, silence)
        except Exception:
            p.write_bytes(b"SILENCE_PLACEHOLDER")
        
        logger.error(f"CosyVoiceTTS: All TTS backends failed, using silence fallback")
        return {"path": str(p), "mode": "FALLBACK", "fallback": True, "backend": "silence"}
