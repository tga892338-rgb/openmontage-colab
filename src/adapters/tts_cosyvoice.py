"""CosyVoice TTS adapter placeholder.
"""
from typing import Dict, Any
from src.abstractions import VoiceGenerator


class CosyVoiceTTS(VoiceGenerator):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.available = True  # always available due to built-in fallbacks
        self._backend = None
        try:
            import cosyvoice  # hypothetical package
            self._backend = "cosyvoice"
        except Exception:
            try:
                import pyttsx3
                self._backend = "pyttsx3"
            except Exception:
                # Fallback: scipy.io.wavfile to generate silence
                try:
                    import scipy.io.wavfile
                    self._backend = "scipy"
                except Exception:
                    # Last resort: plain file write
                    self._backend = "dummy"

    def run(self, text: str, **kwargs) -> Dict[str, Any]:
        output = kwargs.get("output") or kwargs.get("path") or "projects/sample-project/assets/audio/voice.wav"
        if self._backend == "cosyvoice":
            try:
                import cosyvoice
                cosy = cosyvoice.Client()
                cosy.synthesize(text, output)
                return {"path": output}
            except Exception as e:
                # Fallback to next method
                pass

        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                from pathlib import Path
                p = Path(output)
                p.parent.mkdir(parents=True, exist_ok=True)
                engine.save_to_file(text, str(p))
                engine.runAndWait()
                return {"path": str(p)}
            except Exception:
                pass

        # Fallback: generate silence or dummy WAV using scipy
        try:
            from pathlib import Path
            import numpy as np
            p = Path(output)
            p.parent.mkdir(parents=True, exist_ok=True)
            # Generate 5 seconds of silence at 16kHz
            sample_rate = 16000
            duration = 5
            silence = np.zeros(sample_rate * duration, dtype=np.int16)
            try:
                import scipy.io.wavfile
                scipy.io.wavfile.write(str(p), sample_rate, silence)
            except Exception:
                # If scipy not available, write raw bytes
                p.write_bytes(b"AUDIO_PLACEHOLDER")
            return {"path": str(p), "fallback": True}
        except Exception as e:
            # Last resort: write a dummy file
            from pathlib import Path
            p = Path(output)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("AUDIO_PLACEHOLDER")
            return {"path": str(p), "fallback": True, "error": str(e)}
