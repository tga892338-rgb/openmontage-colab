"""
TTS Quality Comparison Test: Qwen3-TTS vs Chatterbox-Turbo

This script tests two production-grade open-source TTS models on Colab GPU.
NO fallback to pyttsx3. Real inference only.

Usage:
  python tools/colab_tts_quality_test.py

Expected outputs:
  - projects/colab-tts-quality/qwen3_tts_output.wav
  - projects/colab-tts-quality/chatterbox_output.wav
  - projects/colab-tts-quality/tts_quality_report.json
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

# For diagnostics only
try:
    import torch
except ImportError:
    torch = None

NARRATION = """A hundred years ago, humanity looked toward the stars and wondered whether we were alone. Tonight, something answered. The signal came from a world no telescope had ever seen before. And buried inside that transmission was a message meant for us."""

CINEMATIC_INSTRUCTION = "Calm, cinematic narration. Natural pacing. Slight sense of mystery and anticipation. Clear pronunciation. Do not sound like an advertisement."

OUTPUT_DIR = Path("projects/colab-tts-quality")


def ensure_output_dir():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_audio_metadata(wav_path: str) -> Optional[Dict[str, Any]]:
    """Use ffprobe to extract audio metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels,duration,nb_read_frames",
                "-show_entries", "format=duration,size",
                "-of", "json",
                wav_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get("streams"):
                stream = data["streams"][0]
                fmt = data.get("format", {})
                return {
                    "sample_rate": int(stream.get("sample_rate", 0)),
                    "channels": int(stream.get("channels", 0)),
                    "duration": float(stream.get("duration", fmt.get("duration", 0))),
                    "file_size_bytes": int(fmt.get("size", 0))
                }
    except Exception as e:
        print(f"ffprobe error: {e}")
    return None


def test_qwen3_tts() -> Dict[str, Any]:
    """Test Qwen3-TTS-12Hz-0.6B-CustomVoice model."""
    print("\n" + "="*70)
    print("Testing Qwen3-TTS-12Hz-0.6B-CustomVoice")
    print("="*70)
    
    result = {
        "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "device": None,
        "sample_rate": None,
        "duration": None,
        "file_size": None,
        "generation_time": None,
        "gpu_memory_used": None,
        "status": "FAIL",
        "error": None,
        "output_path": str(OUTPUT_DIR / "qwen3_tts_output.wav")
    }
    
    try:
        # Get device info
        if torch:
            result["device"] = f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
            gpu_mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Import Qwen3 TTS
        print("Installing Qwen3-TTS...")
        subprocess.run(
            ["pip", "install", "-q", "git+https://github.com/QwenLM/Qwen3-TTS.git"],
            timeout=120
        )
        
        from qwen3tts.utils import TTS as Qwen3TTS
        
        print("Initializing Qwen3-TTS model...")
        start_time = time.time()
        
        # Initialize model
        tts = Qwen3TTS("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device="cuda" if torch and torch.cuda.is_available() else "cpu")
        
        # Generate audio
        print(f"Generating audio: {len(NARRATION)} characters")
        print(f"With instruction: {CINEMATIC_INSTRUCTION}")
        
        # Qwen3 supports instruction context
        audio_data = tts.synthesize(
            text=NARRATION,
            instruction=CINEMATIC_INSTRUCTION,
            sampling_rate=12000  # Model default
        )
        
        generation_time = time.time() - start_time
        result["generation_time"] = generation_time
        
        # Save audio
        output_path = OUTPUT_DIR / "qwen3_tts_output.wav"
        tts.save_wav(audio_data, str(output_path), sampling_rate=12000)
        print(f"Saved to: {output_path}")
        
        # Validate with ffprobe
        metadata = get_audio_metadata(str(output_path))
        if metadata:
            result["sample_rate"] = metadata["sample_rate"]
            result["duration"] = metadata["duration"]
            result["file_size"] = metadata["file_size_bytes"]
            result["status"] = "PASS"
            print(f"✓ FFmpeg validation PASSED")
            print(f"  Sample rate: {metadata['sample_rate']} Hz")
            print(f"  Duration: {metadata['duration']:.2f}s")
            print(f"  File size: {metadata['file_size_bytes']} bytes")
        else:
            result["error"] = "FFmpeg validation failed"
            result["status"] = "FAIL"
            print(f"✗ FFmpeg validation FAILED")
        
        # GPU memory
        if torch and torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated()
            result["gpu_memory_used"] = (gpu_mem_after - gpu_mem_before) / 1024 / 1024
            print(f"  GPU memory used: {result['gpu_memory_used']:.1f} MB")
        
        print(f"  Generation time: {generation_time:.2f}s")
        
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "FAIL"
        print(f"✗ Qwen3-TTS FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def test_chatterbox_turbo() -> Dict[str, Any]:
    """Test Chatterbox-Turbo model."""
    print("\n" + "="*70)
    print("Testing Chatterbox-Turbo")
    print("="*70)
    
    result = {
        "model": "Chatterbox-Turbo",
        "device": None,
        "sample_rate": None,
        "duration": None,
        "file_size": None,
        "generation_time": None,
        "gpu_memory_used": None,
        "status": "FAIL",
        "error": None,
        "output_path": str(OUTPUT_DIR / "chatterbox_output.wav")
    }
    
    try:
        # Get device info
        if torch:
            result["device"] = f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
            gpu_mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Install and import Chatterbox
        print("Installing Chatterbox...")
        subprocess.run(
            ["pip", "install", "-q", "git+https://github.com/resemble-ai/chatterbox.git"],
            timeout=120
        )
        
        from chatterbox import ChatterboxTurbo
        
        print("Initializing Chatterbox-Turbo model...")
        start_time = time.time()
        
        # Initialize model (loads ResembleAI/chatterbox-turbo)
        tts = ChatterboxTurbo(model_id="ResembleAI/chatterbox-turbo", device="cuda" if torch and torch.cuda.is_available() else "cpu")
        
        # Generate audio
        print(f"Generating audio: {len(NARRATION)} characters")
        
        audio_data, sample_rate = tts.synthesize(
            text=NARRATION,
            voice="default"  # Use default voice
        )
        
        generation_time = time.time() - start_time
        result["generation_time"] = generation_time
        result["sample_rate"] = sample_rate
        
        # Save audio
        output_path = OUTPUT_DIR / "chatterbox_output.wav"
        import soundfile as sf
        sf.write(str(output_path), audio_data, sample_rate)
        print(f"Saved to: {output_path}")
        
        # Validate with ffprobe
        metadata = get_audio_metadata(str(output_path))
        if metadata:
            result["sample_rate"] = metadata["sample_rate"]
            result["duration"] = metadata["duration"]
            result["file_size"] = metadata["file_size_bytes"]
            result["status"] = "PASS"
            print(f"✓ FFmpeg validation PASSED")
            print(f"  Sample rate: {metadata['sample_rate']} Hz")
            print(f"  Duration: {metadata['duration']:.2f}s")
            print(f"  File size: {metadata['file_size_bytes']} bytes")
        else:
            result["error"] = "FFmpeg validation failed"
            result["status"] = "FAIL"
            print(f"✗ FFmpeg validation FAILED")
        
        # GPU memory
        if torch and torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated()
            result["gpu_memory_used"] = (gpu_mem_after - gpu_mem_before) / 1024 / 1024
            print(f"  GPU memory used: {result['gpu_memory_used']:.1f} MB")
        
        print(f"  Generation time: {generation_time:.2f}s")
        
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "FAIL"
        print(f"✗ Chatterbox-Turbo FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def generate_report(qwen_result: Dict[str, Any], chatterbox_result: Dict[str, Any]):
    """Generate comparison report."""
    print("\n" + "="*70)
    print("TTS QUALITY COMPARISON REPORT")
    print("="*70)
    
    report = {
        "narration_length": len(NARRATION),
        "models": {
            "qwen3": qwen_result,
            "chatterbox": chatterbox_result
        }
    }
    
    # Print table
    print("\n" + "-"*100)
    print(f"{'Model':<25} {'Real GPU':<12} {'Duration':<12} {'Sample Rate':<15} {'Gen Time':<12} {'File Size':<12} {'Status':<8}")
    print("-"*100)
    
    for name, res in [("Qwen3-TTS", qwen_result), ("Chatterbox-Turbo", chatterbox_result)]:
        gpu = "Yes" if res["device"] and "CUDA: True" in res["device"] else "No"
        duration = f"{res['duration']:.1f}s" if res['duration'] else "—"
        sr = f"{res['sample_rate']} Hz" if res['sample_rate'] else "—"
        gen_time = f"{res['generation_time']:.2f}s" if res['generation_time'] else "—"
        file_size = f"{res['file_size'] / 1024:.1f} KB" if res['file_size'] else "—"
        status = res['status']
        
        print(f"{name:<25} {gpu:<12} {duration:<12} {sr:<15} {gen_time:<12} {file_size:<12} {status:<8}")
    
    print("-"*100)
    
    # Listening checklist template
    print("\nLISTENING CHECKLIST (manual evaluation in Colab):")
    print("""
For each model, evaluate:
  ✓ Natural voice?
  ✓ Robotic artifacts?
  ✓ Pronunciation clarity?
  ✓ Pacing and pauses?
  ✓ Emotional expression?
  ✓ Background noise?
  ✓ Clipping or distortion?
  ✓ Suitable for YouTube narration?
  ✓ Suitable for long-form content?
""")
    
    # Save report
    report_path = OUTPUT_DIR / "tts_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return report


def main():
    print("TTS Quality Comparison Test")
    print(f"Output directory: {OUTPUT_DIR}")
    
    ensure_output_dir()
    
    # Test both models
    qwen_result = test_qwen3_tts()
    chatterbox_result = test_chatterbox_turbo()
    
    # Generate report
    generate_report(qwen_result, chatterbox_result)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR / 'qwen3_tts_output.wav'} ({qwen_result['status']})")
    print(f"  - {OUTPUT_DIR / 'chatterbox_output.wav'} ({chatterbox_result['status']})")
    print(f"  - {OUTPUT_DIR / 'tts_quality_report.json'}")
    print("\nNext: Load these WAVs in Colab and evaluate audio quality manually.")


if __name__ == "__main__":
    main()
