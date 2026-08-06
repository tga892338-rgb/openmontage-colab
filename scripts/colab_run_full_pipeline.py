"""
Orchestrate full pipeline inside a Colab-like environment using isolated venvs.
This script is intended to be invoked inside a Colab notebook cell (or any Linux/GPU host).
It centralizes CUDA detection, venv creation, optional torch wheel installation, dependency
installation (in venvs), per-shot visual generation, TTS, MusicGen, and final assembly.

Usage:
    python scripts/colab_run_full_pipeline.py [--project PROJECT] [--tts qwen|chatterbox] [--shots-json PATH] [--dry-run]

Notes:
- In --dry-run mode this creates placeholder assets and assembles them instead of
  installing heavy model packages.
- The script is careful to install torch wheels inside each venv when possible to
  avoid contaminating the notebook kernel.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


import subprocess

def run(cmd, env=None, check=True):
    """Run a shell command and print output. On failure, print the captured output for easier debugging."""
    print('\n>>> RUN:', cmd)
    try:
        completed = subprocess.run(cmd, shell=True, env=env, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.stdout:
            print(completed.stdout)
    except subprocess.CalledProcessError as e:
        print('--- COMMAND FAILED ---')
        print('Command:', cmd)
        print('Return code:', e.returncode)
        if hasattr(e, 'stdout') and e.stdout:
            print('Output:\n', e.stdout)
        raise


def detect_cuda_tag():
    """Return a pip wheel tag like 'cu118', 'cu121', 'cu128' or 'cpu'."""
    try:
        out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, text=True)
        # Look for common CUDA mentions
        if 'CUDA Version: 11.8' in out or '11.8' in out:
            return 'cu118'
        if 'CUDA Version: 12.8' in out or '12.8' in out:
            return 'cu128'
        if 'CUDA Version: 12.1' in out or '12.1' in out:
            return 'cu121'
        # Fallback heuristic
        if 'Compute Capability' in out or 'Tesla' in out or 'T4' in out:
            return 'cu118'
    except Exception:
        pass
    return 'cpu'


def make_venv(venv_path: Path):
    cmd = f"python -m venv {shlex.quote(str(venv_path))}"
    run(cmd)
    py = venv_path / 'bin' / 'python' if os.name != 'nt' else venv_path / 'Scripts' / 'python.exe'
    return py


SIMULATE_INSTALL = False

def pip_install_requirements(py: Path, reqs_path: Path):
    """Install requirements into a venv python. In simulate mode, only print commands."""
    cmd_up = f"{shlex.quote(str(py))} -m pip install --upgrade pip setuptools wheel"
    cmd_req = f"{shlex.quote(str(py))} -m pip install -r {shlex.quote(str(reqs_path))}"
    if SIMULATE_INSTALL:
        print('SIMULATE:', cmd_up)
        print('SIMULATE:', cmd_req)
        return
    run(cmd_up)
    run(cmd_req)


def install_torch_wheel(py: Path, tag: str):
    """Install a compatible torch + vision/audio wheels into the venv python if tag is not 'cpu'.

    Installs torch, torchvision, and torchaudio from the official PyTorch wheel index for the
    detected CUDA tag. On CPU-only nodes this is skipped.
    """
    if tag == 'cpu':
        print('Skipping torch wheel install (cpu tag)')
        return
    url = f"https://download.pytorch.org/whl/{tag}/torch_stable.html"
    # Install torch, torchvision, torchaudio from the same wheel index to avoid ABI mismatches
    cmds = [
        f"{shlex.quote(str(py))} -m pip install --upgrade pip setuptools wheel", 
        f"{shlex.quote(str(py))} -m pip install --no-cache-dir -f {shlex.quote(url)} torch torchvision torchaudio"
    ]
    print(f'Installing torch/vision/torchaudio wheels for tag {tag} from {url}')
    if SIMULATE_INSTALL:
        for c in cmds:
            print('SIMULATE:', c)
        return
    for c in cmds:
        run(c)


def run_visual_generator(py_cmd: str, shot_json: str, out_path: str, shot_id: str = None):
    cmd = f"{shlex.quote(py_cmd)} {shlex.quote(str(ROOT / 'scripts' / 'generate_shot_visual.py'))} --shot-json {shlex.quote(shot_json)} --output {shlex.quote(out_path)}"
    if shot_id:
        cmd += f" --shot-id {shlex.quote(shot_id)}"
    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='projects/first-creative-video')
    parser.add_argument('--tts', choices=['qwen', 'chatterbox'], default='qwen')
    parser.add_argument('--shots-json', default='projects/first-creative-video/shot_plan.json')
    parser.add_argument('--dry-run', action='store_true', help='Skip heavy installs and use placeholders')
    parser.add_argument('--simulate', action='store_true', help='Simulate installs (print commands) without running pip)')
    args = parser.parse_args()
    global SIMULATE_INSTALL
    SIMULATE_INSTALL = bool(args.simulate)

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    shots_json_path = Path(args.shots_json)
    if not shots_json_path.exists():
        raise FileNotFoundError(f'Shots JSON not found: {shots_json_path}')

    # Run preflight check to print GPU/CUDA and recommended wheel when doing a real run
    preflight = ROOT / 'scripts' / 'colab_preflight_check.py'
    if preflight.exists() and not args.dry_run:
        print('\nRunning preflight check...')
        try:
            run(f"{shlex.quote(str(sys.executable))} {shlex.quote(str(preflight))}")
        except Exception as e:
            print('Preflight check failed:', e)
            # continue; don't block orchestration on preflight failures

    shots = json.loads(shots_json_path.read_text(encoding='utf-8'))

    venvs = project / 'venvs'
    venvs.mkdir(parents=True, exist_ok=True)

    # Venvs
    tts_venv = venvs / f'tts_{args.tts}'
    music_venv = venvs / 'musicgen'
    vision_venv = venvs / 'vision'

    if args.dry_run:
        print('DRY RUN: creating placeholders and skipping heavy installs')
        # Create placeholder assets and short silent audio
        run(f"{sys.executable} scripts/make_placeholders.py")
        # Assemble using existing assembler
        run(f"{sys.executable} scripts/assemble_montage.py --project {shlex.quote(str(project))}")
        print('DRY_RUN complete. Final video (placeholder) at', project / 'final.mp4')
        return

    # Real run: detect CUDA tag for wheel selection
    cuda_tag = detect_cuda_tag()
    print('Detected CUDA tag:', cuda_tag)

    # Create venvs
    print('Creating venvs...')
    tts_py = make_venv(tts_venv)
    music_py = make_venv(music_venv)
    vision_py = make_venv(vision_venv)

    # Install torch wheel first (into each venv that needs it), then requirements
    # TTS
    reqs_tts = ROOT / 'requirements-tts.txt'
    if reqs_tts.exists():
        install_torch_wheel(tts_py, cuda_tag)
        pip_install_requirements(tts_py, reqs_tts)
    else:
        print('No requirements-tts.txt found; skipping TTS install')

    # Music
    reqs_music = ROOT / 'requirements-music.txt'
    if reqs_music.exists():
        install_torch_wheel(music_py, cuda_tag)
        pip_install_requirements(music_py, reqs_music)
    else:
        print('No requirements-music.txt found; skipping Music install')

    # Vision (optional)
    reqs_vision = ROOT / 'requirements-vision.txt'
    if reqs_vision.exists():
        install_torch_wheel(vision_py, cuda_tag)
        pip_install_requirements(vision_py, reqs_vision)
    else:
        print('No requirements-vision.txt found; vision generator will run with system python or the TTS venv if available')

    # Generate visuals per shot
    assets = project / 'assets'
    assets.mkdir(parents=True, exist_ok=True)

    # When simulating, create placeholder assets to avoid invoking heavy generators
    if SIMULATE_INSTALL:
        print('SIMULATE mode: creating placeholder visuals and audio')
        run(f"{sys.executable} scripts/make_placeholders.py")

    for shot in shots:
        shot_id = shot.get('shot_id', 'shot')
        out = assets / f"{shot_id}.jpg"
        if SIMULATE_INSTALL:
            if out.exists():
                print('SIMULATE: placeholder exists for', out)
            else:
                # fallback: touch the file to avoid downstream failure
                out.write_bytes(b'')
                print('SIMULATE: created empty placeholder', out)
            continue
        # Prefer vision venv if it has requirements, else try tts venv, else system python
        if reqs_vision.exists():
            run_visual_generator(str(vision_py), str(shots_json_path), str(out), shot_id=shot_id)
        elif reqs_tts.exists():
            run_visual_generator(str(tts_py), str(shots_json_path), str(out), shot_id=shot_id)
        else:
            run_visual_generator(sys.executable, str(shots_json_path), str(out), shot_id=shot_id)

    # Generate narration
    narration_out = project / 'narration.wav'
    full_text = "\n".join(s.get('narration','') for s in shots)
    tts_script = ROOT / 'scripts' / 'generate_narration.py'
    if SIMULATE_INSTALL:
        print('SIMULATE: skipping TTS generation; using placeholder narration.wav')
    else:
        run(f"{shlex.quote(str(tts_py))} {shlex.quote(str(tts_script))} --text {shlex.quote(full_text)} --output {shlex.quote(str(narration_out))} --tts {args.tts}")

    # Generate music
    music_out = project / 'music.wav'
    # Prefer the robust v2 script if present
    music_script = ROOT / 'scripts' / 'generate_music_v2.py'
    if not music_script.exists():
        music_script = ROOT / 'scripts' / 'generate_music.py'

    if SIMULATE_INSTALL:
        print('SIMULATE: skipping MusicGen generation; using placeholder music.wav')
    else:
        try:
            run(f"{shlex.quote(str(music_py))} {shlex.quote(str(music_script))} --mood mysterious --duration 45 --output {shlex.quote(str(music_out))}")
        except Exception as e:
            print('Music generation failed:', e)
            # if a placeholder exists, continue; otherwise create a 45s silent wav to allow assembly
            if music_out.exists() and music_out.stat().st_size > 0:
                print('Using existing music file:', music_out)
            else:
                print('Creating silent placeholder music.wav (45s)')
                from wave import open as wave_open
                import struct
                rate = 22050
                nframes = 45 * rate
                with wave_open(str(music_out), 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    chunk = struct.pack('<h', 0) * 1024
                    written = 0
                    while written < nframes:
                        to_write = min(1024, nframes - written)
                        wf.writeframes(chunk[:to_write*2])
                        written += to_write
                print('WROTE placeholder', music_out)

    # Assemble final montage
    run(f"{sys.executable} scripts/assemble_montage.py --project {shlex.quote(str(project))}")

    print('Pipeline completed. Final video at', project / 'final.mp4')


if __name__ == '__main__':
    main()
