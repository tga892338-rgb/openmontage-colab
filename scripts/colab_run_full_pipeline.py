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


def run(cmd, env=None, check=True):
    print('\n>>> RUN:', cmd)
    subprocess.check_call(cmd, shell=True, env=env)


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
    """Install a compatible torch wheel into the venv python if tag is not 'cpu'."""
    if tag == 'cpu':
        print('Skipping torch wheel install (cpu tag)')
        return
    url = f"https://download.pytorch.org/whl/{tag}/torch_stable.html"
    cmd = f"{shlex.quote(str(py))} -m pip install torch -f {shlex.quote(url)}"
    print(f'Installing torch wheel for tag {tag} from {url}')
    if SIMULATE_INSTALL:
        print('SIMULATE:', cmd)
        return
    run(cmd)


def run_visual_generator(py_cmd: str, shot_json: str, out_path: str):
    cmd = f"{shlex.quote(py_cmd)} {shlex.quote(str(ROOT / 'scripts' / 'generate_shot_visual.py'))} --shot-json {shlex.quote(shot_json)} --output {shlex.quote(out_path)}"
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
    for shot in shots:
        out = assets / f"{shot.get('shot_id', 'shot')}.jpg"
        # Prefer vision venv if it has requirements, else try tts venv, else system python
        if reqs_vision.exists():
            run_visual_generator(str(vision_py), str(shots_json_path), str(out))
        elif reqs_tts.exists():
            run_visual_generator(str(tts_py), str(shots_json_path), str(out))
        else:
            run_visual_generator(sys.executable, str(shots_json_path), str(out))

    # Generate narration
    narration_out = project / 'narration.wav'
    full_text = "\n".join(s.get('narration','') for s in shots)
    tts_script = ROOT / 'scripts' / 'generate_narration.py'
    run(f"{shlex.quote(str(tts_py))} {shlex.quote(str(tts_script))} --text {shlex.quote(full_text)} --output {shlex.quote(str(narration_out))} --tts {args.tts}")

    # Generate music
    music_out = project / 'music.wav'
    music_script = ROOT / 'scripts' / 'generate_music.py'
    run(f"{shlex.quote(str(music_py))} {shlex.quote(str(music_script))} --mood mysterious --duration 45 --output {shlex.quote(str(music_out))}")

    # Assemble final montage
    run(f"{sys.executable} scripts/assemble_montage.py --project {shlex.quote(str(project))}")

    print('Pipeline completed. Final video at', project / 'final.mp4')


if __name__ == '__main__':
    main()
