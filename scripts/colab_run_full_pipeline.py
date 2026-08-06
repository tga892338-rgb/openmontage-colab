"""
Orchestrate full pipeline inside a Colab-like environment using isolated venvs.
This script is intended to be invoked inside a Colab notebook cell (or any Linux/GPU host).
It creates venvs for TTS and Music, installs requirements, runs generation scripts per the
scene_plan/shot_plan, and assembles the final montage.

Warning: Running this will install packages and may download model weights. Use in
an isolated Colab runtime only.
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
    print('RUN:', cmd)
    subprocess.check_call(cmd, shell=True, env=env)


def make_venv(venv_path: Path):
    run(f"python -m venv {shlex.quote(str(venv_path))}")
    py = venv_path / 'bin' / 'python' if os.name != 'nt' else venv_path / 'Scripts' / 'python.exe'
    return py


def pip_install(py: Path, reqs: str):
    run(f"{shlex.quote(str(py))} -m pip install --upgrade pip setuptools wheel")
    run(f"{shlex.quote(str(py))} -m pip install -r {shlex.quote(reqs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='projects/first-creative-video')
    parser.add_argument('--tts', choices=['qwen', 'chatterbox'], default='qwen')
    parser.add_argument('--shots-json', default='projects/first-creative-video/shot_plan.json')
    args = parser.parse_args()

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    # Load shots
    shots = json.loads(Path(args.shots_json).read_text(encoding='utf-8'))

    venvs = project / 'venvs'
    venvs.mkdir(parents=True, exist_ok=True)

    tts_venv = venvs / f'tts_{args.tts}'
    music_venv = venvs / 'musicgen'

    print('Creating TTS venv at', tts_venv)
    tts_py = make_venv(tts_venv)
    # Install tts requirements (requirements-tts.txt should exist at repo root)
    reqs_tts = ROOT / 'requirements-tts.txt'
    if reqs_tts.exists():
        pip_install(tts_py, str(reqs_tts))
    else:
        print('WARNING: requirements-tts.txt not found; skipping TTS install')

    print('Creating Music venv at', music_venv)
    music_py = make_venv(music_venv)
    reqs_music = ROOT / 'requirements-music.txt'
    if reqs_music.exists():
        pip_install(music_py, str(reqs_music))
    else:
        print('WARNING: requirements-music.txt not found; skipping Music install')

    # Generate visuals per shot using the validated qwen image path helper script
    assets = project / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        out = assets / f"{shot['shot_id']}.jpg"
        cmd = f"{sys.executable} scripts/generate_shot_visual.py --shot-json {shlex.quote(args.shots_json)} --output {shlex.quote(str(out))}"
        # In Colab this should be run in a venv where qwen-image is installed; here use system python
        run(cmd)

    # Generate narration using selected TTS venv
    narration_out = project / 'narration.wav'
    full_text = "\n".join(s['narration'] for s in shots)
    tts_cmd = f"{shlex.quote(str(tts_py))} {shlex.quote(str(ROOT / 'scripts' / 'generate_narration.py'))} --text {shlex.quote(full_text)} --output {shlex.quote(str(narration_out))} --tts {args.tts}"
    run(tts_cmd)

    # Generate music using music venv
    music_out = project / 'music.wav'
    music_cmd = f"{shlex.quote(str(music_py))} {shlex.quote(str(ROOT / 'scripts' / 'generate_music.py'))} --mood mysterious --duration 45 --output {shlex.quote(str(music_out))}"
    run(music_cmd)

    # Assemble final montage
    run(f"{sys.executable} scripts/assemble_montage.py --project {shlex.quote(str(project))}")

    print('Pipeline completed. Final video at', project / 'final.mp4')


if __name__ == '__main__':
    main()
