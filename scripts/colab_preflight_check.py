"""
Colab preflight check for GPU, CUDA, and recommended torch wheel tag.
Prints commands the orchestration will run and suggests a torch wheel tag.
"""
import subprocess, sys
from pathlib import Path

def run(cmd):
    print('>>>', cmd)
    subprocess.call(cmd, shell=True)

print('Environment preflight check')
# GPU / CUDA
try:
    out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, text=True)
    print('nvidia-smi output:')
    print(out)
    if 'CUDA Version: 11.8' in out or '11.8' in out:
        tag='cu118'
    elif 'CUDA Version: 12.1' in out or '12.1' in out:
        tag='cu121'
    elif 'CUDA Version: 12.8' in out or '12.8' in out:
        tag='cu128'
    elif 'T4' in out:
        tag='cu118'
    else:
        tag='cpu'
except Exception as e:
    print('nvidia-smi not found or failed:', e)
    tag='cpu'

print('\nSuggested torch wheel tag:', tag)
print('\nPlanned pip installs:')
repo_root = Path(__file__).resolve().parents[1]
reqs_tts = repo_root / 'requirements-tts.txt'
reqs_music = repo_root / 'requirements-music.txt'
print('- BASE (existing environment, not modifying)')
if reqs_tts.exists():
    print(f"- TTS (will install into venv): python -m pip install -r {reqs_tts}")
if reqs_music.exists():
    print(f"- MUSIC (will install into venv): python -m pip install -r {reqs_music}")

print('\nIf pip installs fail due to torch/transformers mismatch, consider running:')
print(f"python -m pip install --upgrade pip setuptools wheel")
if tag != 'cpu':
    print(f"# Recommended torch wheel (install inside venv): python -m pip install torch -f https://download.pytorch.org/whl/{tag}/torch_stable.html")
print('\nPreflight check complete.')
