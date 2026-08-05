"""
Orchestration script intended to run in Colab (or locally with GPU) to execute
the vertical slice pipeline. It calls tools/director_first_creative.py to produce
shot_plan.json and outputs scaffolding. Heavy model operations are executed in
isolated venvs by the Colab notebook, not in this script when run from the
notebook kernel to avoid contaminating the kernel's torch.

This script is safe to call from the notebook subprocess approach.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / 'projects' / 'first-creative-video'
OUT.mkdir(parents=True, exist_ok=True)

def run_director(idea: str = None):
    cmd = [sys.executable, 'tools/director_first_creative.py']
    if idea:
        cmd += ['--idea', idea]
    print('Running director:', ' '.join(cmd))
    subprocess.check_call(cmd)
    print('Director finished; shot_plan at', OUT / 'shot_plan.json')

if __name__ == '__main__':
    # default - generate plan
    idea = None
    if len(sys.argv) > 1:
        idea = sys.argv[1]
    run_director(idea)
    print('\nNext: use the Colab notebook notebooks/colab_first_creative_video.ipynb to run the pipeline.\n')
