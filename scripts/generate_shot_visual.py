"""
Generate a visual asset for a single shot using the validated Qwen image path.
This script is designed to be run inside an isolated venv in Colab where the
required model packages are installed. It accepts a shot JSON file and an output path.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--shot-json', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--shot-id', required=False, help='Shot id to extract from a shots array JSON')
args = parser.parse_args()

data = json.loads(Path(args.shot_json).read_text())
# support both a single-shot dict or a list of shots
if isinstance(data, list):
    if args.shot_id:
        shot = next((s for s in data if s.get('shot_id') == args.shot_id), None)
        if shot is None:
            raise SystemExit(f"Shot id {args.shot_id} not found in {args.shot_json}")
    else:
        # fallback to first shot
        shot = data[0]
else:
    shot = data

prompt = shot.get('generation_prompt')
if not prompt:
    raise SystemExit('No generation_prompt found for shot')

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

# In Colab venv, use the qwen-image package or other validated generator.
# Here we call a placeholder command that the Colab notebook will run in the venv.
cmd = [ 'python', '-c', '\n'.join([
    'from qwen_image import QwenImage',
    "m=QwenImage(model='Qwen/Image-1.0', device='cuda')",
    "img=m.generate(\"%s\")" % prompt.replace('"', '\\"'),
    "img.save(\"%s\")" % str(out_path)
])]

try:
    subprocess.check_call(cmd)
    print('WROTE', out_path)
except Exception as e:
    print('VISUAL_GEN_FAIL', e)
    raise
