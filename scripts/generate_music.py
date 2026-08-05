"""
Generate background music using MusicGen in an isolated venv.
Arguments: --mood MOOD --duration SEC --output music.wav
"""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--mood', default='mysterious')
p.add_argument('--duration', type=int, default=30)
p.add_argument('--output', required=True)
args = p.parse_args()

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

cmd = [sys.executable, '-c', '\n'.join([
    'from musicgen import MusicGen',
    "mg=MusicGen.get_pretrained('melody')",
    "audio = mg.generate(\"%s\", duration=%d)" % (args.mood.replace('"','\\"'), args.duration),
    "import soundfile as sf; sf.write(\"%s\", audio, 32000)" % str(out)
])]

print('RUNNING:', ' '.join(cmd))
subprocess.check_call(cmd)
print('WROTE', out)
