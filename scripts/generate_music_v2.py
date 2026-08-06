"""
Robust music generation wrapper.
Tries audiocraft and musicgen imports and provides actionable error messages.
Usage: python scripts/generate_music_v2.py --mood mysterious --duration 45 --output projects/first-creative-video/music.wav
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

# Detect available backend
backend = None
try:
    import importlib
    if importlib.util.find_spec('audiocraft') is not None:
        backend = 'audiocraft'
    elif importlib.util.find_spec('musicgen') is not None:
        backend = 'musicgen'
except Exception:
    backend = None

if backend is None:
    print('\nERROR: No supported music generation package found in the current environment.')
    print('Try installing one of the following inside the music venv:')
    print('  pip install audiocraft soundfile numpy scipy')
    print('  or (alternative) pip install musicgen soundfile numpy scipy')
    sys.exit(2)

if backend == 'audiocraft':
    cmd_lines = [
        'from audiocraft.models import MusicGen',
        "m = MusicGen.get_pretrained('musicgen_melody')",
        f"m.set_generation_params(duration={args.duration})",
        f"audio = m.generate('{args.mood.replace("'","\\'")}')",
        f"import soundfile as sf; sf.write('{str(out)}', audio, 32000)"
    ]
else:
    cmd_lines = [
        'from musicgen import MusicGen',
        "mg = MusicGen.get_pretrained('melody')",
        f"audio = mg.generate('{args.mood.replace("'","\\'")}', duration={args.duration})",
        f"import soundfile as sf; sf.write('{str(out)}', audio, 32000)"
    ]

cmd = [sys.executable, '-c', '\n'.join(cmd_lines)]
print('RUNNING music generation: backend=', backend)
try:
    subprocess.check_call(cmd)
    print('WROTE', out)
except subprocess.CalledProcessError as e:
    print('Music generation failed with return code', e.returncode)
    # print captured output if possible by re-running and capturing
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print('---- musicgen output ----')
        print(completed.stdout)
    except Exception:
        pass
    raise
