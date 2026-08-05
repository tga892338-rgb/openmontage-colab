"""
Generate full narration WAV using Qwen3-TTS or Chatterbox in an isolated venv.
Arguments: --text TEXT --output narration.wav --tts qwen|chatterbox
"""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--text', required=True)
p.add_argument('--output', required=True)
p.add_argument('--tts', choices=['qwen','chatterbox'], default='qwen')
args = p.parse_args()

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

if args.tts == 'qwen':
    # assume a venv with qwen3tts installed and a small runner script
    cmd = [sys.executable, '-c', '\n'.join([
        'from qwen_tts import Qwen3TTS',
        "tts=Qwen3TTS('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', device='cuda')",
        "audio=tts.synthesize(text=\"%s\", instruction='Calm cinematic documentary narration', sampling_rate=24000)" % args.text.replace('\"','\\\"'),
        "import soundfile as sf; sf.write(\"%s\", audio, 24000)" % str(out)
    ])]
else:
    cmd = [sys.executable, '-c', '\n'.join([
        'from chatterbox import ChatterboxTurbo',
        "tts=ChatterboxTurbo(model_id='ResembleAI/chatterbox-turbo', device='cuda')",
        "audio, sr = tts.synthesize(text=\"%s\")" % args.text.replace('\"','\\\"'),
        "import soundfile as sf; sf.write(\"%s\", audio, sr)" % str(out)
    ])]

print('RUNNING:', ' '.join(cmd))
subprocess.check_call(cmd)
print('WROTE', out)
