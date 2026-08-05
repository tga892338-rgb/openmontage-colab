# Colab Stage 2 diagnostic cells
# Paste each section as separate Colab cells. These are lightweight checks only:
# - CUDA check
# - Adapter import + instantiate (no heavy downloads)
# - One-shot safe test when feasible (transcriber small)

# --- Cell 1: Setup and repo path ---
cell_1 = r"""
# Mount /repo and add to PYTHONPATH if running from notebook connected to repo
import os, sys
# If running inside the repo, adjust path as needed
if '/content/openmontage-colab' not in sys.path:
    sys.path.insert(0, '/content/openmontage-colab')

# Ensure strict real mode and allow adapter imports
os.environ['OM_LOAD_ADAPTERS'] = '1'
os.environ['OM_REAL_STRICT'] = '1'
print('OM_LOAD_ADAPTERS, OM_REAL_STRICT set')
"""

# --- Cell 2: CUDA check ---
cell_2 = r"""
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print('device name:', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print('vram GB:', round(props.total_memory/(1024**3),2))
    except Exception as e:
        print('cuda device query failed:', e)
"""

# --- Cell 3: Optional adapters quick import/instantiate ---
cell_3 = r"""
import json, traceback
from src.adapters.adapter_loader import MODEL_TO_ADAPTER
from src.registry import load_registry
reg = load_registry().data
roles = ['music','tts','transcriber','enhancement']
summary = {}
for role in roles:
    try:
        choice = reg.get(role, {}).get('default_choice')
        mapping = MODEL_TO_ADAPTER.get(choice)
        module_name, class_name = mapping if mapping else (None,None)
        full_module = f'src.adapters.{module_name}' if module_name else None
        entry = {'choice': choice, 'mapping': mapping, 'module': full_module}
        if full_module is None:
            entry['import_success'] = False
            entry['error'] = 'no mapping'
            summary[role] = entry
            continue
        try:
            mod = __import__(full_module, fromlist=['*'])
            entry['import_success'] = True
        except Exception:
            entry['import_success'] = False
            entry['import_traceback'] = traceback.format_exc()
            summary[role] = entry
            continue
        # instantiate lightweight where possible
        try:
            cls = getattr(mod, class_name)
            adapters_conf = reg.get('adapters', {})
            model_conf = adapters_conf.get(choice, {})
            # For transcriber, force small model to avoid large downloads
            if role == 'transcriber':
                model_conf = dict(model='small')
            inst = cls(choice, config=model_conf or {})
            entry['constructor_ok'] = True
            entry['available_flag'] = getattr(inst, 'available', None)
            entry['class'] = inst.__class__.__name__
        except Exception:
            entry['constructor_ok'] = False
            entry['constructor_traceback'] = traceback.format_exc()
        summary[role] = entry
    except Exception:
        summary[role] = {'error': traceback.format_exc()}
print(json.dumps(summary, indent=2))
"""

# --- Cell 4: One-shot transcriber test (small) ---
cell_4 = r"""
# This will synthesize a tiny WAV and run the transcriber (small model).
# May download small model weights; abort if you prefer not to.
import os, sys, wave
from pathlib import Path
p = Path('projects/colab-stage2-test')
p.mkdir(parents=True, exist_ok=True)
wav_path = p/'silence_short.wav'
# create 1s of silence at 16kHz
try:
    import numpy as np
    import scipy.io.wavfile as wavfile
    sr = 16000
    silence = np.zeros(sr, dtype=np.int16)
    wavfile.write(str(wav_path), sr, silence)
    print('WAV created:', wav_path)
except Exception as e:
    print('Failed creating WAV:', e)

# Run transcriber
from src.adapters.adapter_loader import get_component
comp = get_component('transcriber', profile='balanced')
if comp is None:
    print('Transcriber component not available')
else:
    print('Transcriber class:', comp.__class__.__name__)
    try:
        res = comp.run(str(wav_path))
        print('Transcription result:', res.get('transcript')[:200])
    except Exception as e:
        print('Transcriber run failed:', repr(e))
"""

# --- Cell 5: Safe enhancement passthrough test ---
cell_5 = r"""
# Check enhancement adapter instantiation and run a passthrough if realesrgan missing
from src.adapters.adapter_loader import get_component
inst = get_component('enhancement', profile='balanced')
print('Enhancement inst:', inst.__class__.__name__ if inst else None)
if inst is None:
    print('Enhancement adapter not available')
else:
    try:
        # create tiny test image
        from PIL import Image
        p = 'projects/colab-stage2-test/small.png'
        Image.new('RGB', (64,64), color=(128,128,128)).save(p)
        out = inst.run(p)
        print('Enhancement run result:', out)
    except Exception as e:
        print('Enhancement run failed:', repr(e))
"""

# --- Cell 6: Music & TTS lightweight check ---
cell_6 = r"""
# Music: instantiate adapter and report available flag; do not generate audio
from src.adapters.adapter_loader import get_component
m = get_component('music', profile='balanced')
print('Music adapter:', m.__class__.__name__ if m else None, 'available:', getattr(m,'available',None) if m else None)

# TTS: try to synthesize a short fallback audio using adapter
tts = get_component('tts', profile='balanced')
print('TTS adapter:', tts.__class__.__name__ if tts else None)
if tts:
    try:
        out = tts.run('Test TTS for Stage 2 check', output='projects/colab-stage2-test/tts.wav')
        print('TTS run result:', out)
    except Exception as e:
        print('TTS run failed:', repr(e))
"""

# --- Cell 7: Summary ---
cell_7 = r"""
print('Stage 2 quick checks completed. Inspect output files in projects/colab-stage2-test and projects/smoke-report/artifacts/real_init.log for diagnostics.')
"""

# Print helper instructions to guide paste into Colab
if __name__ == '__main__':
    print('# Copy each triple-quoted cell (cell_1..cell_7) into separate Colab cells in order.')
    for i, name in enumerate(['cell_1','cell_2','cell_3','cell_4','cell_5','cell_6','cell_7'], start=1):
        print(f'--- Cell {i}: {name} ---')
        print(globals()[name])
        print('\n')
