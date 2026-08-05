"""Diagnostic for image adapter instantiation.
This script gathers detailed diagnostics about why the image adapter may fail to load on Colab.
It stubs model download (StableDiffusionPipeline.from_pretrained) only for this diagnostic instantiation to avoid large downloads.
It MUST NOT be used to assert real model readiness; the script reports whether constructor succeeded under the stub.
"""
import os
import sys
import traceback
import json
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adapters import adapter_loader
from src.registry import load_registry
from src.selector import ModelSelector

OUT = Path('projects') / 'smoke-report' / 'artifacts'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'real_init.log'

report = {}
try:
    os.environ.setdefault('OM_LOAD_ADAPTERS','1')
    os.environ.setdefault('OM_REAL_STRICT','1')
    reg = load_registry().data
    sel = ModelSelector(reg)
    pick = sel.pick('image_generation', profile='balanced')
    choice = pick.get('choice')
    report['selected_choice'] = choice
    report['selector_source'] = pick.get('source')
    entry = adapter_loader.MODEL_TO_ADAPTER.get(choice)
    report['mapping'] = entry

    # log device info
    try:
        import torch
        report['torch_version'] = getattr(torch, '__version__', None)
        report['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            try:
                report['device_name'] = torch.cuda.get_device_name(0)
                report['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2)
            except Exception as e:
                report['device_info_error'] = str(e)
    except Exception as e:
        report['torch_import_error'] = str(e)

    # Try import adapter module
    adapter_mod = None
    adapter_import_exc = None
    if entry:
        module_name = entry[0]
        full_mod = f"src.adapters.{module_name}"
        try:
            adapter_mod = import_module(full_mod)
            report['adapter_module_import'] = f"OK: {full_mod}"
        except Exception as e:
            adapter_import_exc = traceback.format_exc()
            report['adapter_module_import'] = f"FAILED: {full_mod}"
            report['adapter_module_import_traceback'] = adapter_import_exc

    # Dependency imports
    deps = ['diffusers','transformers','PIL','torch']
    dep_results = {}
    for d in deps:
        try:
            m = import_module(d if d!='PIL' else 'PIL')
            dep_results[d] = {'imported': True}
        except Exception as e:
            dep_results[d] = {'imported': False, 'error': str(e)}
    report['dependency_checks'] = dep_results

    # Attempt constructor with model-download stub for diffusers
    constructor_result = None
    constructor_trace = None
    stub_used = False
    if adapter_mod is not None:
        try:
            cls_name = entry[1]
            AdapterCls = getattr(adapter_mod, cls_name)

            # Monkeypatch StableDiffusionPipeline.from_pretrained if available
            try:
                import diffusers
                from diffusers import StableDiffusionPipeline
                orig_from_pretrained = StableDiffusionPipeline.from_pretrained
                stub_used = True

                class DummyPipe:
                    def __init__(self, *args, **kwargs):
                        pass
                    def to(self, device):
                        return self
                    def __call__(self, *args, **kwargs):
                        class Out:
                            def __init__(self):
                                from PIL import Image
                                self.images = [Image.new('RGB', (512,512), color=(123,123,123))]
                        return Out()

                StableDiffusionPipeline.from_pretrained = lambda *a, **k: DummyPipe()
            except Exception as e:
                # diffusers not importable or patch failed
                report['diffusers_patch_error'] = str(e)

            try:
                # instantiate adapter (this will use the stub if patch succeeded)
                inst = AdapterCls(choice, config=(reg.get('adapters',{}).get(choice, {})))
                constructor_result = 'OK' if getattr(inst, 'available', True) else 'AVAILABLE_FALSE'
            except Exception as e:
                constructor_result = 'FAILED'
                constructor_trace = traceback.format_exc()

            # restore stub if applied
            try:
                if stub_used:
                    StableDiffusionPipeline.from_pretrained = orig_from_pretrained
            except Exception:
                pass

        except Exception as e:
            constructor_result = 'FAILED_TO_LOCATE_CLASS'
            constructor_trace = traceback.format_exc()

    report['constructor_result_with_stub'] = constructor_result
    if constructor_trace:
        report['constructor_traceback'] = constructor_trace
    report['stub_used_for_diagnostic'] = stub_used

except Exception as e:
    report['diagnostic_exception'] = traceback.format_exc()

# Append to real_init.log with detailed formatting
with open(LOG, 'a', encoding='utf-8') as lf:
    lf.write('\n=== Image Adapter Diagnostic ===\n')
    lf.write('REPORT JSON:\n')
    lf.write(json.dumps(report, indent=2))
    lf.write('\n')

print(json.dumps(report, indent=2))
