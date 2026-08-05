import os,sys,json,traceback
from pathlib import Path

# Move side-effects into main() and make paths configurable via env vars.

def main():
    sys.path.insert(0, os.environ.get('OM_REPO_ROOT', '.'))
    from src.adapters import adapter_loader
    from src.registry import load_registry
    reg = load_registry().data
    roles = ['music','tts','transcriber','enhancement']
    role_deps = {
        'music': ['transformers','stable_audio'],
        'tts': ['cosyvoice','pyttsx3','scipy','numpy'],
        'transcriber': ['faster_whisper','whisper'],
        'enhancement': ['realesrgan','rife']
    }
    art_dir = Path(os.environ.get('OM_SMOKE_ART', 'projects/smoke-report/artifacts'))
    art_dir.mkdir(parents=True, exist_ok=True)
    for role in roles:
        payload = {'event':'optional_adapter_diagnostic','role':role}
        choice = reg.get(role,{}).get('default_choice')
        payload['choice'] = choice
        mapping = adapter_loader.MODEL_TO_ADAPTER.get(choice)
        payload['mapping'] = mapping
        module_name,class_name = mapping if mapping else (None,None)
        full_module = f"src.adapters.{module_name}" if module_name else None
        payload['module'] = full_module
        # dependency checks
        deps = {}
        for d in role_deps.get(role,[]):
            try:
                __import__(d)
                deps[d] = {'imported': True}
            except Exception as e:
                deps[d] = {'imported': False, 'error': str(e)}
        payload['dependency_checks'] = deps
        # import module
        try:
            if full_module:
                mod = __import__(full_module, fromlist=['*'])
            payload['import_success'] = True
        except Exception:
            payload['import_success'] = False
            payload['import_traceback'] = traceback.format_exc()
            with open(art_dir / 'real_init.log','a',encoding='utf-8') as f:
                f.write(json.dumps(payload)+"\n")
            continue
        # attempt lightweight instantiation for roles that are safe
        constructor_ok = False
        constructor_tb = None
        available_flag = None
        try:
            cls = getattr(mod, class_name)
            adapters_conf = reg.get('adapters', {})
            model_conf = adapters_conf.get(choice, {})
            # For transcriber, avoid heavy model load: override model to 'small' if present
            if role == 'transcriber':
                model_conf = dict(model='small')
            try:
                inst = cls(choice, config=model_conf or {})
                constructor_ok = True
                available_flag = getattr(inst,'available',None)
            except Exception:
                constructor_tb = traceback.format_exc()
        except Exception:
            constructor_tb = traceback.format_exc()
        payload['constructor_ok'] = constructor_ok
        payload['constructor_traceback'] = constructor_tb
        payload['available_flag'] = available_flag
        # device info
        try:
            import torch
            payload['cuda_available'] = torch.cuda.is_available()
            if payload['cuda_available']:
                payload['device_name'] = torch.cuda.get_device_name(0)
                payload['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2)
        except Exception:
            payload['cuda_available'] = False
        with open(art_dir / 'real_init.log','a',encoding='utf-8') as f:
            f.write(json.dumps(payload)+"\n")
        print('WROTE',role)
    print('DONE')

if __name__ == '__main__':
    main()
