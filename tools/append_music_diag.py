import os,sys,json,traceback
sys.path.insert(0, r'C:\Users\hp\PycharmProjects\jvvghj123-sudo_openmontage-colab')
from src.adapters import adapter_loader
from src.registry import load_registry
reg = load_registry().data
choice = reg.get('music', {}).get('default_choice')
mapping = adapter_loader.MODEL_TO_ADAPTER.get(choice)
module_name, class_name = mapping if mapping else (None,None)
full_module = f"src.adapters.{module_name}" if module_name else None
# dependency checks
deps = {}
for d in ['transformers','stable_audio']:
    try:
        __import__(d)
        deps[d] = {'imported': True}
    except Exception as e:
        deps[d] = {'imported': False, 'error': str(e)}
# try import module
import_success = False
import_tb = None
try:
    mod = __import__(full_module, fromlist=['*'])
    import_success = True
except Exception as e:
    import_tb = traceback.format_exc()
# try constructor
constructor_ok = False
constructor_tb = None
available_flag = None
try:
    if import_success:
        cls = getattr(mod, class_name)
        adapters_conf = reg.get('adapters', {})
        model_conf = adapters_conf.get(choice, {})
        try:
            inst = cls(choice, config=model_conf or {})
            constructor_ok = True
            available_flag = getattr(inst, 'available', None)
        except Exception as e:
            constructor_tb = traceback.format_exc()
except Exception as e:
    constructor_tb = traceback.format_exc()
# device info
device_info = {}
try:
    import torch
    device_info['cuda_available'] = torch.cuda.is_available()
    if device_info['cuda_available']:
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2)
except Exception:
    device_info['cuda_available'] = False
payload = {
    'event': 'optional_adapter_diagnostic',
    'role': 'music',
    'choice': choice,
    'mapping': mapping,
    'module': full_module,
    'import_success': import_success,
    'import_traceback': import_tb,
    'dependency_checks': deps,
    'constructor_ok': constructor_ok,
    'constructor_traceback': constructor_tb,
    'available_flag': available_flag,
}
payload.update(device_info)
art = os.path.join('projects','smoke-report','artifacts')
os.makedirs(art, exist_ok=True)
with open(os.path.join(art,'real_init.log'),'a',encoding='utf-8') as f:
    f.write(json.dumps(payload)+"\n")
print('WROTE_DIAG')
