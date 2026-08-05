import os, sys, traceback, json
sys.path.insert(0, r'C:\Users\hp\PycharmProjects\jvvghj123-sudo_openmontage-colab')
os.environ['OM_LOAD_ADAPTERS'] = '1'
os.environ['OM_REAL_STRICT'] = '1'
from src.adapters.adapter_loader import get_component
roles = ['planner','image_generation','montage']
results = {}
for r in roles:
    try:
        comp = get_component(r, profile='balanced')
        results[r] = {'loaded': comp is not None, 'class': comp.__class__.__name__ if comp else None}
    except Exception as e:
        results[r] = {'loaded': False, 'error': str(e), 'traceback': traceback.format_exc()}
print(json.dumps(results, indent=2))
