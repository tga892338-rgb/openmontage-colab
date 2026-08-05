import os
import json

def main():
    p = os.environ.get('OM_DEV_NOTEBOOK_PATH', 'docs/colab_stage2_real_tests.ipynb')
    if not os.path.exists(p):
        print(f"Notebook not found: {p}")
        return
    with open(p,'r',encoding='utf-8') as f:
        s=f.read()
    old='f.write(json.dumps(payload)+"\\\\n")'
    new="f.write(json.dumps(payload)+'\\\\n')"
    if old in s:
        s=s.replace(old,new)
        with open(p,'w',encoding='utf-8') as f:
            f.write(s)
        print('REPLACED v2')
    else:
        print('OLD_NOT_FOUND_v2')

if __name__ == '__main__':
    main()
