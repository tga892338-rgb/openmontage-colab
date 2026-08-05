p='C:/Users/hp/PycharmProjects/jvvghj123-sudo_openmontage-colab/docs/colab_stage2_real_tests.ipynb'
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
