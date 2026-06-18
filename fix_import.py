with open('phase2_structured.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'import re' not in content:
    content = 'import re\n' + content
    with open('phase2_structured.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed - added import re')
else:
    print('import re already present')
