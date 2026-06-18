import re

with open('phase2_structured.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = """    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\\n")
        cleaned = "\\n".join(lines[1:-1]) if len(lines) > 2 else cleaned"""

new = """    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\\s*([\\s\\S]*?)```", cleaned)
    if match:
        cleaned = match.group(1).strip()"""

if old in content:
    content = content.replace(old, new)
    with open('phase2_structured.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Patched successfully')
else:
    print('Pattern not found - showing current validate_response:')
    start = content.find('def validate_response')
    print(content[start:start+500])
