import re

def strip_code_blocks(raw):
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        return match.group(1).strip()
    return raw

sample = '```json\n{\n    "answer": "Paris",\n    "confidence": 0.95,\n    "sources": []\n}\n```'
print("Before:")
print(sample)
print()
print("After:")
print(strip_code_blocks(sample))
