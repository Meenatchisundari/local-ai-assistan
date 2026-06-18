import httpx, json

payload = {
    "model": "tinyllama",
    "prompt": "Return this exact JSON: {\"answer\": \"Paris\", \"confidence\": 0.95, \"sources\": []}",
    "stream": False,
    "options": {"temperature": 0.0},
    "format": "json",
}

print("Sending request...")
r = httpx.post("http://127.0.0.1:8000/chat", json=payload, timeout=60.0)
print(f"Status code: {r.status_code}")
print(f"Raw response:")
print(r.text[:1000])
print()
data = r.json()
print(f"Parsed keys: {list(data.keys())}")
print(f"Response field: {repr(data.get('response', 'NOT FOUND'))[:500]}")
