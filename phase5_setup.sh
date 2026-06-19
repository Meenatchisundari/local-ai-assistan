#!/bin/bash
set -e

echo "=================================================="
echo "  Phase 5 - RunPod Setup Script"
echo "=================================================="

# ---------------------------------------------------------------
# 1. Confirm / install Ollama
# ---------------------------------------------------------------
if ! command -v ollama &> /dev/null; then
    echo "[1/6] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1/6] Ollama already installed, skipping."
fi

# Start Ollama server in background if not already running
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "      Starting Ollama server..."
    nohup ollama serve > /root/ollama.log 2>&1 &
    sleep 5
else
    echo "      Ollama server already running."
fi

# ---------------------------------------------------------------
# 2. Pull all 4 hero models (slowest step)
# ---------------------------------------------------------------
echo ""
echo "[2/6] Pulling hero models (this takes 10-20 min depending on speed)..."
for model in mistral:7b llama3:8b phi3 qwen2:7b; do
    echo "      Pulling $model ..."
    ollama pull "$model"
done

echo ""
echo "      Models installed:"
ollama list

# ---------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------
echo ""
echo "[3/6] Setting up Python environment..."
cd /root || cd ~
mkdir -p local_ai_phase5/app local_ai_phase5/data/results
cd local_ai_phase5

python3 -m venv venv
source venv/bin/activate
pip install --quiet fastapi uvicorn pydantic httpx requests psutil pandas matplotlib

# ---------------------------------------------------------------
# 4. Write FastAPI wrapper (identical to local Phase 0/1 version)
# ---------------------------------------------------------------
echo ""
echo "[4/6] Writing FastAPI wrapper..."
cat > app/__init__.py << 'EOF'
EOF

cat > app/main.py << 'PYEOF'
import json
from typing import Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

OLLAMA_BASE = "http://127.0.0.1:11434"
HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)

app = FastAPI(title="Local AI Assistant - Phase 5 GPU", version="0.1.0")

class ChatRequest(BaseModel):
    model: str = Field(..., examples=["mistral:7b"])
    prompt: str = Field(..., min_length=1)
    stream: bool = True
    options: dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3.0)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ollama_reachable": ollama_ok}

@app.get("/models")
def list_models():
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")

@app.post("/chat")
async def chat(req: ChatRequest):
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": req.options,
    }
    if req.stream:
        async def _stream():
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                async with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"data: {json.dumps({'error': body.decode()})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
PYEOF

# ---------------------------------------------------------------
# 5. Start FastAPI server
# ---------------------------------------------------------------
echo ""
echo "[5/6] Starting FastAPI server on port 8000..."
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > /root/fastapi.log 2>&1 &
sleep 4

# ---------------------------------------------------------------
# 6. Verify everything works
# ---------------------------------------------------------------
echo ""
echo "[6/6] Verifying setup..."
echo ""
echo "Ollama tags:"
curl -s http://127.0.0.1:11434/api/tags | head -c 500
echo ""
echo ""
echo "FastAPI health:"
curl -s http://127.0.0.1:8000/health
echo ""
echo ""
echo "=================================================="
echo "  Setup complete."
echo "  Working directory: /root/local_ai_phase5"
echo "  Virtual env: source venv/bin/activate (already active in this shell)"
echo "  Next: paste benchmark_harness.py and phase3_comparison.py"
echo "=================================================="