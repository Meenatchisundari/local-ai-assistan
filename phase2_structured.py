import re
"""
phase2_structured.py  -  Phase 2: Structured Output & Reliability
=================================================================
Defines the Pydantic schema, prompt engineering, validation,
retry logic, and temperature comparison pipeline.

Run:
  python phase2_structured.py --quick
  python phase2_structured.py --models tinyllama qwen2:1.5b
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Step 1 - Pydantic Schema
# ---------------------------------------------------------------------------

class StructuredResponse(BaseModel):
    """
    The exact shape every model response must match.
    Pydantic checks every field automatically:
      - answer     : must be a non-empty string
      - confidence : must be a float between 0.0 and 1.0
      - sources    : must be a list of strings (can be empty list)
    """
    answer: str = Field(..., min_length=1, description="The model's answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    sources: list[str] = Field(default_factory=list, description="List of sources")


# ---------------------------------------------------------------------------
# Step 2 - Prompt Engineering
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are a precise assistant that always responds in valid JSON.
Your response must be a single JSON object with exactly these three fields:
- "answer": your response as a string
- "confidence": your confidence as a float between 0.0 and 1.0
- "sources": a list of strings describing your sources

Do not include any text outside the JSON object.
Do not use markdown code blocks.
Return only the raw JSON object."""

RETRY_INSTRUCTION = """Your previous response was not valid JSON or was missing required fields.
Required format:
{{"answer": "your answer here", "confidence": 0.95, "sources": ["source1"]}}

Previous response: {previous_response}

Try again. Return ONLY the JSON object, nothing else."""

def build_prompt(question: str) -> str:
    """Combine system instruction with the actual question."""
    return f"{SYSTEM_INSTRUCTION}\n\nQuestion: {question}"

def build_retry_prompt(question: str, previous_response: str) -> str:
    """Build a correction prompt that includes what went wrong."""
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question: {question}\n\n"
        + RETRY_INSTRUCTION.format(previous_response=previous_response[:200])
    )


# ---------------------------------------------------------------------------
# Step 3 - Single attempt: call model + validate
# ---------------------------------------------------------------------------

FASTAPI_BASE = "http://127.0.0.1:8000"
OLLAMA_BASE  = "http://127.0.0.1:11434"

def call_model(model: str, prompt: str, temperature: float) -> str:
    """
    Send prompt to FastAPI, collect full response as a string.
    Uses stream=False here - we need the complete text before validating.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
        "format": "json",
    }
    try:
        r = httpx.post(f"{FASTAPI_BASE}/chat", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        # Ollama returns {"response": "...", "done": true, ...}
        return data.get("response", "").strip()
    except Exception as exc:
        return f"REQUEST_ERROR: {exc}"


def validate_response(raw: str) -> tuple[StructuredResponse | None, str | None]:
    """
    Try to parse and validate the raw model output.
    Returns (StructuredResponse, None) on success.
    Returns (None, error_message) on failure.
    """
    if raw.startswith("REQUEST_ERROR"):
        return None, raw

    # Sometimes models wrap JSON in markdown code blocks - strip them
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        parsed = StructuredResponse.model_validate_json(cleaned)
        return parsed, None
    except ValidationError as exc:
        # Extract the first error message for logging
        errors = exc.errors()
        error_msg = f"ValidationError: {errors[0]['loc']} - {errors[0]['msg']}"
        return None, error_msg
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    except Exception as exc:
        return None, f"UnexpectedError: {exc}"


# ---------------------------------------------------------------------------
# Step 4 - Full pipeline with retry
# ---------------------------------------------------------------------------

def run_with_retry(
    model: str,
    prompt_text: str,
    question: str,
    temperature: float,
) -> dict:
    """
    Attempt 1: call model, validate.
    If fail: attempt 2 with correction prompt.
    Returns a result dict with status, parsed fields, errors.
    """
    result = {
        "status": None,           # PASS / RETRY_PASS / FAIL
        "attempts": 0,
        "attempt1_raw": None,
        "attempt1_error": None,
        "attempt2_raw": None,
        "attempt2_error": None,
        "answer": None,
        "confidence": None,
        "sources": None,
        "parse_error": None,
    }

    # --- Attempt 1 ---
    result["attempts"] = 1
    raw1 = call_model(model, prompt_text, temperature)
    result["attempt1_raw"] = raw1[:300]  # truncate for CSV
    parsed, error = validate_response(raw1)

    if parsed:
        result["status"] = "PASS"
        result["answer"] = parsed.answer
        result["confidence"] = parsed.confidence
        result["sources"] = json.dumps(parsed.sources)
        return result

    result["attempt1_error"] = error

    # --- Attempt 2 (retry with correction) ---
    result["attempts"] = 2
    retry_prompt = build_retry_prompt(question, raw1)
    raw2 = call_model(model, retry_prompt, temperature)
    result["attempt2_raw"] = raw2[:300]
    parsed2, error2 = validate_response(raw2)

    if parsed2:
        result["status"] = "RETRY_PASS"
        result["answer"] = parsed2.answer
        result["confidence"] = parsed2.confidence
        result["sources"] = json.dumps(parsed2.sources)
        result["parse_error"] = error  # log what failed first time
        return result

    # --- Both failed ---
    result["status"] = "FAIL"
    result["attempt2_error"] = error2
    result["parse_error"] = f"Attempt1: {error} | Attempt2: {error2}"
    return result


# ---------------------------------------------------------------------------
# Prompt suite for Phase 2
# ---------------------------------------------------------------------------

PROMPTS = [
    {"id": "fqa_01", "category": "factual_qa",
     "question": "What is the capital of France?"},
    {"id": "fqa_02", "category": "factual_qa",
     "question": "What does RAM stand for in computing?"},
    {"id": "fqa_03", "category": "factual_qa",
     "question": "What is the chemical symbol for water?"},
    {"id": "fqa_04", "category": "factual_qa",
     "question": "In what year was Python programming language first released?"},
    {"id": "fqa_05", "category": "factual_qa",
     "question": "What is the speed of light in kilometres per second?"},
    {"id": "sum_01", "category": "summarisation",
     "question": "Summarise what a CPU does in one sentence."},
    {"id": "sum_02", "category": "summarisation",
     "question": "Summarise what machine learning is in one sentence."},
    {"id": "rsn_01", "category": "reasoning",
     "question": "If a car travels at 60 km/h for 2 hours, how far does it travel?"},
    {"id": "rsn_02", "category": "reasoning",
     "question": "What is 15 percent of 200?"},
    {"id": "inst_01", "category": "instruction_following",
     "question": "Name the first three planets in our solar system in order."},
]

CSV_FIELDNAMES = [
    "run_id", "session_id", "timestamp_utc", "model", "prompt_id",
    "prompt_category", "temperature", "status", "attempts",
    "answer", "confidence", "sources", "parse_error",
    "attempt1_error", "attempt2_error",
]


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_phase2(models, prompts, temperatures, out_dir, quick=False):
    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"phase2_{session_id}.csv"
    suite = prompts[:5] if quick else prompts

    total = len(models) * len(suite) * len(temperatures)
    counter = 0

    print(f"\n{'='*55}")
    print(f"  Phase 2 Structured Output  -  session {session_id}")
    print(f"{'='*55}")
    print(f"  Models       : {', '.join(models)}")
    print(f"  Prompts      : {len(suite)} ({'quick' if quick else 'full'})")
    print(f"  Temperatures : {temperatures}")
    print(f"  Total runs   : {total}")
    print(f"  Output       : {csv_path}")
    print(f"{'='*55}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for model in models:
            for temp in temperatures:
                print(f"\n--- {model}  temp={temp} ---")
                for prompt in suite:
                    counter += 1
                    run_id = f"{session_id}__{model.replace(':','_')}__{prompt['id']}__t{temp}"
                    ts = datetime.utcnow().isoformat(timespec="milliseconds")

                    print(f"  [{counter:>3}/{total}] {prompt['id']}  ", end="", flush=True)

                    full_prompt = build_prompt(prompt["question"])
                    result = run_with_retry(model, full_prompt, prompt["question"], temp)

                    status_display = {
                        "PASS": "PASS      ",
                        "RETRY_PASS": "RETRY_PASS",
                        "FAIL": "FAIL      ",
                    }.get(result["status"], result["status"])

                    print(f"{status_display}  attempts={result['attempts']}  "
                          f"confidence={result['confidence']}")

                    row = {
                        "run_id": run_id,
                        "session_id": session_id,
                        "timestamp_utc": ts,
                        "model": model,
                        "prompt_id": prompt["id"],
                        "prompt_category": prompt["category"],
                        "temperature": temp,
                        "status": result["status"],
                        "attempts": result["attempts"],
                        "answer": result["answer"],
                        "confidence": result["confidence"],
                        "sources": result["sources"],
                        "parse_error": result["parse_error"],
                        "attempt1_error": result["attempt1_error"],
                        "attempt2_error": result["attempt2_error"],
                    }
                    writer.writerow(row)
                    fh.flush()

    print(f"\n  Done. {counter} runs written to:\n  {csv_path}\n")
    return csv_path


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(csv_path):
    try:
        import pandas as pd
    except ImportError:
        print("pip install pandas")
        return

    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print("  Phase 2 Results")
    print(f"{'='*55}")

    for temp in sorted(df["temperature"].unique()):
        print(f"\n  Temperature {temp}")
        subset = df[df["temperature"] == temp]
        total = len(subset)
        summary = subset.groupby("model")["status"].value_counts().unstack(fill_value=0)

        for model in summary.index:
            row = summary.loc[model]
            passes     = row.get("PASS", 0)
            retries    = row.get("RETRY_PASS", 0)
            fails      = row.get("FAIL", 0)
            model_total = passes + retries + fails
            print(f"\n    {model} ({model_total} runs)")
            print(f"      First-try pass  : {passes:>3}  ({100*passes/model_total:.0f}%)")
            print(f"      Pass after retry: {retries:>3}  ({100*retries/model_total:.0f}%)")
            print(f"      Final fail      : {fails:>3}  ({100*fails/model_total:.0f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["tinyllama", "qwen2:1.5b"])
p.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
p.add_argument("--out-dir", type=Path, default=Path("data/results"))
p.add_argument("--quick", action="store_true", help="Run only 5 prompts")
p.add_argument("--no-summary", action="store_true")
args = p.parse_args()

csv_path = run_phase2(args.models, PROMPTS, args.temperatures, args.out_dir, args.quick)
if not args.no_summary:
    print_summary(csv_path)
