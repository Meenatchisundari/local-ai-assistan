"""
phase4_quantization.py  -  Phase 4: Quantization Tradeoff Analysis (dry run)
===============================================================================
Compares the SAME base model (gemma2:2b-instruct) at three quantization levels:
  q4_K_M (most compressed) -> q5_K_M (middle) -> q8_0 (least compressed)

Reuses Phase 3's timing + RAM tracking logic, applied to one model family.

Run:
  python phase4_quantization.py
  python phase4_quantization.py --quick
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil

FASTAPI_BASE = "http://127.0.0.1:8000"
OLLAMA_BASE  = "http://127.0.0.1:11434"

# Same base model, three quantization levels - this IS the experiment
QUANT_MODELS = [
    "gemma2:2b-instruct-q4_K_M",
    "gemma2:2b-instruct-q5_K_M",
    "gemma2:2b-instruct-q8_0",
]

PROMPTS = [
    {"id": "fqa_01", "category": "factual_qa",
     "text": "What is the capital of Italy? One sentence."},
    {"id": "fqa_02", "category": "factual_qa",
     "text": "What does GPU stand for? One sentence."},
    {"id": "sum_01", "category": "summarisation",
     "text": "Summarise in one sentence: Quantization reduces model precision to save memory and increase speed, at some cost to output quality."},
    {"id": "inst_01", "category": "instruction_following",
     "text": "List exactly three colours. One per line. No other text."},
    {"id": "inst_02", "category": "instruction_following",
     "text": "Answer only with a number: how many continents are there?"},
    {"id": "rsn_01", "category": "reasoning",
     "text": "If a book costs 12 dollars and I buy 4, how much do I spend in total? Show your working."},
    {"id": "rsn_02", "category": "reasoning",
     "text": "What is 25 percent of 80? Show your working briefly."},
    {"id": "json_01", "category": "json_generation",
     "text": "Return ONLY a valid JSON object with keys: country, capital for Italy. No markdown."},
    {"id": "edge_01", "category": "edge_case",
     "text": "Respond with exactly one word: the opposite of fast."},
    {"id": "lng_01", "category": "longer_generation",
     "text": "Write 3 to 4 sentences explaining what quantization means for AI models running on phones."},
]

CSV_FIELDNAMES = [
    "run_id", "session_id", "timestamp_utc", "model", "quant_level",
    "prompt_id", "prompt_category", "prompt_text",
    "ttft_s", "tokens_per_sec", "total_latency_s", "output_tokens",
    "ram_before_mb", "ram_after_mb", "ram_delta_mb", "ram_peak_mb",
    "model_disk_size_gb",
    "response_text",
    "score_correctness", "score_coherence", "score_instruction", "score_overall",
    "error",
]

MODEL_DISK_SIZES_GB = {
    "gemma2:2b-instruct-q4_K_M": 1.7,
    "gemma2:2b-instruct-q5_K_M": 1.9,
    "gemma2:2b-instruct-q8_0": 2.8,
}

def extract_quant_level(model_name):
    """gemma2:2b-instruct-q4_K_M -> q4_K_M"""
    return model_name.split("-")[-1]

def get_system_ram_mb():
    return round(psutil.virtual_memory().used / 1024 / 1024, 1)

def measure_with_ram(model, prompt_text, temperature=0.0):
    result = {
        "ttft_s": None, "tokens_per_sec": None, "total_latency_s": None,
        "output_tokens": None, "ram_before_mb": None, "ram_after_mb": None,
        "ram_delta_mb": None, "ram_peak_mb": None,
        "response_text": None, "error": None,
    }
    payload = {
        "model": model, "prompt": prompt_text, "stream": True,
        "options": {"temperature": temperature},
    }
    ram_before = get_system_ram_mb()
    result["ram_before_mb"] = ram_before
    t_start = time.perf_counter()
    t_first = None
    tokens = []
    out_count = None
    peak_ram = ram_before

    try:
        with httpx.stream("POST", f"{FASTAPI_BASE}/chat",
                          json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                line = raw.removeprefix("data: ").strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                tok = chunk.get("response", "")
                if tok and t_first is None:
                    t_first = time.perf_counter()
                if tok:
                    tokens.append(tok)
                current_ram = get_system_ram_mb()
                if current_ram > peak_ram:
                    peak_ram = current_ram
                if chunk.get("done"):
                    out_count = chunk.get("eval_count")
                    break
    except Exception as exc:
        result["error"] = str(exc)
        return result

    t_end = time.perf_counter()
    if t_first is None:
        result["error"] = "no tokens received"
        return result

    ram_after = get_system_ram_mb()
    ttft = t_first - t_start
    total = t_end - t_start
    gen_window = total - ttft
    output_tokens = out_count if (out_count and out_count > 0) else len("".join(tokens).split())
    tok_s = output_tokens / gen_window if gen_window > 0 else 0.0
    response_text = "".join(tokens).strip()

    result.update({
        "ttft_s": round(ttft, 4),
        "tokens_per_sec": round(tok_s, 2),
        "total_latency_s": round(total, 4),
        "output_tokens": output_tokens,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "ram_delta_mb": round(ram_after - ram_before, 1),
        "ram_peak_mb": round(peak_ram, 1),
        "response_text": response_text,
    })
    return result

def run_phase4(models, prompts, out_dir, temperature=0.0, quick=False):
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"phase4_{session_id}.csv"
    suite = prompts[:5] if quick else prompts
    total = len(models) * len(suite)
    counter = 0

    print(f"\n{'='*60}")
    print(f"  Phase 4 Quantization Comparison  -  session {session_id}")
    print(f"{'='*60}")
    print(f"  Base model : gemma2:2b-instruct")
    print(f"  Quant levels: q4_K_M, q5_K_M, q8_0")
    print(f"  Prompts    : {len(suite)} ({'quick' if quick else 'full'})")
    print(f"  Total runs : {total}")
    print(f"  Output     : {csv_path}")
    print(f"{'='*60}\n")
    print("  NOTE: score columns blank - fill manually after, same as Phase 3.\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for model in models:
            quant_level = extract_quant_level(model)
            disk_size = MODEL_DISK_SIZES_GB.get(model, None)
            print(f"\n--- {model}  ({quant_level}, {disk_size}GB on disk) ---")

            for prompt in suite:
                counter += 1
                run_id = f"{session_id}__{model.replace(':','_')}__{prompt['id']}"
                ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

                print(f"  [{counter:>3}/{total}] {prompt['id']}  ", end="", flush=True)
                m = measure_with_ram(model, prompt["text"], temperature)

                if m["error"]:
                    print(f"ERROR: {m['error']}")
                else:
                    print(
                        f"tok/s={m['tokens_per_sec']:.1f}  "
                        f"TTFT={m['ttft_s']:.2f}s  "
                        f"RAM_peak={m['ram_peak_mb']:.0f}MB  "
                        f"tokens={m['output_tokens']}"
                    )

                row = {
                    "run_id": run_id, "session_id": session_id, "timestamp_utc": ts,
                    "model": model, "quant_level": quant_level,
                    "prompt_id": prompt["id"], "prompt_category": prompt["category"],
                    "prompt_text": prompt["text"],
                    "ttft_s": m["ttft_s"], "tokens_per_sec": m["tokens_per_sec"],
                    "total_latency_s": m["total_latency_s"], "output_tokens": m["output_tokens"],
                    "ram_before_mb": m["ram_before_mb"], "ram_after_mb": m["ram_after_mb"],
                    "ram_delta_mb": m["ram_delta_mb"], "ram_peak_mb": m["ram_peak_mb"],
                    "model_disk_size_gb": disk_size,
                    "response_text": m["response_text"],
                    "score_correctness": "", "score_coherence": "",
                    "score_instruction": "", "score_overall": "",
                    "error": m["error"],
                }
                writer.writerow(row)
                fh.flush()

    print(f"\n  Done. {counter} runs written to:\n  {csv_path}")
    print(f"\n  Next: open CSV, fill in score columns (1-5), save, then run phase4_charts.py\n")
    return csv_path

def print_summary(csv_path):
    try:
        import pandas as pd
    except ImportError:
        return
    df = pd.read_csv(csv_path)
    ok = df[df["error"].isna()]
    if ok.empty:
        print("  No successful runs.")
        return
    print(f"\n{'='*60}")
    print("  Performance summary by quantization level (scores not yet filled)")
    print(f"{'='*60}")
    s = ok.groupby("quant_level").agg(
        runs=("run_id","count"),
        tok_s_mean=("tokens_per_sec","mean"),
        ttft_mean=("ttft_s","mean"),
        ram_peak_mean=("ram_peak_mb","mean"),
        disk_size_gb=("model_disk_size_gb","first"),
    ).round(2)
    print(s.to_string())
    print()

p = argparse.ArgumentParser()
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--out-dir", type=Path, default=Path("data/results"))
p.add_argument("--quick", action="store_true")
p.add_argument("--no-summary", action="store_true")
args = p.parse_args()
csv_path = run_phase4(QUANT_MODELS, PROMPTS, args.out_dir, args.temperature, args.quick)
if not args.no_summary:
    print_summary(csv_path)
