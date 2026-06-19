"""
phase3_comparison.py  -  Phase 3: Model Comparison Pipeline
============================================================
Collects responses, timing, and RAM usage per model per prompt.
Saves to CSV with empty score columns ready for manual scoring.

Run:
  python phase3_comparison.py --quick
  python phase3_comparison.py --models tinyllama qwen2:1.5b
"""

import argparse
import csv
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil

FASTAPI_BASE = "http://127.0.0.1:8000"
OLLAMA_BASE  = "http://127.0.0.1:11434"

# ---------------------------------------------------------------------------
# Prompt suite - 30 prompts across 6 categories
# ---------------------------------------------------------------------------
PROMPTS = [
    # Factual QA
    {"id": "fqa_01", "category": "factual_qa",
     "text": "What is the capital of Japan? One sentence."},
    {"id": "fqa_02", "category": "factual_qa",
     "text": "What does HTTP stand for? One sentence."},
    {"id": "fqa_03", "category": "factual_qa",
     "text": "What is the largest planet in our solar system? One sentence."},
    {"id": "fqa_04", "category": "factual_qa",
     "text": "Who wrote the play Romeo and Juliet? One sentence."},
    {"id": "fqa_05", "category": "factual_qa",
     "text": "What year did World War 2 end? One sentence."},
    # Summarisation
    {"id": "sum_01", "category": "summarisation",
     "text": "Summarise in one sentence: A neural network is a series of algorithms that attempt to recognise underlying relationships in a set of data through a process that mimics the way the human brain operates."},
    {"id": "sum_02", "category": "summarisation",
     "text": "Summarise in one sentence: Quantization reduces the precision of the numbers used to represent a model's parameters, which reduces memory usage and can speed up inference at the cost of some accuracy."},
    {"id": "sum_03", "category": "summarisation",
     "text": "Summarise in two sentences: The Transformer architecture, introduced in the paper Attention Is All You Need, replaced recurrent networks with self-attention mechanisms, enabling parallelisation during training and achieving state-of-the-art results on many NLP tasks."},
    {"id": "sum_04", "category": "summarisation",
     "text": "Summarise in one sentence: FastAPI is a modern Python web framework for building APIs with automatic data validation using Pydantic and automatic documentation generation."},
    {"id": "sum_05", "category": "summarisation",
     "text": "Summarise in one sentence: Ollama is a tool that lets you run large language models locally on your own hardware without needing an internet connection or cloud API."},
    # Instruction following
    {"id": "inst_01", "category": "instruction_following",
     "text": "List exactly three benefits of running AI models locally. Use bullet points. No other text."},
    {"id": "inst_02", "category": "instruction_following",
     "text": "Write exactly two sentences about Python. First sentence: what it is. Second sentence: one use case."},
    {"id": "inst_03", "category": "instruction_following",
     "text": "Answer only with a number: how many days are in a leap year?"},
    {"id": "inst_04", "category": "instruction_following",
     "text": "List the planets in our solar system in order from the sun. One planet per line. No numbering."},
    {"id": "inst_05", "category": "instruction_following",
     "text": "Respond with only the word CORRECT if 5 times 7 equals 35, or only the word WRONG if it does not."},
    # Reasoning
    {"id": "rsn_01", "category": "reasoning",
     "text": "A shop sells apples for 30p each. I buy 7 apples. How much do I pay in total? Show your working."},
    {"id": "rsn_02", "category": "reasoning",
     "text": "If today is Wednesday and my meeting is in 5 days, what day is my meeting? Explain briefly."},
    {"id": "rsn_03", "category": "reasoning",
     "text": "A bottle holds 750ml. I need 3 litres of water. How many bottles do I need? Show your working."},
    {"id": "rsn_04", "category": "reasoning",
     "text": "All dogs are mammals. Rex is a dog. Is Rex a mammal? Explain your reasoning in one sentence."},
    {"id": "rsn_05", "category": "reasoning",
     "text": "What is 20 percent of 350? Show your working briefly."},
    # JSON generation
    {"id": "json_01", "category": "json_generation",
     "text": "Return ONLY a valid JSON object with keys: name, capital, population_millions for France. No markdown."},
    {"id": "json_02", "category": "json_generation",
     "text": "Return ONLY a valid JSON array of two objects, each with keys: language, year_created for Python and JavaScript. No markdown."},
    {"id": "json_03", "category": "json_generation",
     "text": "Return ONLY a valid JSON object with keys: model, parameters_billions, use_case for the tinyllama model. No markdown."},
    # Edge cases
    {"id": "edge_01", "category": "edge_case",
     "text": "Respond with exactly one word: the opposite of hot."},
    {"id": "edge_02", "category": "edge_case",
     "text": "What is the meaning of life? Answer in exactly three words."},
    {"id": "edge_03", "category": "edge_case",
     "text": "Translate good morning into French, Spanish, and German. Format: Language: Translation, one per line."},
    {"id": "edge_04", "category": "edge_case",
     "text": "Count from 1 to 5. Output only the numbers, one per line, nothing else."},
    # Longer generation
    {"id": "lng_01", "category": "longer_generation",
     "text": "Write 3 to 5 sentences explaining what an API is and why developers use them."},
    {"id": "lng_02", "category": "longer_generation",
     "text": "Write 3 to 5 sentences explaining the difference between RAM and storage in a computer."},
    {"id": "lng_03", "category": "longer_generation",
     "text": "Write 3 to 5 sentences explaining why privacy matters when using AI assistants."},
]

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
CSV_FIELDNAMES = [
    "run_id", "session_id", "timestamp_utc", "model", "prompt_id",
    "prompt_category", "prompt_text",
    # timing (reused from Phase 1)
    "ttft_s", "tokens_per_sec", "total_latency_s", "output_tokens",
    # RAM
    "ram_before_mb", "ram_after_mb", "ram_delta_mb", "ram_peak_mb",
    # response
    "response_text",
    # quality scores - LEFT BLANK for manual scoring
    "score_correctness",    # 1-5
    "score_coherence",      # 1-5
    "score_instruction",    # 1-5
    "score_overall",        # average of above three
    # error
    "error",
]

# ---------------------------------------------------------------------------
# RAM snapshot
# ---------------------------------------------------------------------------
def get_ram_mb():
    """Current process + system RAM used in MB."""
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)

def get_system_ram_mb():
    """Total system RAM used in MB."""
    return round(psutil.virtual_memory().used / 1024 / 1024, 1)

# ---------------------------------------------------------------------------
# Single timed + RAM-tracked call
# ---------------------------------------------------------------------------
def measure_with_ram(model, prompt_text, temperature=0.0):
    """
    Stream response, measure TTFT/tok/s/latency, track RAM before and after.
    Returns dict with all metrics + full response text.
    """
    result = {
        "ttft_s": None, "tokens_per_sec": None, "total_latency_s": None,
        "output_tokens": None, "ram_before_mb": None, "ram_after_mb": None,
        "ram_delta_mb": None, "ram_peak_mb": None,
        "response_text": None, "error": None,
    }

    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "options": {"temperature": temperature},
    }

    # RAM before
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
                # Track peak RAM during generation
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

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_phase3(models, prompts, out_dir, temperature=0.0, quick=False):
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"phase3_{session_id}.csv"
    suite = prompts[:10] if quick else prompts
    total = len(models) * len(suite)
    counter = 0

    print(f"\n{'='*55}")
    print(f"  Phase 3 Comparison  -  session {session_id}")
    print(f"{'='*55}")
    print(f"  Models   : {', '.join(models)}")
    print(f"  Prompts  : {len(suite)} ({'quick' if quick else 'full'})")
    print(f"  Temp     : {temperature}")
    print(f"  Total    : {total} runs")
    print(f"  Output   : {csv_path}")
    print(f"{'='*55}\n")
    print("  NOTE: score columns will be blank - fill them in manually after.")
    print("        score_correctness, score_coherence, score_instruction (1-5 each)\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for model in models:
            print(f"\n--- {model} ---")
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
                        f"RAM_delta={m['ram_delta_mb']:+.0f}MB  "
                        f"tokens={m['output_tokens']}"
                    )

                row = {
                    "run_id": run_id,
                    "session_id": session_id,
                    "timestamp_utc": ts,
                    "model": model,
                    "prompt_id": prompt["id"],
                    "prompt_category": prompt["category"],
                    "prompt_text": prompt["text"],
                    "ttft_s": m["ttft_s"],
                    "tokens_per_sec": m["tokens_per_sec"],
                    "total_latency_s": m["total_latency_s"],
                    "output_tokens": m["output_tokens"],
                    "ram_before_mb": m["ram_before_mb"],
                    "ram_after_mb": m["ram_after_mb"],
                    "ram_delta_mb": m["ram_delta_mb"],
                    "ram_peak_mb": m["ram_peak_mb"],
                    "response_text": m["response_text"],
                    # blank score columns - fill manually
                    "score_correctness": "",
                    "score_coherence": "",
                    "score_instruction": "",
                    "score_overall": "",
                    "error": m["error"],
                }
                writer.writerow(row)
                fh.flush()

    print(f"\n  Done. {counter} runs written to:\n  {csv_path}")
    print(f"\n  Next step: open the CSV and fill in score columns (1-5) for each response.")
    print(f"  Then run: python phase3_charts.py\n")
    return csv_path

# ---------------------------------------------------------------------------
# Quick summary (no scores yet)
# ---------------------------------------------------------------------------
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
    print(f"\n{'='*55}")
    print("  Performance summary (scores not yet filled in)")
    print(f"{'='*55}")
    s = ok.groupby("model").agg(
        runs=("run_id","count"),
        tok_s_mean=("tokens_per_sec","mean"),
        ttft_mean=("ttft_s","mean"),
        latency_mean=("total_latency_s","mean"),
        ram_peak_mean=("ram_peak_mb","mean"),
        ram_delta_mean=("ram_delta_mb","mean"),
    ).round(2)
    print(s.to_string())
    print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["tinyllama", "qwen2:1.5b"])
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--out-dir", type=Path, default=Path("data/results"))
p.add_argument("--quick", action="store_true", help="Run only first 10 prompts")
p.add_argument("--no-summary", action="store_true")
args = p.parse_args()
csv_path = run_phase3(args.models, PROMPTS, args.out_dir, args.temperature, args.quick)
if not args.no_summary:
    print_summary(csv_path)
