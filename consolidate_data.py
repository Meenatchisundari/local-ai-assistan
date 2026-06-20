import pandas as pd
from pathlib import Path

print("="*70)
print("  PHASE 1 - CPU baseline (tinyllama, qwen2:1.5b)")
print("="*70)
p1 = sorted(Path("data/results").glob("benchmark_2026061[8]*.csv"))
if p1:
    df = pd.read_csv(p1[-1])
    ok = df[df["error"].isna()]
    print(ok.groupby("model").agg(
        tok_s=("tokens_per_sec","mean"), ttft=("ttft_s","mean"),
        latency=("total_latency_s","mean")).round(2).to_string())

print("\n" + "="*70)
print("  PHASE 3 - CPU comparison + quality (tinyllama, qwen2:1.5b)")
print("="*70)
p3 = Path("data/results/phase3_scored_utf8.csv")
if p3.exists():
    df = pd.read_csv(p3)
    print(df.groupby("model").agg(
        tok_s=("tokens_per_sec","mean"), ram_peak=("ram_peak_mb","mean"),
        quality=("score_overall","mean")).round(2).to_string())

print("\n" + "="*70)
print("  PHASE 4 - Quantization (gemma2:2b-instruct)")
print("="*70)
p4 = Path("data/results/phase4_scored_utf8.csv")
if p4.exists():
    df = pd.read_csv(p4)
    print(df.groupby("quant_level").agg(
        tok_s=("tokens_per_sec","mean"), ram_peak=("ram_peak_mb","mean"),
        disk_gb=("model_disk_size_gb","first"),
        quality=("score_overall","mean")).reindex(["q4_K_M","q5_K_M","q8_0"]).round(2).to_string())

print("\n" + "="*70)
print("  PHASE 5 - GPU baseline (mistral, llama3, phi3, qwen2:7b)")
print("="*70)
p5a = Path("data/results/benchmark_20260619T121201.csv")
if p5a.exists():
    df = pd.read_csv(p5a)
    print(df.groupby("model").agg(
        tok_s=("tokens_per_sec","mean"), ttft=("ttft_s","mean"),
        latency=("total_latency_s","mean")).round(2).to_string())

print("\n" + "="*70)
print("  PHASE 5 - GPU comparison + quality")
print("="*70)
p5b = Path("data/results/phase3_gpu_scored_utf8.csv")
if p5b.exists():
    df = pd.read_csv(p5b)
    print(df.groupby("model").agg(
        tok_s=("tokens_per_sec","mean"), ram_peak=("ram_peak_mb","mean"),
        quality=("score_overall","mean")).round(2).to_string())
