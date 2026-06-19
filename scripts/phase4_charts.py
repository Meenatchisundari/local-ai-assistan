"""
phase4_charts.py  -  Phase 4: Quantization Tradeoff Charts
=============================================================
Reads the scored Phase 4 CSV and generates two charts:
  1. Quality vs tokens/sec  - does compression cost quality?
  2. Memory vs quantization level - how much RAM is actually saved?

Run:
  python phase4_charts.py
'''
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("data/charts")

# Order matters - most compressed to least compressed
QUANT_ORDER = ["q4_K_M", "q5_K_M", "q8_0"]

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["error"].isna()]
    df = df.dropna(subset=["score_overall"])
    return df

def chart_quality_vs_speed(df, out_dir):
    """Chart: tokens/sec (bar) vs quality score (line) per quantization level."""
    summary = df.groupby("quant_level").agg(
        tok_s=("tokens_per_sec", "mean"),
        quality=("score_overall", "mean"),
    ).reindex(QUANT_ORDER)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(summary)))
    bars = ax1.bar(summary.index, summary["tok_s"], color=colors,
                   edgecolor="black", linewidth=0.5, alpha=0.85, label="Tokens/sec")
    ax1.set_xlabel("Quantization level (most -> least compressed)", fontsize=11)
    ax1.set_ylabel("Tokens / second", fontsize=11, color="#2c5f8a")
    ax1.set_ylim(0, summary["tok_s"].max() * 1.3)

    for bar, val in zip(bars, summary["tok_s"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(summary.index, summary["quality"], color="#d62728", marker="o",
             markersize=10, linewidth=2.5, label="Quality score")
    ax2.set_ylabel("Quality score (1-5)", fontsize=11, color="#d62728")
    ax2.set_ylim(0, 5.5)

    for x, val in zip(summary.index, summary["quality"]):
        ax2.text(x, val + 0.15, f"{val:.2f}", ha="center", fontsize=10,
                 fontweight="bold", color="#d62728")

    ax1.set_title("Quantization Tradeoff: Speed vs Quality\n(gemma2:2b-instruct)",
                  fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    path = out_dir / "chart4_quality_vs_speed.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path

def chart_ram_vs_quantization(df, out_dir):
    """Chart: RAM usage and disk size across quantization levels."""
    summary = df.groupby("quant_level").agg(
        ram_peak=("ram_peak_mb", "mean"),
        disk_gb=("model_disk_size_gb", "first"),
    ).reindex(QUANT_ORDER)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    x = np.arange(len(summary))
    width = 0.35

    colors_ram = plt.cm.Blues(np.linspace(0.5, 0.85, len(summary)))
    bars1 = ax1.bar(x - width/2, summary["ram_peak"] / 1024, width,
                    color=colors_ram, edgecolor="black", linewidth=0.5, label="Peak RAM (GB)")
    ax1.set_ylabel("Peak system RAM (GB)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary.index)
    ax1.set_xlabel("Quantization level", fontsize=11)

    ax2 = ax1.twinx()
    colors_disk = plt.cm.Oranges(np.linspace(0.5, 0.85, len(summary)))
    bars2 = ax2.bar(x + width/2, summary["disk_gb"], width,
                    color=colors_disk, edgecolor="black", linewidth=0.5, label="Disk size (GB)")
    ax2.set_ylabel("Model disk size (GB)", fontsize=11)

    for bar, val in zip(bars1, summary["ram_peak"] / 1024):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, summary["disk_gb"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax1.set_title("Memory Footprint by Quantization Level\n(gemma2:2b-instruct)",
                  fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    path = out_dir / "chart5_ram_vs_quantization.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path

def print_summary_table(df):
    print(f"\n{'='*65}")
    print("  Phase 4 Final Summary")
    print(f"{'='*65}")
    summary = df.groupby("quant_level").agg(
        tok_s_mean=("tokens_per_sec", "mean"),
        ttft_mean=("ttft_s", "mean"),
        ram_peak_mean=("ram_peak_mb", "mean"),
        disk_size_gb=("model_disk_size_gb", "first"),
        correctness=("score_correctness", "mean"),
        coherence=("score_coherence", "mean"),
        instruction=("score_instruction", "mean"),
        overall_quality=("score_overall", "mean"),
    ).reindex(QUANT_ORDER).round(2)
    print(summary.to_string())
    print()

p = argparse.ArgumentParser()
p.add_argument("--csv", type=Path, default=Path("data/results/phase4_scored_utf8.csv"))
args = p.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
df = load_data(args.csv)

print(f"\nLoaded {len(df)} scored rows from {args.csv}\n")
print("Generating charts...")
chart_quality_vs_speed(df, OUT_DIR)
chart_ram_vs_quantization(df, OUT_DIR)
print_summary_table(df)
print(f"All charts saved to: {OUT_DIR}\n")
