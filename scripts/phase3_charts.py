"""
phase3_charts.py  -  Phase 3: Comparison Charts
=================================================
Reads the scored CSV and generates three charts:
  1. Bar chart   - tokens/sec per model
  2. Scatter plot - RAM vs tokens/sec, bubble = quality
  3. Radar chart - quality dimensions per model

Run:
  python phase3_charts.py
  python phase3_charts.py --csv data/results/phase3_scored_utf8.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("data/charts")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Drop rows with missing scores or errors
    df = df[df["error"].isna()]
    df = df.dropna(subset=["score_overall"])
    return df

def chart_bar_tokens_per_sec(df, out_dir):
    """Chart 1: Simple bar chart of mean tokens/sec per model."""
    summary = df.groupby("model")["tokens_per_sec"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(summary)))
    bars = ax.bar(summary.index, summary.values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_title("Mean Generation Speed by Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, summary.values.max() * 1.2)

    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = out_dir / "chart1_tokens_per_sec.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path

def chart_scatter_ram_vs_speed(df, out_dir):
    """Chart 2: Scatter - RAM (x) vs tokens/sec (y), bubble size = quality."""
    summary = df.groupby("model").agg(
        ram_peak_mb=("ram_peak_mb", "mean"),
        tokens_per_sec=("tokens_per_sec", "mean"),
        score_overall=("score_overall", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(summary)))

    for i, row in summary.iterrows():
        size = (row["score_overall"] ** 2) * 150  # scale bubble noticeably with score
        ax.scatter(row["ram_peak_mb"], row["tokens_per_sec"], s=size,
                  color=colors[i], alpha=0.7, edgecolor="black", linewidth=1.2,
                  label=f"{row['model']} (quality={row['score_overall']:.1f})")
        ax.annotate(row["model"], (row["ram_peak_mb"], row["tokens_per_sec"]),
                   textcoords="offset points", xytext=(0, 18), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Peak RAM used (MB)", fontsize=11)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_title("Efficiency: Speed vs Memory\n(bubble size = quality score)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "chart2_ram_vs_speed.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path

def chart_radar_quality(df, out_dir):
    """Chart 3: Radar chart - correctness, coherence, instruction-following per model."""
    dims = ["score_correctness", "score_coherence", "score_instruction"]
    dim_labels = ["Correctness", "Coherence", "Instruction\nFollowing"]

    summary = df.groupby("model")[dims].mean()

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(summary)))

    for i, (model, row) in enumerate(summary.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("Quality Profile by Model", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    path = out_dir / "chart3_quality_radar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path

def print_summary_table(df):
    print(f"\n{'='*60}")
    print("  Phase 3 Final Summary")
    print(f"{'='*60}")
    summary = df.groupby("model").agg(
        tok_s_mean=("tokens_per_sec", "mean"),
        ttft_mean=("ttft_s", "mean"),
        ram_peak_mean=("ram_peak_mb", "mean"),
        correctness=("score_correctness", "mean"),
        coherence=("score_coherence", "mean"),
        instruction=("score_instruction", "mean"),
        overall_quality=("score_overall", "mean"),
    ).round(2)
    print(summary.to_string())
    print()

p = argparse.ArgumentParser()
p.add_argument("--csv", type=Path, default=Path("data/results/phase3_scored_utf8.csv"))
args = p.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
df = load_data(args.csv)

print(f"\nLoaded {len(df)} scored rows from {args.csv}\n")
print("Generating charts...")
chart_bar_tokens_per_sec(df, OUT_DIR)
chart_scatter_ram_vs_speed(df, OUT_DIR)
chart_radar_quality(df, OUT_DIR)
print_summary_table(df)
print(f"All charts saved to: {OUT_DIR}\n")
