import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("data/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/results/phase3_gpu_scored_utf8.csv")
df = df[df["error"].isna()]
df = df.dropna(subset=["score_overall"])

print(f"Loaded {len(df)} scored rows\n")

# ---------- Chart 6: GPU tokens/sec bar chart ----------
summary = df.groupby("model")["tokens_per_sec"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(summary)))
bars = ax.bar(summary.index, summary.values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Tokens / second", fontsize=11)
ax.set_title("GPU Hero Run: Mean Generation Speed by Model\n(RTX 4090)", fontsize=13, fontweight="bold")
ax.set_ylim(0, summary.values.max() * 1.2)
for bar, val in zip(bars, summary.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f"{val:.1f}",
             ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "chart6_gpu_tokens_per_sec.png", dpi=150)
plt.close()
print("Saved: chart6_gpu_tokens_per_sec.png")

# ---------- Chart 7: GPU scatter - RAM vs speed, bubble = quality ----------
scatter_summary = df.groupby("model").agg(
    ram_peak_mb=("ram_peak_mb", "mean"),
    tokens_per_sec=("tokens_per_sec", "mean"),
    score_overall=("score_overall", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(scatter_summary)))
for i, row in scatter_summary.iterrows():
    size = (row["score_overall"] ** 2) * 150
    ax.scatter(row["ram_peak_mb"], row["tokens_per_sec"], s=size,
              color=colors[i], alpha=0.7, edgecolor="black", linewidth=1.2,
              label=f"{row['model']} (quality={row['score_overall']:.1f})")
    ax.annotate(row["model"], (row["ram_peak_mb"], row["tokens_per_sec"]),
               textcoords="offset points", xytext=(0, 18), ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Peak system RAM (MB)", fontsize=11)
ax.set_ylabel("Tokens / second", fontsize=11)
ax.set_title("GPU Hero Run: Speed vs Memory\n(bubble size = quality score)", fontsize=13, fontweight="bold")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "chart7_gpu_ram_vs_speed.png", dpi=150)
plt.close()
print("Saved: chart7_gpu_ram_vs_speed.png")

# ---------- Chart 8: GPU radar - quality dimensions ----------
dims = ["score_correctness", "score_coherence", "score_instruction"]
dim_labels = ["Correctness", "Coherence", "Instruction\nFollowing"]
radar_summary = df.groupby("model")[dims].mean()

angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(radar_summary)))
for i, (model, row) in enumerate(radar_summary.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_labels, fontsize=10)
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_title("GPU Hero Run: Quality Profile by Model", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "chart8_gpu_quality_radar.png", dpi=150)
plt.close()
print("Saved: chart8_gpu_quality_radar.png")

# ---------- Summary table ----------
print(f"\n{'='*70}")
print("  GPU Hero Run - Final Summary")
print(f"{'='*70}")
final = df.groupby("model").agg(
    tok_s_mean=("tokens_per_sec", "mean"),
    ttft_mean=("ttft_s", "mean"),
    ram_peak_mean=("ram_peak_mb", "mean"),
    correctness=("score_correctness", "mean"),
    coherence=("score_coherence", "mean"),
    instruction=("score_instruction", "mean"),
    overall_quality=("score_overall", "mean"),
).round(2)
print(final.to_string())
print(f"\nAll charts saved to: {OUT_DIR}\n")
