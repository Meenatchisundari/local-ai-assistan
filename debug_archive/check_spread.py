import pandas as pd
df = pd.read_csv("data/results/phase3_scored_utf8.csv")

print("qwen2:1.5b - ram_peak_mb and tokens_per_sec, sorted by ram_peak_mb:")
q = df[df["model"] == "qwen2:1.5b"][["prompt_id", "ram_peak_mb", "tokens_per_sec"]].sort_values("ram_peak_mb")
print(q.to_string(index=False))

print()
print("qwen2:1.5b ram_peak_mb stats:")
print(df[df["model"] == "qwen2:1.5b"]["ram_peak_mb"].describe())
