import pandas as pd
df = pd.read_csv("data/results/phase3_scored_utf8.csv")
print("Unique model values:")
for m in df["model"].unique():
    print(f"  {repr(m)}")
print()
print("Row counts per exact model string:")
print(df["model"].value_counts())
