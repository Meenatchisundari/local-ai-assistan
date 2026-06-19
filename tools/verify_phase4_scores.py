import pandas as pd
from pathlib import Path

path = Path("data/phase4_scored.csv")
if not path.exists():
    path = Path("data/results/phase4_scored.csv")

print(f"Reading: {path}")

df = None
for enc in ["utf-8", "cp1252", "latin1", "utf-8-sig"]:
    try:
        df = pd.read_csv(path, encoding=enc)
        print(f"Successfully read with encoding: {enc}")
        break
    except Exception as e:
        print(f"Failed with {enc}")

if df is None:
    print("Could not read file")
else:
    clean_path = path.parent / "phase4_scored_utf8.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"\nRe-saved as UTF-8: {clean_path}")

    print(f"\nTotal rows: {len(df)}")
    print(f"\nUnique quant_level values:")
    for q in df["quant_level"].unique():
        print(f"  {repr(q)}")

    score_cols = ["score_correctness", "score_coherence", "score_instruction", "score_overall"]
    print("\nMissing/blank scores per column:")
    for col in score_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} missing out of {len(df)}")

    print("\nScore ranges (should be 1-5):")
    for col in ["score_correctness", "score_coherence", "score_instruction"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                print(f"  {col}: min={valid.min()}, max={valid.max()}")

    print("\nSample of scored rows:")
    print(df[["quant_level", "prompt_id", "score_correctness", "score_coherence", "score_instruction", "score_overall"]].head(10).to_string(index=False))
