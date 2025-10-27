# merge_datasets.py
import pandas as pd
from pathlib import Path

base = Path(r"C:\Users\User\Desktop\phishing detection")

# File paths
paths = {
    "main": base / "cleaned_dataset.csv",
    "synth": base / "synthetic_phishing_variants.csv",
    "legit": base / "legit_brand_emails.csv",
    "out": base / "final_merged_dataset.csv"
}

# Helper to load CSV safely
def load_csv(p):
    if p.exists():
        print(f"Loaded: {p.name}")
        return pd.read_csv(p)
    else:
        print(f"⚠️ Warning: {p.name} not found — using empty DataFrame.")
        return pd.DataFrame(columns=["subject", "body", "label"])

# Load datasets
df_main = load_csv(paths["main"])
df_synth = load_csv(paths["synth"])
df_legit = load_csv(paths["legit"])

# Ensure all required columns exist
required_cols = ["subject", "body", "label"]
for df in (df_main, df_synth, df_legit):
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

# Assign labels
if not df_synth.empty:
    df_synth["label"] = 1
if not df_legit.empty:
    df_legit["label"] = 0

# Merge, drop duplicates, shuffle
merged = pd.concat([df_main, df_synth, df_legit], ignore_index=True)
merged = merged.drop_duplicates(subset="subject", keep="first")
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
merged.to_csv(paths["out"], index=False)

print(f"\n✅ Merged dataset saved to: {paths['out']}")
print(f"Total rows: {len(merged)}")
print(f"Phishing (1): {(merged['label']==1).sum()}")
print(f"Legit (0): {(merged['label']==0).sum()}")
