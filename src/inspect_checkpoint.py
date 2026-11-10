"""Checks both .pt files load, and prints the CSV files in models/."""
from pathlib import Path
import pandas as pd
from model_loader import load_model

for f in sorted(Path("models").glob("*.pt")):
    print(f"\n=== {f.name} ===")
    try:
        model, names, size = load_model(f)
        print(f"Loaded OK | classes: {names} | image size: {size}")
    except Exception as e:
        print("LOAD FAILED:", e)

for c in sorted(Path("models").glob("*.csv")):
    print(f"\n=== {c.name} ===")
    df = pd.read_csv(c)
    print(df.head(10).to_string(index=False) if len(df) > 10 else df.to_string(index=False))
