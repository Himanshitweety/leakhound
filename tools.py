import re, subprocess, sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def run_training(csv_path, drop=None):
    cmd = [sys.executable, "train_demo.py", csv_path]
    if drop:
        cmd += ["--drop", drop]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    m = re.search(r"accuracy:\s*([0-9.]+)", out.stdout)
    if not m:
        return {"ok": False, "error": (out.stderr or out.stdout)[-500:]}
    return {"ok": True, "accuracy": float(m.group(1)), "dropped": drop}

def scan_suspect_columns(csv_path, target="Churn"):
    df = pd.read_csv(csv_path)
    y = df[target]
    results = []
    for col in df.columns:
        if col == target:
            continue
        missing_by_class = df[col].isna().groupby(y).mean().round(3).to_dict()
        if pd.api.types.is_numeric_dtype(df[col]):
            X = df[[col]].fillna(-1)
        else:
            X = pd.get_dummies(df[[col]].astype("object").fillna("__MISSING__"))
        acc = cross_val_score(
            DecisionTreeClassifier(max_depth=3, random_state=42), X, y, cv=3
        ).mean()
        results.append({
            "column": col,
            "missing_rate_by_class": missing_by_class,
            "single_column_accuracy": round(float(acc), 4),
        })
    results.sort(key=lambda r: r["single_column_accuracy"], reverse=True)
    return results[:5]