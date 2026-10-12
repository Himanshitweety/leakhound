import ast
import re
import subprocess
import sys

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from train_demo import build_model

def run_training(csv_path, drop=None, target="Churn"):
    cmd = [sys.executable, "train_demo.py", csv_path, "--target", target]
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


def check_split_duplicates(csv_path, target="Churn"):
    data = pd.read_csv(csv_path)
    labels = data[target]
    features = data.drop(columns=target)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_hashes = set(
        pd.util.hash_pandas_object(x_train, index=False).tolist()
    )
    test_hashes = pd.util.hash_pandas_object(x_test, index=False)
    duplicate_mask = test_hashes.isin(train_hashes).to_numpy()

    model = build_model(x_train)
    model.fit(x_train, y_train)
    baseline_accuracy = accuracy_score(y_test, model.predict(x_test))
    unique_test_mask = ~duplicate_mask
    if not unique_test_mask.any():
        raise ValueError("No test rows remain after excluding train duplicates.")
    honest_accuracy = accuracy_score(
        y_test.iloc[unique_test_mask],
        model.predict(x_test.loc[unique_test_mask]),
    )
    duplicate_count = int(duplicate_mask.sum())

    return {
        "duplicate_count": duplicate_count,
        "duplicate_fraction_of_test": duplicate_count / len(x_test),
        "baseline_accuracy": float(baseline_accuracy),
        "honest_accuracy": float(honest_accuracy),
        "accuracy_drop": float(baseline_accuracy - honest_accuracy),
    }


def check_preprocessing_order(script_path):
    with open(script_path, encoding="utf-8") as script:
        source = script.read()
    tree = ast.parse(source, filename=script_path)
    split_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            getattr(node.func, "id", None) == "train_test_split"
            or getattr(node.func, "attr", None) == "train_test_split"
        )
    ]
    split_line = min(split_lines) if split_lines else None

    fitted_objects = set()
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call):
                constructor = (
                    getattr(value.func, "id", "")
                    or getattr(value.func, "attr", "")
                ).lower()
                if any(
                    kind in constructor
                    for kind in ("scaler", "imputer", "encoder")
                ):
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                    )
                    fitted_objects.update(
                        target.id
                        for target in targets
                        if isinstance(target, ast.Name)
                    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func, ast.Attribute
        ):
            continue
        if node.func.attr not in ("fit", "fit_transform"):
            continue
        receiver = node.func.value
        constructor = (
            getattr(receiver.func, "id", "")
            or getattr(receiver.func, "attr", "")
        ).lower() if isinstance(receiver, ast.Call) else ""
        is_preprocessor = isinstance(receiver, ast.Name) and (
            receiver.id in fitted_objects
        )
        is_preprocessor = is_preprocessor or any(
            kind in constructor for kind in ("scaler", "imputer", "encoder")
        )
        if is_preprocessor and split_line is not None and node.lineno < split_line:
            findings.append(
                {
                    "line": node.lineno,
                    "operation": node.func.attr,
                    "object": (
                        receiver.id
                        if isinstance(receiver, ast.Name)
                        else constructor
                    ),
                }
            )

    findings.sort(key=lambda finding: finding["line"])
    return {
        "status": "suspected, not measured",
        "suspected": bool(findings),
        "findings": findings,
        "train_test_split_line": split_line,
    }