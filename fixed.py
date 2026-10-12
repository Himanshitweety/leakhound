"""Build a corrected copy of the dataset and verify the fix. Never edits the original."""
import json, sys
from pathlib import Path
import pandas as pd
from tools import run_training, scan_suspect_columns

def main(csv_path, target="Churn"):
    csv = Path(csv_path)
    report = json.loads((Path("results") / f"{csv.name}.json").read_text(encoding="utf-8"))
    if report.get("verdict") != "leak_confirmed":
        print("No confirmed leak, nothing to fix.")
        return
    col = report["suspect_column"]
    fixed = Path("data") / f"{csv.stem}_fixed.csv"
    pd.read_csv(csv).drop(columns=[col]).to_csv(fixed, index=False)

    run = run_training(str(fixed))
    scan = scan_suspect_columns(str(fixed), target)
    honest = report["honest_accuracy"]
    top = max(s["single_column_accuracy"] for s in scan)
    checks = {
        "column_removed": col not in pd.read_csv(fixed).columns,
        "fixed_accuracy": run.get("accuracy"),
        "matches_honest_score": bool(run.get("ok") and abs(run["accuracy"] - honest) <= 0.005),
        "top_single_column_accuracy_after_fix": top,
        "no_suspect_left": top < 0.95,
    }
    out = {
        "fixed_file": str(fixed),
        "removed_column": col,
        "reported_accuracy": report["reported_accuracy"],
        "honest_accuracy": honest,
        "checks": checks,
        "fix_verified": all([checks["column_removed"], checks["matches_honest_score"], checks["no_suspect_left"]]),
    }
    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"{csv.stem}_fix.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1])