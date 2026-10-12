import json, subprocess, sys, statistics
from collections import Counter
from pathlib import Path

RUNS = 3   # each run uses some Gemini free quota
summary = {}

for csv in ["data/telco_leaky.csv", "data/telco_clean.csv"]:
    out = Path("results") / f"{Path(csv).name}.json"
    verdicts, calls, honest = [], [], []
    for i in range(RUNS):
        out.unlink(missing_ok=True)
        p = subprocess.run([sys.executable, "agent.py", csv, "--target", "Churn"],
                           capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            verdicts.append("ERROR")
            print(csv, "run", i + 1, "failed:", p.stderr[-300:])
            continue
        r = json.loads(out.read_text(encoding="utf-8"))
        verdicts.append(r["verdict"])
        calls.append(r["tool_calls_used"])
        if r.get("honest_accuracy") is not None:
            honest.append(r["honest_accuracy"])
    summary[csv] = {
        "verdicts": dict(Counter(verdicts)),
        "tool_calls_mean": round(statistics.mean(calls), 1) if calls else None,
        "tool_calls_min_max": [min(calls), max(calls)] if calls else None,
        "honest_accuracy_values": honest,
    }

Path("results/benchmark_after.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))