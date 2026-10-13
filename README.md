# Leakhound

Leakhound is a small agent-assisted auditor for possible machine-learning data
leakage. It currently checks for suspicious individual columns and for exact
feature rows that occur in both the training and test partitions. The
preprocessing-order check is a static AST warning; it does not run or validate
the analyzed script.

## What it detects

- **Column-level leakage candidates:** a single-column decision-tree scan ranks
  features by three-fold cross-validated accuracy and reports missingness by
  target class. For selected candidates, Leakhound retrains without the column.
  A measured accuracy drop of at least 0.05 (absolute accuracy, not percent
  relative change) can confirm a column-level finding.
- **Exact split duplicates:** the tool reproduces the 80/20 stratified split
  (`random_state=42`), hashes feature rows, and identifies test rows whose
  feature hashes occur in training. It measures accuracy on the test rows that
  do not occur in training. The agent's duplicate verdict requires more than
  2% duplicate test rows and an accuracy decrease of at least 0.02.
- **Preprocessing order (static suspicion):** `check_preprocessing_order`
  looks for scaler, imputer, or encoder `fit`/`fit_transform` calls textually
  before a `train_test_split` call. Its output is explicitly labelled
  **suspected, not measured**; it is not a general Python control-flow or
  notebook analyzer.

The training and duplicate checks use the pipeline in `train_demo.py`: a
stratified train/test split, categorical imputation and one-hot encoding,
numeric median imputation, and a random forest classifier.

## How the agent works

`agent.py` computes a baseline training score, then provides Gemini with three
tools: the single-column scan, drop-and-retrain, and split-duplicate check.
Tool results are cached within an audit. The agent enforces measured thresholds
in code and asks Gemini for a constrained JSON assessment. Neatlogs records the
workflow, model calls, and tool spans. The model request code retries
rate-limit/service-unavailable errors and then attempts the configured fallback
model; the fallback may also be unavailable.

## Results in this repository

The table below summarizes the existing result files as stored; they are
historical runs and have not been re-created by this documentation change.

| Result file | Data / observation |
|---|---|
| `results/telco_leaky.csv.json` | `leak_confirmed`; `cancellation_reason`; reported accuracy 0.9972 and honest accuracy 0.78 |
| `results/telco_clean.csv.json` | `no_column_leak_found`; baseline accuracy 0.78; three columns tested in the saved report |
| `results/telco_leaky_before.json` | Older leaky report; reports `leak_confirmed` and honest accuracy 0.78, but includes model-authored claims not enforced by current report generation |
| `results/telco_clean_before.json` | Older clean report uses the verdict label `no_leak`; not the current verdict schema |
| `results/telco_leaky_fix.json` | Records a fixed-file check: `cancellation_reason` removed, accuracy 0.78, `fix_verified: true` |
| `results/benchmark_after.json` | Saved three-run summary: leaky verdict in 3/3 runs; clean verdict in 3/3 runs |
| `results/benchmark_after_run2.json` | Another saved summary: leaky verdict in 3/3 runs; clean verdict in 2/3 runs and one `ERROR` |

These records do not validate the new duplicate or static preprocessing checks.
The benchmark script `bench.py` runs the LLM agent repeatedly and consumes  
Gemini quota; do not use it for ordinary tool tests.

## Setup and run

Install the packages in `requirements.txt`, configure the Gemini and neatlogs
environment variables expected by `agent.py` in your local environment, then
run:

```text
python agent.py data/telco_leaky.csv --target Churn
```

The report is written to `results/<csv-filename>.json`. `agent.py` requires
working Gemini credentials, an available model/quota, and neatlogs
configuration. The free-tier quota is limited.

Create the Telco sample data (requires network access to OpenML):

```text
python data_loading.py
```

Run the local browser UI:

```text
python app.py
```

The UI binds to localhost only. Uploaded CSVs must have a binary, non-missing
target column. Training is performed with this repository's `train_demo.py`
pipeline; arbitrary notebooks or user-defined training pipelines are **not
supported yet**.

Run the direct tool tests without invoking the LLM:

```text
python -m pytest tests/test_tools.py -q
```

