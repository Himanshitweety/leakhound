import pandas as pd
import gradio as gr

import agent


def audit_upload(csv_path, target):
    if not csv_path:
        return "Choose a CSV file.", "", None, []
    target = (target or "").strip()
    if not target:
        return "Enter the target column name.", "", None, []

    try:
        data = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return "The uploaded file could not be read as a CSV.", "", None, []
    if target not in data.columns:
        return f"Target column {target!r} was not found.", "", None, []
    if data[target].isna().any() or data[target].nunique() != 2:
        return (
            "The target column must contain exactly two non-missing classes.",
            "",
            None,
            [],
        )

    report = agent.main([csv_path, "--target", target])

    error = report.get("error", "") if isinstance(report, dict) else ""
    if isinstance(report, dict) and report.get("error_status") == 429:
        return (
            "Gemini's free-tier quota is exhausted (HTTP 429). Please try again "
            "after the quota resets.",
            "",
            None,
            report.get("evidence", []) if isinstance(report, dict) else [],
        )
    if not isinstance(report, dict):
        return "The audit returned no report.", "", None, []
    if error:
        return (
            "The audit could not be completed. Check the local agent configuration.",
            "",
            None,
            report.get("evidence", []),
        )

    verdict = report.get("verdict", "needs_human")
    duplicate_result = next(
        (
            item
            for item in report.get("evidence", [])
            if item.get("tool") == "check_split_duplicates"
        ),
        None,
    )
    if verdict == "duplicate_leak_confirmed" and duplicate_result:
        suspect_or_duplicates = (
            f"Duplicate test rows: {duplicate_result['duplicate_count']}"
        )
    else:
        suspect_or_duplicates = report.get("suspect_column") or "None identified"
    verdict_text = verdict
    if report.get("explanation"):
        verdict_text = f"{verdict}: {report['explanation']}"

    return (
        verdict_text,
        suspect_or_duplicates,
        report.get("honest_accuracy"),
        report.get("evidence", []),
    )


demo = gr.Interface(
    fn=audit_upload,
    inputs=[
        gr.File(type="filepath", label="Upload CSV"),
        gr.Textbox(value="Churn", label="Target column"),
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Textbox(label="Suspect column / duplicate rows"),
        gr.Number(label="Honest accuracy", precision=4),
        gr.JSON(label="Evidence"),
    ],
    title="Leakhound",
    description=(
        "Audit a CSV for column-level and train/test duplicate leakage. "
        "The target must be binary; training uses this repository's "
        "train_demo.py pipeline."
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
