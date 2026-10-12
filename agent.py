import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

import neatlogs
from tools import (
    check_split_duplicates,
    run_training,
    scan_suspect_columns,
)


MAX_TOOL_STEPS = 8
MAX_RETRIES = 3
LEAK_ACCURACY_DROP = 0.05
DUPLICATE_FRACTION_THRESHOLD = 0.02
DUPLICATE_ACCURACY_DROP = 0.02
MAX_UNCONFIRMED_SUSPECTS = 3

TOOL_DECLARATIONS = [
    {
        "name": "scan_suspect_columns",
        "description": "Scan feature columns for signs of target leakage.",
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["csv_path", "target"],
        },
    },
    {
        "name": "run_training",
        "description": "Retrain without one feature column to measure its effect.",
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "drop": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["csv_path", "drop", "target"],
        },
    },
    {
        "name": "check_split_duplicates",
        "description": (
            "Measure test rows duplicated in training and accuracy after "
            "excluding them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["csv_path", "target"],
        },
    },
]


def _http_status(error):
    for value in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            status = int(value)
        except (TypeError, ValueError):
            continue
        return status if status in (429, 503) else None

    message = str(error)
    if re.search(
        r"(?i)(bad|invalid|unauthorized|forbidden).{0,20}(key|model)",
        message,
    ):
        return None
    match = re.search(r"\b(429|503)\b", message)
    return int(match.group(1)) if match else None


def _generate_with_fallback(generate, contents, config, primary_model, fallback_model):
    for retry in range(MAX_RETRIES + 1):
        try:
            return generate(
                model=primary_model, contents=contents, config=config
            )
        except Exception as error:
            if _http_status(error) not in (429, 503):
                raise
            if retry == MAX_RETRIES:
                break
            time.sleep(2**retry)

    return generate(model=fallback_model, contents=contents, config=config)


def _response_function_calls(response):
    calls = []
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.function_call:
                calls.append(part.function_call)
    return calls


def _json_report(response):
    text = response.text or ""
    try:
        report = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            report = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return report if isinstance(report, dict) else {}


def _without_accuracy_claims(text):
    return re.sub(r"(?<!\w)\d+(?:\.\d+)?\s*%?", "[numeric value]", str(text))


def _tool_cache_key(name, arguments):
    argument_names = {
        "scan_suspect_columns": ("csv_path", "target"),
        "run_training": ("csv_path", "drop", "target"),
        "check_split_duplicates": ("csv_path", "target"),
    }.get(name, tuple(sorted(arguments)))
    normalized = {key: arguments.get(key) for key in argument_names}
    return json.dumps([name, normalized], sort_keys=True, separators=(",", ":"))


def _write_report(csv_path, report):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{Path(csv_path).name}.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit an ML result for possible target leakage."
    )
    parser.add_argument("csv_path")
    parser.add_argument("--target", default="Churn")
    args = parser.parse_args(argv)

    load_dotenv()
    neatlogs.init(
        api_key=os.environ["NEATLOGS_API_KEY"],
        workflow_name="leakhound-audit",
    )

    audit_span = None
    audit_span_entered = False
    try:
        audit_span = neatlogs.trace("leakhound-audit", kind="WORKFLOW")
        audit_span.__enter__()
        audit_span_entered = True
        client = neatlogs.wrap(
            genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        )
        primary_model = os.environ["GEMINI_MODEL"]
        fallback_model = os.environ["GEMINI_FALLBACK"]

        @neatlogs.span(kind="AGENT", name="gemini_model_call")
        def generate(model, contents, config):
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

        @neatlogs.span(
            kind="TOOL", name="run_training", tool_name="run_training"
        )
        def training_tool(csv_path, drop=None, target="Churn"):
            if drop is None:
                return run_training(csv_path, target=target)
            return run_training(csv_path, drop=drop, target=target)

        @neatlogs.span(
            kind="TOOL",
            name="scan_suspect_columns",
            tool_name="scan_suspect_columns",
        )
        def scan_tool(csv_path, target):
            return scan_suspect_columns(csv_path, target)

        @neatlogs.span(
            kind="TOOL",
            name="check_split_duplicates",
            tool_name="check_split_duplicates",
        )
        def duplicate_tool(csv_path, target):
            return check_split_duplicates(csv_path, target)

        @neatlogs.span(kind="TOOL", tool_name="cached_tool_call")
        def cached_tool_result(result):
            return {
                "result": result,
                "note": "This exact tool call was already done; returning the cached result.",
            }

        baseline = training_tool(args.csv_path, target=args.target)
        tool_results = [{"tool": "run_training", "result": baseline}]
        candidates = {}
        tested_columns = set()
        scan_completed = False
        call_cache = {}
        tool_calls_used = 1
        checks_run = []
        verdict = None
        suspect_column = None
        baseline_accuracy = (
            baseline.get("accuracy")
            if isinstance(baseline, dict) and baseline.get("ok")
            else None
        )
        history = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Audit the reported model accuracy for target leakage. "
                            f"CSV path: {args.csv_path!r}; target: {args.target!r}. "
                            "The code reproduced the baseline training result; "
                            f"result: {json.dumps(baseline, default=str)}. "
                            "Choose exactly one available tool per step. First "
                            "check for train/test duplicate rows and scan for "
                            "suspect columns, then test promising columns by "
                            "retraining without them. Accuracy drops are measured "
                            "in absolute points; do not conclude leak_confirmed "
                            "unless a tested feature lowers accuracy by at least "
                            "0.05. Do not repeat calls; if a prior result is "
                            "reported, choose another suspect. Stop investigating "
                            "after three tested suspects without a 0.05 drop. "
                            "If evidence is incomplete or unclear, use needs_human."
                        )
                    )
                ],
            )
        ]

        function_config = types.GenerateContentConfig(
            tools=[
                types.Tool(function_declarations=TOOL_DECLARATIONS)
            ],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY"
                )
            ),
        )

        for _ in range(MAX_TOOL_STEPS):
            response = _generate_with_fallback(
                generate,
                history,
                function_config,
                primary_model,
                fallback_model,
            )
            calls = _response_function_calls(response)
            if len(calls) != 1:
                history.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text="Return exactly one function call."
                            )
                        ],
                    )
                )
                continue

            call = calls[0]
            call_args = call.args if isinstance(call.args, dict) else {}
            history.append(response.candidates[0].content)
            cache_key = _tool_cache_key(call.name, call_args)
            if cache_key in call_cache:
                result = call_cache[cache_key]
                response_result = cached_tool_result(result)
            else:
                tool_calls_used += 1
                if call.name == "scan_suspect_columns":
                    if (
                        call_args.get("csv_path") != args.csv_path
                        or call_args.get("target") != args.target
                    ):
                        result = {
                            "ok": False,
                            "error": "Tool arguments do not match the audit input.",
                        }
                    else:
                        result = scan_tool(args.csv_path, args.target)
                        if "single-column scan" not in checks_run:
                            checks_run.append("single-column scan")
                        scan_completed = True
                        if isinstance(result, list):
                            candidates = {
                                item["column"]: item
                                for item in result
                                if isinstance(item, dict)
                                and isinstance(item.get("column"), str)
                            }
                    tool_results.append({"tool": call.name, "result": result})
                elif call.name == "run_training":
                    column = call_args.get("drop")
                    if (
                        call_args.get("csv_path") != args.csv_path
                        or call_args.get("target") != args.target
                    ):
                        result = {
                            "ok": False,
                            "error": "Tool arguments do not match the audit input.",
                        }
                    elif not isinstance(column, str) or column not in candidates:
                        result = {
                            "ok": False,
                            "error": "Scan for a candidate column before testing it.",
                        }
                    else:
                        result = training_tool(
                            args.csv_path, drop=column, target=args.target
                        )
                        if isinstance(result, dict) and result.get("ok"):
                            tested_columns.add(column)
                            if "drop-and-retrain" not in checks_run:
                                checks_run.append("drop-and-retrain")
                            accuracy = result.get("accuracy")
                            if (
                                isinstance(baseline_accuracy, (int, float))
                                and isinstance(accuracy, (int, float))
                                and baseline_accuracy - accuracy
                                >= LEAK_ACCURACY_DROP
                            ):
                                verdict = "leak_confirmed"
                                suspect_column = column
                    tool_results.append({"tool": call.name, "result": result})
                elif call.name == "check_split_duplicates":
                    if (
                        call_args.get("csv_path") != args.csv_path
                        or call_args.get("target") != args.target
                    ):
                        result = {
                            "ok": False,
                            "error": "Tool arguments do not match the audit input.",
                        }
                    else:
                        result = {
                            "ok": True,
                            **duplicate_tool(args.csv_path, args.target),
                        }
                        if "split-duplicate check" not in checks_run:
                            checks_run.append("split-duplicate check")
                        fraction = result["duplicate_fraction_of_test"]
                        accuracy_drop = result["accuracy_drop"]
                        if (
                            fraction > DUPLICATE_FRACTION_THRESHOLD
                            and accuracy_drop >= DUPLICATE_ACCURACY_DROP
                        ):
                            verdict = "duplicate_leak_confirmed"
                    tool_results.append({"tool": call.name, "result": result})
                else:
                    result = {"ok": False, "error": "Unknown tool."}
                    tool_results.append(
                        {"tool": str(call.name), "result": result}
                    )
                call_cache[cache_key] = result
                response_result = {"result": result}

            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name,
                            response=response_result,
                        )
                    ],
                )
            )
            if verdict in ("leak_confirmed", "duplicate_leak_confirmed"):
                break

            successful_no_drop_tests = {
                entry["result"]["dropped"]
                for entry in tool_results
                if entry.get("tool") == "run_training"
                and isinstance(entry.get("result"), dict)
                and entry["result"].get("ok")
                and entry["result"].get("dropped")
                and isinstance(baseline_accuracy, (int, float))
                and isinstance(entry["result"].get("accuracy"), (int, float))
                and baseline_accuracy - entry["result"]["accuracy"]
                < LEAK_ACCURACY_DROP
            }
            if len(successful_no_drop_tests) >= MAX_UNCONFIRMED_SUSPECTS:
                verdict = "no_column_leak_found"
                break
            if (
                isinstance(baseline_accuracy, (int, float))
                and scan_completed
                and not candidates
            ):
                verdict = "no_column_leak_found"
                break
            if (
                isinstance(baseline_accuracy, (int, float))
                and scan_completed
                and set(candidates).issubset(tested_columns)
            ):
                verdict = "no_column_leak_found"
                break

        final_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "verdict": {
                        "type": "STRING",
                        "enum": [
                            "leak_confirmed",
                            "duplicate_leak_confirmed",
                            "no_column_leak_found",
                            "needs_human",
                        ],
                    },
                    "suspect_column": {"type": "STRING", "nullable": True},
                    "evidence": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                },
                "required": [
                    "verdict",
                    "suspect_column",
                    "evidence",
                    "explanation",
                ],
            },
        )
        final_prompt = (
            "Write the final JSON report with keys verdict, suspect_column, "
            "evidence, and explanation. Use only verdicts leak_confirmed, "
            "duplicate_leak_confirmed, no_column_leak_found, or needs_human. "
            "State only what the tool "
            "results prove. Mark any claim about timing or cause as 'likely'; "
            "do not state timing or cause as fact. If scan evidence shows a "
            "column is missing for every No-target row and its removal lowers "
            "accuracy, say so explicitly. leak_confirmed is permitted only "
            "when code-computed retraining lowers baseline accuracy by at least "
            "0.05. duplicate_leak_confirmed is permitted only when more than "
            "2% of test rows duplicate training rows and excluding those rows "
            "lowers accuracy by at least 0.02. If evidence is unclear, use "
            "needs_human. Accuracy values "
            "will be inserted by code. Consider these tool results:\n"
            f"{json.dumps(tool_results, default=str)}"
        )
        final_response = _generate_with_fallback(
            generate,
            [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=final_prompt)],
                )
            ],
            final_config,
            primary_model,
            fallback_model,
        )
        model_report = _json_report(final_response)

        drop_results = {}
        for entry in tool_results:
            result = entry.get("result")
            dropped = result.get("dropped") if isinstance(result, dict) else None
            if (
                entry.get("tool") == "run_training"
                and dropped
                and result.get("ok")
                and isinstance(result.get("accuracy"), (int, float))
            ):
                drop_results[dropped] = result["accuracy"]

        if verdict is None:
            verdict = "needs_human"
        if verdict != "leak_confirmed" and not suspect_column:
            suggested_column = model_report.get("suspect_column")
            suspect_column = (
                suggested_column
                if isinstance(suggested_column, str)
                and suggested_column in candidates
                else None
            )

        evidence = []
        if baseline_accuracy is not None:
            evidence.append(
                {
                    "tool": "run_training",
                    "accuracy": baseline_accuracy,
                    "dropped": None,
                }
            )
        for entry in tool_results:
            result = entry.get("result")
            if entry.get("tool") == "scan_suspect_columns" and isinstance(
                result, list
            ):
                evidence.extend(
                    {
                        "tool": "scan_suspect_columns",
                        "column": item.get("column"),
                        "single_column_accuracy": item.get(
                            "single_column_accuracy"
                        ),
                        "missing_rate_by_class": item.get(
                            "missing_rate_by_class"
                        ),
                    }
                    for item in result
                    if isinstance(item, dict)
                )
            elif (
                entry.get("tool") == "run_training"
                and isinstance(result, dict)
                and result.get("dropped")
                and result.get("ok")
                and isinstance(result.get("accuracy"), (int, float))
            ):
                evidence.append(
                    {
                        "tool": "run_training",
                        "accuracy": result["accuracy"],
                        "dropped": result["dropped"],
                    }
                )
            elif entry.get("tool") == "check_split_duplicates" and isinstance(
                result, dict
            ) and result.get("ok"):
                evidence.append(
                    {
                        "tool": "check_split_duplicates",
                        "duplicate_count": result["duplicate_count"],
                        "duplicate_fraction_of_test": result[
                            "duplicate_fraction_of_test"
                        ],
                        "baseline_accuracy": result["baseline_accuracy"],
                        "honest_accuracy": result["honest_accuracy"],
                        "accuracy_drop": result["accuracy_drop"],
                        "interpretation": "measured on this split",
                    }
                )
            elif isinstance(result, dict) and not result.get("ok"):
                evidence.append(
                    {"tool": entry.get("tool"), "error": result.get("error")}
                )

        duplicate_result = next(
            (
                entry["result"]
                for entry in tool_results
                if entry.get("tool") == "check_split_duplicates"
                and isinstance(entry.get("result"), dict)
                and entry["result"].get("ok")
            ),
            None,
        )
        if verdict == "leak_confirmed":
            honest_accuracy = drop_results[suspect_column]
            accuracy_drop_points = round(
                (baseline_accuracy - honest_accuracy) * 100, 4
            )
            scan = candidates.get(suspect_column, {})
            missing_by_class = scan.get("missing_rate_by_class", {})
            if isinstance(missing_by_class, dict) and missing_by_class.get("No") == 1:
                explanation = (
                    f"Column {suspect_column} is empty for every customer who "
                    f"stayed, and dropping it lowers accuracy by "
                    f"{accuracy_drop_points:g} points. The column's relationship "
                    "to the outcome is established by these checks; it is likely "
                    "to reflect target-related information, but when or how it "
                    "was recorded is not established."
                )
            else:
                explanation = (
                    f"Dropping column {suspect_column} lowers accuracy by "
                    f"{accuracy_drop_points:g} points in this test, proving that "
                    "it materially affects the measured result. It is likely "
                    "that the column carries target-related information, but "
                    "these checks do not establish its cause or timing."
                )
        elif verdict == "duplicate_leak_confirmed":
            honest_accuracy = duplicate_result["honest_accuracy"]
            accuracy_drop_points = round(
                (baseline_accuracy - honest_accuracy) * 100, 4
            )
            explanation = (
                f"{duplicate_result['duplicate_count']} test rows "
                f"({duplicate_result['duplicate_fraction_of_test']:.1%}) have "
                "feature rows also present in training. Evaluating only test "
                f"rows not found in training lowers accuracy from "
                f"{baseline_accuracy:.4f} to {honest_accuracy:.4f}; this "
                "measured split meets the duplicate-leak threshold."
            )
        elif verdict == "no_column_leak_found":
            tested_count = len(drop_results)
            explanation = (
                f"No tested column removal lowered accuracy by at least "
                f"{LEAK_ACCURACY_DROP * 100:g} points; {tested_count} column"
                f"{'' if tested_count == 1 else 's'} were checked. This does "
                "not rule out leakage in untested columns or other forms of "
                "leakage."
            )
            honest_accuracy = (
                min(drop_results.values()) if drop_results else None
            )
            accuracy_drop_points = (
                round((baseline_accuracy - honest_accuracy) * 100, 4)
                if baseline_accuracy is not None and honest_accuracy is not None
                else None
            )
        else:
            explanation = (
                "The available checks are inconclusive. Further validation is "
                "likely needed to determine whether any untested column or "
                "other mechanism affects the reported result."
            )
            honest_accuracy = min(drop_results.values()) if drop_results else None
            accuracy_drop_points = (
                round((baseline_accuracy - honest_accuracy) * 100, 4)
                if baseline_accuracy is not None and honest_accuracy is not None
                else None
            )
        largest_drop_points = (
            round((baseline_accuracy - min(drop_results.values())) * 100, 2)
            if drop_results and baseline_accuracy is not None
            else None
        )
        if verdict not in ("leak_confirmed", "duplicate_leak_confirmed"):
            honest_accuracy = None
            accuracy_drop_points = None
        report = {
            "verdict": verdict,
            "suspect_column": suspect_column,
            "evidence": evidence,
            "explanation": explanation,
            "reported_accuracy": baseline_accuracy,
            "honest_accuracy": honest_accuracy,
            "accuracy_drop_points": accuracy_drop_points,
            "tool_calls_used": tool_calls_used,
            "checks_run": checks_run,
            "largest_drop_points": largest_drop_points,
        }
        return _write_report(args.csv_path, report)
    except Exception as error:
        return _write_report(args.csv_path, {
            "verdict": "needs_human",
            "suspect_column": None,
            "explanation": "The audit could not finish, so no conclusion is given.",
            "error": f"{type(error).__name__}: {str(error)[:300]}",
            "error_status": _http_status(error),
            "reported_accuracy": None,
            "honest_accuracy": None,
            "accuracy_drop_points": None,
        })
    finally:

        try:
            if audit_span_entered:
                audit_span.__exit__(*sys.exc_info())
        finally:
            neatlogs.flush()
            neatlogs.shutdown()


if __name__ == "__main__":
    main()
