"""Follow-up measures. Source labels, trial outcomes and gate results stay separate."""

import json
from pathlib import Path
import re
import tomllib

import pandas as pd

from task_verifier_features import DELIVERABLES, as_list, grade_file_url

STRICT_REPORT = re.compile(
    r"(?:^|[.;!?]\s+)(?:please\s+)?(?:write|submit|provide|deliver|save|include|produce|return)\s+"
    r"(?:(?:a|an|the|your|final|concise|brief|written|technical|detailed|short|text|markdown)\s+){0,5}"
    r"(?:answer|report|write[- ]up|summary|explanation|analysis)\b"
    r"|\b(?:your (?:final )?(?:answer|report|response) must|save (?:your|the) (?:answer|report|write[- ]up))\b",
    re.I,
)
ASSET_REPORT = re.compile(r"(?:answer|report|writeup|summary)\.(?:md|txt|pdf)\b", re.I)
DURATION_BANDS = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 3}
ORDINAL = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
IMPLEMENTATIONS = {
    "Executable tests": "Run the language's test runner on the submitted code or installed program; turn process status or a parsed test report into a gate.",
    "State / artifact check": "Read the produced file, return value, or final state and compare its existence, shape, contents, or behavior with expectations.",
    "Performance / numeric measurement": "Collect a metric and compare it with a threshold, tolerance, or reference. Many workloads check correctness before timing.",
    "Mutation check": "Run the candidate tests against the original implementation and intentionally changed implementations. A useful test passes on the original and detects the specified mutants.",
    "LLM / rubric judging": "Evaluate individual criteria against the answer or patch, normalize verdicts, and then apply mandatory and optional rubric rules.",
    "Manifest / scope check": "Read the candidate's manifest or diff and determine whether its listed tests or changed files match the task's allowed scope.",
    "Build or install gate": "Run an install or build step in a defined workspace and stop or fail grading if the submission cannot be prepared.",
    "Regression groups": "Track named fix and regression tests separately and require the configured groups to pass. Missing/error results are handled by the specific runner.",
}


def deliverable_audit(evidence):
    rows = []
    for (benchmark, task_id), item in evidence.items():
        clauses = item["clauses"]
        broad = item["deliverables"]["signals"]["Written answer / report"]
        strict = [c for c in clauses if STRICT_REPORT.search(c["text"]) or (
            ASSET_REPORT.search(c["text"]) and re.match(r"^(?:write|save|provide|submit|produce)\b", c["text"], re.I)
        )]
        terms = tuple(sorted({match.group().lower() for c in broad for match in re.finditer(
            DELIVERABLES["Written answer / report"], c["text"], re.I
        )}))
        rows.append({
            "benchmark": benchmark, "task_id": task_id, "broad": bool(broad), "strict": bool(strict),
            "atlas_answer_track": benchmark == "SWE Atlas Q&A",
            "broad_terms": terms,
            "old_only_example": broad[0]["text"] if broad and not strict else "",
        })
    return pd.DataFrame(rows)


def read_difficulty(sources, raw):
    """Read publisher labels. Preserve the measurement scale per benchmark."""
    rows = []
    for source in sources:
        if source["cohort"] == "subset":
            continue
        name, kind = source["name"], source["format"]
        path = raw / source["location"]
        if kind == "harbor":
            for file in sorted(path.glob("*/task.toml")):
                meta = tomllib.loads(file.read_text()).get("metadata", {})
                label = str(meta.get("difficulty", "")).lower()
                if label in ORDINAL:
                    rows.append((name, file.parent.name, "Author difficulty", label, ORDINAL[label]))
                try:
                    hours = float(meta.get("expert_time_estimate_hours", ""))
                except (TypeError, ValueError):
                    continue
                if hours > 0:
                    rows.append((name, file.parent.name, "Expert estimated hours", str(hours), hours))
        elif kind == "verified":
            for item in pd.read_parquet(path, columns=["instance_id", "difficulty"]).to_dict("records"):
                label = item["difficulty"]
                if label in DURATION_BANDS:
                    rows.append((name, item["instance_id"], "Author fix-time band", label, DURATION_BANDS[label]))
        elif name == "SWE-bench Pro V2":
            for item in pd.read_parquet(path, columns=["instance_id", "hard"]).to_dict("records"):
                label = "Hard-51" if item["hard"] else "Remaining V2"
                rows.append((name, item["instance_id"], "Publisher model-selected cohort", label, int(item["hard"])))
    return pd.DataFrame(rows, columns=["benchmark", "task_id", "scale", "label", "value"])


def difficulty_by_signal(difficulty, features, field):
    joined = difficulty.merge(features[["benchmark", "task_id", field]], on=["benchmark", "task_id"], validate="many_to_one")
    rows = []
    for (benchmark, scale), block in joined.groupby(["benchmark", "scale"], sort=False):
        for signal in sorted({item for values in block[field] for item in values}):
            present = block[field].map(lambda values: signal in values)
            for description, partition in (("Detected", block[present]), ("Not detected", block[~present])):
                if partition.empty:
                    continue
                values = partition["value"]
                for label, count in partition["label"].value_counts().items():
                    rows.append({
                        "benchmark": benchmark, "scale": scale, "signal": signal,
                        "group": description, "label": label, "n": count,
                        "group_n": len(partition), "share": count / len(partition),
                        "median_estimated_hours": values.median() if scale == "Expert estimated hours" else None,
                    })
    return pd.DataFrame(rows)


def code_examples(evidence, per_benchmark=1):
    rows, seen = [], {}
    for (benchmark, task_id), item in evidence.items():
        for label, location in item["checks"].items():
            if not re.search(r":\d+$", location):
                continue
            key = label, benchmark
            if seen.get(key, 0) >= per_benchmark:
                continue
            seen[key] = seen.get(key, 0) + 1
            path, _, line = location.rpartition(":")
            local = next((f for f in item["files"] if f["path"] == path), {})
            url = local.get("url") or grade_file_url(benchmark, task_id, path)
            rows.append({"check": label, "benchmark": benchmark, "task_id": task_id, "source_location": location,
                         "source_url": f"{url}#L{line}" if url else "",
                         "value": "Static signal; inspect source to confirm"})
    return pd.DataFrame(rows)


def read_outcomes(path):
    """Normalize measured dispositions while retaining how each source scored them."""
    rows = []
    raw_files = []
    for location in sorted((path / "deep" / "files").glob("*/*/reward.json")):
        document = json.loads(location.read_text())
        raw_files.append({
            "benchmark": "DeepSWE 1.1", "config": location.parent.parent.name,
            "trial": location.parent.name, "path": str(location.relative_to(path)),
            **{k: v for k, v in document.items() if isinstance(v, (str, int, float, bool))},
        })

    for item in json.loads((path / "deep" / "trials.json").read_text())["rows"]:
        if item.get("source") != "deep-swe" or item.get("eval_scope") != "full":
            continue
        value = item.get("score_value")
        if not item.get("included_in_score") or value not in (0, 1):
            value = None
        rows.append({
            "benchmark": "DeepSWE 1.1", "task_id": item["task_name"], "config": item["config"],
            "trial": item["trial_name"], "pass": value, "publisher_error": item.get("errored"),
            "artifact": "Publisher trial index",
            **{k: item.get(k) for k in ("f2p_total", "f2p_passed", "p2p_total", "p2p_passed")},
        })

    for location in sorted((path / "verified").glob("*/per_instance_details.json")):
        for task, item in json.loads(location.read_text()).items():
            value = item.get("resolved")
            rows.append({"benchmark": "SWE-bench Verified", "task_id": task, "config": location.parent.name,
                         "trial": task, "pass": int(value) if isinstance(value, bool) else None,
                         "artifact": "Submission per-instance result"})

    for location in sorted((path / "pro-v1" / "traj").glob("*/eval_results.json")):
        for task, value in json.loads(location.read_text()).items():
            if isinstance(value, bool):
                rows.append({"benchmark": "SWE-bench Pro V1", "task_id": task, "config": location.parent.name,
                             "trial": task, "pass": int(value), "artifact": "Publisher per-instance result"})

    for location in sorted((path / "terminal").glob("*/*.json")):
        if location.name == "board.json":
            continue
        document = json.loads(location.read_text())
        version = location.parent.name
        for item in document["trials"]:
            reward = item.get("reward")
            if item.get("is_scored") is False:
                value = None
            elif reward in (0, 1):
                value = int(reward)
            elif reward is None and item.get("error_type"):
                value = 0  # The Terminal-Bench leaderboard counts errored trials as zero.
            else:
                value = None
            rows.append({
                "benchmark": f"Terminal-Bench {version}",
                "task_id": item["task_name"].rsplit("/", 1)[-1],
                "config": document["config"] + f" [{location.stem[:8]}]",
                "trial": item["id"], "pass": value, "recorded_reward": reward,
                "publisher_error": bool(item.get("error_type")),
                "artifact": "Public leaderboard trial receipt", "receipt_url": document["source"],
            })

    marathon = path / "marathon" / "receipts.json"
    if marathon.exists():
        receipts = json.loads(marathon.read_text())
        for item in receipts["trials"]:
            value = item.get("reward")
            label = str(item.get("modelVariant") or item.get("model")) + " / " + str(item.get("agent"))
            rows.append({
                "benchmark": "SWE-Marathon 1.1", "task_id": item["task_id"],
                "config": label, "trial": item["id"], "pass": value if value in (0, 1) else None,
                "publisher_error": item.get("status") == "error",
                "artifact": "Publisher dashboard trial receipt",
                "partial": item.get("partial"), "receipt_url": receipts["source"],
            })

    atlas_file = path / "atlas" / "results" / "per_task_results.csv"
    if atlas_file.exists():
        for item in pd.read_csv(atlas_file).to_dict("records"):
            task = str(item["task_id"]).removeprefix("task-").removeprefix("qa-")
            value = str(item["resolved"]).lower()
            rows.append({"benchmark": "SWE Atlas Q&A", "task_id": task, "config": "big-pickle / Q&A pre-v1.1",
                         "trial": task, "pass": {"true": 1, "false": 0, "1": 1, "0": 0}.get(value),
                         "artifact": "Community outcome CSV", "rubric_aggregate": item["agg_score"],
                         "rubrics_passed": item["rubrics_passed"], "rubrics_total": item["rubrics_total"]})
    return pd.DataFrame(rows), pd.DataFrame(raw_files)


def comparison_by_check(outcomes, features, benchmark, config, min_tasks=5):
    scoped = outcomes.loc[outcomes["benchmark"].eq(benchmark) & outcomes["config"].eq(config)]
    by_task = scoped.groupby(["benchmark", "task_id"], as_index=False).agg(pass_rate=("pass", "mean"), scored_trials=("pass", "count"))
    merged = by_task.merge(features[["benchmark", "task_id", "check_signals"]], on=["benchmark", "task_id"], validate="one_to_one")
    merged = merged.loc[merged["scored_trials"].gt(0)]
    rows = []
    for tag in IMPLEMENTATIONS:
        flag = merged["check_signals"].map(lambda labels: tag in labels)
        a, b = merged[flag], merged[~flag]
        rows.append({
            "signal": tag, "detected_tasks": len(a), "other_tasks": len(b),
            "detected_pass_rate": a["pass_rate"].mean(),
            "other_pass_rate": b["pass_rate"].mean(),
            "delta_pp": 100 * (a["pass_rate"].mean() - b["pass_rate"].mean())
                if min(len(a), len(b)) >= min_tasks else None,
        })
    return pd.DataFrame(rows).sort_values("delta_pp", na_position="last")


def paired_regression_gates(outcomes, benchmark="DeepSWE 1.1"):
    deep = outcomes.loc[outcomes["benchmark"].eq(benchmark) & outcomes["pass"].notna()]
    deep = deep.dropna(subset=["f2p_total", "f2p_passed", "p2p_total", "p2p_passed"])
    # Compare the same trials only when both test groups are nonempty.
    deep = deep.loc[deep["f2p_total"].gt(0) & deep["p2p_total"].gt(0)].copy()
    deep["fix_gate_pass"] = deep["f2p_passed"].eq(deep["f2p_total"])
    deep["regression_gate_pass"] = deep["p2p_passed"].eq(deep["p2p_total"])
    return deep.groupby("config", sort=False).agg(
        n=("trial", "size"), tasks=("task_id", "nunique"),
        fix_gate=("fix_gate_pass", "mean"), regression_gate=("regression_gate_pass", "mean"),
        trial_pass=("pass", "mean"),
    ).sort_index()


def read_atlas_votes(path, rubric_csv):
    """Join old community log vote IDs only where they match source criterion IDs."""
    rubric_id = {}
    for row in pd.read_csv(rubric_csv, usecols=["task_id", "rubric"]).to_dict("records"):
        for criterion in json.loads(row["rubric"]):
            annotation = criterion.get("annotations", {})
            rubric_id[row["task_id"], str(criterion["id"])] = (
                annotation.get("importance"), annotation.get("type"),
            )
    votes = []
    pattern = re.compile(r"^\s*Rubric ([\w-]+):\s*([01])\s*\([^\n]*?raw\s*=\s*([01])\b", re.M)
    for location in sorted((path / "atlas" / "results" / "verifier_logs").glob("*.txt")):
        task = location.stem.removeprefix("task-")
        for match in pattern.finditer(location.read_text()):
            criterion_id, effective, raw = match.groups()
            importance, kind = rubric_id.get((task, criterion_id), (None, None))
            votes.append({
                "task_id": task, "criterion_id": criterion_id, "effective": int(effective), "raw": int(raw),
                "importance_in_bundle": importance, "type_in_bundle": kind,
                "bundle_id_matched": (task, criterion_id) in rubric_id,
                "path": str(location.relative_to(path)),
            })
    return pd.DataFrame(votes)


def read_pro_test_reports(outcome_root, raw):
    """Reconstruct the publisher's V1 set membership rule from original reports."""
    task_file = raw / "swe-bench-pro/data/v1/test-00000-of-00001.parquet"
    records = pd.read_parquet(task_file, columns=["instance_id", "fail_to_pass", "pass_to_pass"]).set_index("instance_id")
    rows = []
    for path in sorted((outcome_root / "pro-v1/s3").glob("*/reports/*.json")):
        task = path.stem
        if task not in records.index:
            continue
        item = records.loc[task]
        f2p, p2p = as_list(item["fail_to_pass"]), as_list(item["pass_to_pass"])
        report = json.loads(path.read_text())
        tests = report.get("tests")
        if f2p is None or p2p is None or not isinstance(tests, list):
            continue
        passed = {entry["name"] for entry in tests if entry.get("status") == "PASSED"}
        f2p, p2p = set(f2p), set(p2p)
        rows.append({
            "benchmark": "SWE-bench Pro V1", "task_id": task, "config": path.parent.parent.name,
            "trial": task, "pass": int((f2p | p2p) <= passed),
            "artifact": "Publisher test report, verdict recomputed",
            "f2p_total": len(f2p), "f2p_passed": len(f2p & passed),
            "p2p_total": len(p2p), "p2p_passed": len(p2p & passed),
            "path": str(path.relative_to(outcome_root)),
        })
    return pd.DataFrame(rows)
