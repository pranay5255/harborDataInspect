"""Static features for public benchmark tasks. Does not execute verifier code."""

from ast import literal_eval
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path, PurePosixPath
import re
import tomllib

import pandas as pd

from fetch_verifiers import SOURCES as VERIFIER_SOURCES

INTENTS = {
    "Bug fix": r"\b(?:fix(?:ed|es|ing)?|regression|broken|incorrect|crash|bug)\b",
    "Feature or behavior change": r"\b(?:add support|implement|introduce|extend|enable|new feature)\b",
    "Refactor or migrate": r"\b(?:refactor|restructur\w*|migrat\w*|port\b|moderniz\w*|extract (?:a |the )?(?:class|function|module))\b",
    "Test writing": r"\b(?:write|add|extend|create|implement)\s+(?:new\s+|unit\s+|regression\s+|integration\s+)*(?:tests?|test cases|test coverage)\b",
    "Explain or investigate": r"\b(?:explain|investigate|trace|analy[sz]e|describe|document|identify|diagnose)\b",
    "Build from scratch": r"\b(?:from scratch|reimplement|recreate|build (?:a|an) (?:new|complete)|empty (?:repository|workspace))\b",
    "Performance or scale": r"\b(?:optim[izs]\w*|latency|throughput|speedup|memory footprint|benchmark|scalability)\b",
    "Build or environment": r"\b(?:configure|dependency|install|setup\.sh|packag\w*|build system|container|docker|deploy)\b",
    "Data processing": r"\b(?:transform(?:ation)?|dataset|pipeline|parse|convert|ingest|schema|aggregate|CSV|JSONL)\b",
    "Compatibility or preservation": r"\b(?:compatib\w*|preserv\w*|backwards?|retain|unchanged|same behavior|existing tests)\b",
    "Reliability or recovery": r"\b(?:recovery|retry|resilien\w*|reliab\w*|fallback|rollback|recover|fault toleran\w*)\b",
    "UI or frontend": r"\b(?:user interface|frontend|front-end|browser|layout|CSS|accessibility|render\w*)\b",
}

MUST = re.compile(r"\b(?:must|shall|required|need to|should|has to|expected to|ensure|make sure)\b", re.I)
FORBID = re.compile(r"\b(?:must not|should not|do not|don't|never|cannot|may not|without changing|no changes?)\b", re.I)
IMPERATIVE = re.compile(
    r"^(?:please\s+)?(?:add|allow|build|change|compare|create|deliver|describe|document|ensure|"
    r"explain|fix|generate|implement|include|keep|maintain|make|move|output|preserve|produce|"
    r"provide|refactor|remove|replace|report|return|save|support|test|update|use|write)\b", re.I,
)
DELIVERABLES = {
    "Patch / code changes": r"\b(?:submit|produce|apply|create|make)\s+(?:a\s+|the\s+)?(?:patch|diff|code changes)\b",
    "Test suite": r"\b(?:write|add|create|extend|implement|submit)\s+(?:\w+\s+){0,2}(?:tests|test suite|test cases)\b",
    "Written answer / report": r"\b(?:answer|report|document|explain|provide an explanation|describe|write[- ]up)\b",
    "File or structured output": r"\b(?:write|save|output|emit|generate|create|produce)\s+(?:\w+\s+){0,4}(?:file|artifact|json|csv|output|report)\b",
    "Buildable project / CLI": r"\b(?:setup\.sh|entrypoint|command.line tool|installable|working (?:app|application|repository))\b",
    "Performance target": r"\b(?:at least|at most|under|within|less than|no more than)\s+\d+(?:\.\d+)?\s*(?:%|ms\b|seconds?\b|MB\b|GB\b|x\b)",
}
FILE_REF = re.compile(r"(?<!\w)(?:\.{0,2}/[\w.-]+(?:/[\w.-]+)+|[\w./-]+\.(?:py|js|ts|rs|go|java|jsonl?|csv|toml|yaml|yml|md|txt|sh|tsx|html|css|cpp|rb|php))\b", re.I)
FENCE = re.compile(r"^\s*(?:\x60{3,}|~{3,})")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
LIST_MARKER = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
REWARD_CHANNEL = re.compile(r"\breward(?:[-_]details)?\.(?:txt|json)\b", re.I)
TEST_NAME = re.compile(
    r"\b(?:def\s+(test\w*)|func\s+(Test\w*)|(?:public\s+)?void\s+(test\w*)|"
    r"function\s+(test\w*)|(?:it|test|describe)\s*\(\s*['\x60\x22]([^'\x60\x22\n]{4,160}))",
    re.I,
)

CHECK_SYMBOL = re.compile(
    r"\b(?:def|func|function)\s+([\w]*(?:check|verify|validate|assert|grade|score)[\w]*)\s*(?=\()",
    re.I,
)
CHECK_RULES = {
    "Executable tests": r"\b(?:pytest|unittest|go test|cargo test|npm test|jest|vitest|ctest|dotnet test|mvn test|gradlew test|testthat|rspec)\b",
    "State / artifact check": r"\b(?:os\.path\.(?:exists|isfile)|assert (?:os\.|.*\.exists\()|filecmp|sha256sum|compare_files|subprocess\.check_output)\b|\[\[?\s+-[fde]\s",
    "Performance / numeric measurement": r"\b(?:perf_counter|throughput|latency|elapsed_time|speedup|threshold|mean_squared_error|tolerance|allclose)\b",
    "Mutation check": r"\b(?:mutation[-_]check|mutation_report|mutant|mutants|mutated|mutation_score|run_mutation)\b",
    "LLM / rubric judging": r"\b(?:judge_client|llm_judge|rubric|rubrics|evaluate_answer|evaluate_rubric)\b",
    "Manifest / scope check": r"\b(?:manifest_check|validate_manifest|verify_manifest|manifest_validation|allowed_files|protected_files|forbidden_files)\b",
    "Build or install gate": r"\b(?:setup\.sh|pip install|npm (?:ci|run build)|cargo build|go build|make build|cmake --build)\b",
    "Regression groups": r"\b(?:FAIL_TO_PASS|PASS_TO_PASS|f2p_node_ids|p2p_node_ids)\b",
}

POLICIES = {
    "verified": ("All configured tests pass", "Apply the candidate patch and require all listed fix and regression tests to pass.", "https://openai.com/index/introducing-swe-bench-verified/"),
    "pro_v1": ("All configured tests pass", "Apply the candidate patch and require the configured test groups to pass.", "https://scale.com/research/swe_bench_pro"),
    "pro_v2": ("All configured tests pass", "Evaluate the submitted diff in a clean image and require the configured test groups to pass.", "https://labs.scale.com/blog/swe-bench-pro-v2"),
    "deep": ("Binary gate + test diagnostics", "Grade the submitted patch in the verifier environment against the configured test groups. Test fractions are diagnostics alongside the binary result.", "https://github.com/datacurve-ai/deep-swe"),
    "atlas_qa": ("Rubric gate", "An LLM scores the submitted answer against the rubric. In the pinned GitHub grader, every declared must-have must be scored and pass. See the task rubric and evaluate_answer.py for criteria and gate.", "https://labs.scale.com/leaderboard/sweatlas-qna"),
    "atlas_tw": ("Programmatic + rubric gates", "Check the manifest, execute mutation checks, and judge the authored tests against the rubric.", "https://labs.scale.com/leaderboard/sweatlas-tw"),
    "atlas_rf": ("Programmatic + rubric gates", "Check executable tests and mandatory rubric criteria for the candidate refactor.", "https://labs.scale.com/leaderboard/sweatlas-refactoring"),
    "e2e": ("Install + all tests pass", "Install the submission in a fresh verifier container and require every hidden test to pass.", "https://github.com/facebookresearch/E2E-SWE"),
}

STOP = set("and the that this from with into have must should please ensure write add return function file are for you your task code test tests use using required may its can alter value values expected result behavior existing".split())


def task_clauses(prompt, supplemental=""):
    """Extract candidate clauses with their role, origin and original line."""
    rows = []
    for origin, text in (("prompt", prompt), ("supplement", supplemental)):
        in_code = False
        section = ""
        for number, raw in enumerate(text.splitlines(), 1):
            if FENCE.match(raw):
                in_code = not in_code
                continue
            if in_code or not raw.strip():
                continue
            if raw.strip().startswith("#"):
                section = raw.lstrip(" #").strip()
                continue
            item = LIST_MARKER.sub("", raw.strip())
            for clause in SENTENCE_BOUNDARY.split(item):
                clause = clause.strip()
                if len(clause.split()) < 3:
                    continue
                role = (
                    "Prohibition" if FORBID.search(clause) else
                    "Explicit requirement" if MUST.search(clause) else
                    "Action request" if IMPERATIVE.match(clause) else
                    "Question" if clause.endswith("?") else
                    "Structured specification" if LIST_MARKER.match(raw) and re.search(
                        r"requirement|specification|behavior|deliverable|acceptance", section, re.I
                    ) else "Expected behavior" if re.search(r"expected(?: behavior| result| output)|actual vs. expected", section, re.I)
                    else "Context or specification"
                )
                rows.append({"origin": origin, "line": number, "role": role, "section": section, "text": clause})
    return rows


def task_intents(clauses):
    """Return multi-label prompt signals and the exact supporting clauses."""
    return {
        label: [clause for clause in clauses if re.search(pattern, clause["text"], re.I)]
        for label, pattern in INTENTS.items()
    }


def task_deliverables(clauses):
    """Find output-language signals and file/path mentions in task clauses."""
    return {
        "signals": {
            key: [clause for clause in clauses if re.search(pattern, clause["text"], re.I)]
            for key, pattern in DELIVERABLES.items()
        },
        "paths": sorted({p for clause in clauses for p in FILE_REF.findall(clause["text"])}),
    }


def verifier_inventory(files, config, structured_fields):
    """Summarize attached files, browsable text, and task-indexed data fields."""
    names = [row["path"] for row in files]
    return {
        "inventory_listed": bool(files),
        "listed_files": len(files) if files else None,
        "readable_text_files": sum(row["selected"] for row in files) if files else None,
        "entrypoint_listed": any(name.lower() in {"test.sh", "test.bat"} for name in names),
        "rubric_files": sum("rubric" in name.lower() for name in names) if files else None,
        "verifier_timeout_sec": config.get("timeout_sec"),
        "structured_fields": tuple(sorted(structured_fields)),
    }


def verifier_boundary(config, format_kind, listed_paths):
    """Resolve published verifier modes and retain packaging/transfer evidence."""
    mode = config.get("environment_mode", config.get("mode"))
    collected = config.get("collect")
    declared_artifacts = config.get("_artifacts")
    build_definition = any(
        PurePosixPath(p).name.lower() in {"dockerfile", "docker-compose.yaml", "docker-compose.yml"}
        for p in listed_paths
    )
    boundary = "Unspecified"
    basis = "No explicit task setting in local artifacts"
    if mode:
        boundary = {"separate": "Separate verifier", "shared": "Shared verifier"}.get(mode, str(mode))
        basis = "task.toml [verifier].environment_mode"
    elif "environment" in config:
        boundary, basis = "Separate verifier", "task.toml [verifier.environment]"
    elif format_kind == "deep":
        boundary, basis = "Separate verifier", "DeepSWE v1.1 source documentation"
    elif format_kind in {"pro_v2", "e2e"}:
        boundary, basis = "Fresh workspace", "Published evaluation protocol"
    elif config.get("_task_config"):
        boundary, basis = "Shared verifier", "Harbor default when the task omits an environment mode"
    return {
        "boundary": boundary, "basis": basis,
        "tests_build_definition": build_definition if listed_paths else None,
        "collect_rules": len(collected) if isinstance(collected, (list, dict)) else None,
        "declared_artifacts": len(declared_artifacts) if isinstance(declared_artifacts, list) else None,
    }


def verifier_checks(documents, groups, rubrics, format_kind):
    """Return check-type signals and example static locations."""
    flags = {}
    if groups["f2p"] is not None or groups["p2p"] is not None:
        flags["Regression groups"] = "Dataset test group fields"
    if rubrics:
        flags["LLM / rubric judging"] = "Task rubric criteria are available"
    for path, body in documents.items():
        if path.endswith((".patch", ".diff")) or "rubrics." in path.lower():
            continue
        # Track the start of each match as a source reference.
        for label, rule in CHECK_RULES.items():
            if label in flags:
                continue
            match = re.search(rule, body, re.I)
            if match:
                line = body.count("\n", 0, match.start()) + 1
                flags[label] = f"{path}:{line}"
    if format_kind in {"e2e", "atlas_tw", "atlas_rf", "verified", "pro_v1", "pro_v2", "deep"}:
        flags.setdefault("Executable tests", "Published benchmark protocol")
    return flags


def as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except ValueError:
            # Pro V1 uses Python literals for many test lists.
            try:
                value = literal_eval(value)
            except (SyntaxError, ValueError):
                return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else None


def grade_test_groups(record, config):
    """Report explicitly configured test groups and declared test case counts."""
    groups = {}
    for out, names in (("f2p", ("FAIL_TO_PASS", "fail_to_pass", "f2p")), ("p2p", ("PASS_TO_PASS", "pass_to_pass", "p2p"))):
        groups[out] = next((parsed for name in names if (parsed := as_list(record.get(name))) is not None), None)
    declared = config.get("test_case_count")
    groups["declared_cases"] = declared if isinstance(declared, int) else None
    return groups


def grade_rubrics(record, documents):
    """Read published rubric criteria, retaining their importance and polarity fields."""
    candidates = [(name, text) for name, text in documents.items() if re.search(r"rubric|criteria", name, re.I) and name.endswith(".json")]
    if isinstance(record.get("rubric"), str):
        candidates = [("Hugging Face rubric field", record["rubric"])]
    criteria = []
    for path, text in candidates:
        try:
            content = json.loads(text)
        except (TypeError, ValueError):
            continue
        def visit(item):
            if isinstance(item, list):
                for value in item:
                    visit(value)
            elif isinstance(item, dict):
                words = next((item.get(key) for key in ("title", "criterion", "description", "text", "prompt") if isinstance(item.get(key), str)), None)
                if words and not any(isinstance(item.get(key), (list, dict)) for key in ("rubrics", "criteria", "items")):
                    annotation = item.get("annotations")
                    tags = annotation if isinstance(annotation, dict) else {}
                    criteria.append({
                        "source": path,
                        "criterion": words,
                        "importance": str(tags.get("importance", item.get("importance", ""))),
                        "type": str(tags.get("type", item.get("type", ""))),
                        "weight": item.get("weight", tags.get("weight")),
                    })
                else:
                    for value in item.values():
                        if isinstance(value, (list, dict)):
                            visit(value)
        visit(content)
    return list({(row["criterion"], row["importance"], row["type"]): row for row in criteria}.values())


def grade_contract(format_kind, documents, config):
    """Expose publisher rules and observed reward file references independently."""
    channels = {}
    for path, body in documents.items():
        if path.endswith((".patch", ".diff")):
            continue
        for match in REWARD_CHANNEL.finditer(body):
            channel = match.group().lower()
            channels.setdefault(channel, f"{path}:{body.count(chr(10), 0, match.start()) + 1}")
    if format_kind in POLICIES:
        family, rule, source = POLICIES[format_kind]
        basis = "Published benchmark protocol"
    else:
        family = "Task-specific reward"
        rule = "Inspect the task's reward writer. A reward file reference alone cannot distinguish binary, graded, or gated partial credit."
        source = "https://docs.harborframework.com/tasks/verifier"
        basis = "Harbor output contract"
    grade_config = {}
    if "config.json" in documents:
        try:
            candidate = json.loads(documents["config.json"])
            grade_config = candidate.get("grade", {}) if isinstance(candidate, dict) else {}
        except ValueError:
            pass
    return {
        "declared_test_case_count": config.get("test_case_count"),
        "grading_family": family, "decision_rule": rule, "rule_source": source,
        "rule_basis": basis, "reward_channels": channels,
        "task_grade_settings": grade_config,
    }


ACTION_ROLES = {"Explicit requirement", "Action request", "Prohibition", "Structured specification", "Question", "Expected behavior", "Issue description candidate"}
LITERAL_REWARD = re.compile(r"\b(?:echo|printf)\s+['\x22]?([01](?:\.0)?)['\x22]?\s*>+[^\n]*\breward\.txt\b", re.I)
REWARD_FIELD = re.compile(r"['\x22](reward|score|pass_fraction|pass_rate|accuracy|resolved|tests_pass|tests_passed|must_have_pass|overall_pass|rubrics_agg_score|mutation_score|[^'\x22\n]{1,35}_pass_fraction)['\x22]\s*:")
TASK_FORMATS = {
    "SWE-bench Pro V1": "pro_v1", "SWE-bench Pro V2": "pro_v2",
    "SWE Atlas Test Writing": "atlas_tw", "SWE Atlas Refactoring": "atlas_rf", "E2E-SWE": "e2e",
}


def reward_evidence(documents):
    """Find literal reward writes and familiar score keys in verifier text."""
    rows = []
    for path, body in documents.items():
        if path.endswith((".patch", ".diff")) or "rubric" in path.lower():
            continue
        for label, regex in (("Literal reward.txt write", LITERAL_REWARD), ("Score-like key", REWARD_FIELD)):
            for match in regex.finditer(body):
                line = body.count("\n", 0, match.start()) + 1
                rows.append({"signal": label, "value": match.group(1), "location": f"{path}:{line}"})
    return rows


def grading_units(documents, groups, rubrics):
    """Create name/criterion-level evidence units for lexical cross-referencing."""
    rows = []
    for name in ("f2p", "p2p"):
        for value in groups[name] or []:
            if isinstance(value, str):
                rows.append({"kind": name.upper(), "text": value, "source": "Dataset test group"})
    for item in rubrics:
        rows.append({"kind": "Rubric", "text": item["criterion"], "source": item["source"]})
    for path, body in documents.items():
        if path.endswith((".json", ".txt", ".md")):
            continue
        for match in TEST_NAME.finditer(body):
            if path.endswith((".patch", ".diff")):
                line_start = body.rfind("\n", 0, match.start()) + 1
                if not body.startswith("+", line_start):
                    continue
            text = next(value for value in match.groups() if value is not None)
            # A tiny set of templates use named scratch tests called "test".
            if len(text) < 5:
                continue
            line = body.count("\n", 0, match.start()) + 1
            rows.append({"kind": "Static test name", "text": text, "source": f"{path}:{line}"})
        if not path.endswith((".patch", ".diff")):
            for match in CHECK_SYMBOL.finditer(body):
                name = match.group(1)
                if len(name) < 8:
                    continue
                line = body.count("\n", 0, match.start()) + 1
                rows.append({"kind": "Static checker symbol", "text": name, "source": f"{path}:{line}"})
    return list({(row["kind"], row["text"]): row for row in rows}.values())


def terms(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return Counter(
        part for part in re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
        if part not in STOP
    )


def match_clauses(clauses, units, min_similarity=0.15):
    """Rank visible tests and rubrics against each request using local TF-IDF cosine.

    Returns one top candidate per request when above the supplied threshold. An
    unpaired request is neither a finding nor a test-coverage diagnosis.
    """
    if not units:
        return []
    unit_words = [terms(unit["text"]) for unit in units]
    document_frequency = Counter(term for words in unit_words for term in words)
    weights = {term: 1 + math.log((1 + len(units)) / (1 + count)) for term, count in document_frequency.items()}

    def vector(words):
        values = {term: (1 + math.log(count)) * weights.get(term, 1) for term, count in words.items()}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {term: value / norm for term, value in values.items()} if norm else {}

    vectors = [vector(words) for words in unit_words]
    postings = {}
    for index, values in enumerate(vectors):
        for term, weight in values.items():
            postings.setdefault(term, []).append((index, weight))

    matches = []
    for clause in clauses:
        if clause["role"] not in ACTION_ROLES:
            continue
        scores = Counter()
        for term, weight in vector(terms(clause["text"])).items():
            for index, other_weight in postings.get(term, ()):
                scores[index] += weight * other_weight
        candidate = scores.most_common(1)
        if candidate:
            index, score = candidate[0]
            if score >= min_similarity:
                shared = vector(terms(clause["text"])).keys() & vectors[index].keys()
                matches.append({
                    **clause, "score": round(min(score, 1), 3),
                    "shared_terms": ", ".join(sorted(shared)), **{f"unit_{k}": v for k, v in units[index].items()},
                })
    return matches


def grade_file_url(benchmark, task_id, path):
    if benchmark not in VERIFIER_SOURCES:
        return ""
    repo, revision, prefix = VERIFIER_SOURCES[benchmark]
    folder = f"qa-{task_id}" if benchmark == "SWE Atlas Q&A" else task_id
    return f"https://github.com/{repo}/blob/{revision}/{prefix}/{folder}/tests/{path}"


def feature_inputs(sources, raw, index):
    """Yield source rows and static verifier material keyed by dataset task ID."""
    @lru_cache(maxsize=256)
    def blob(sha):
        return (raw / "verification" / "blobs" / sha).read_text(encoding="utf-8", errors="replace")

    for source in sources:
        benchmark = source["name"]
        if source["cohort"] == "subset":
            continue
        format_kind = TASK_FORMATS.get(benchmark, source["format"])
        location = raw / source["location"]
        if source["format"] == "harbor":
            def harbor():
                for f in sorted(location.glob("*/task.toml")):
                    task = tomllib.loads(f.read_text(encoding="utf-8"))
                    config = {**task.get("verifier", {}), "_task_config": True, "_artifacts": task.get("artifacts")}
                    yield f.parent.name, {}, config, task.get("metadata", {})
            records = harbor()
        else:
            table = pd.read_csv(location) if format_kind == "atlas_qa" else pd.read_parquet(location)
            id_key = "task_id" if format_kind in {"atlas_qa", "deep"} else "instance_id"
            records = ((row[id_key], row, {}, {}) for row in table.to_dict("records"))

        for task_id, record, config, metadata in records:
            files = index.get(benchmark, {}).get(task_id, [])
            documents = {f["path"]: blob(f["sha"]) for f in files if f["selected"]}
            for key, name in (
                ("test_sh", "test.sh"), ("grader_py", "grader.py"), ("config_json", "config.json"),
                ("test_patch", "test.patch"),
            ):
                if isinstance(record.get(key), str) and record[key].strip():
                    documents.setdefault(name, record[key])
            if format_kind == "deep" and record.get("verifier_timeout_sec") is not None:
                config.setdefault("timeout_sec", record["verifier_timeout_sec"])
            yield benchmark, task_id, format_kind, record, config, metadata, files, documents


def build_feature_tables(tasks, sources, raw, min_similarity=0.15):
    """Return a task-level feature frame and an evidence dictionary for drilldown."""
    index = json.loads((raw / "verification" / "index.json").read_text(encoding="utf-8"))
    keyed = tasks.set_index(["benchmark", "task_id"], verify_integrity=True)
    features, evidence = [], {}
    for benchmark, task_id, kind, record, config, meta, files, docs in feature_inputs(sources, raw, index):
        key = benchmark, task_id
        task = keyed.loc[key]
        clauses = task_clauses(task["prompt"], task["supplemental_text"])
        if kind in {"verified", "pro_v1", "pro_v2", "deep"} and not any(c["role"] in ACTION_ROLES for c in clauses):
            for clause in [c for c in clauses if c["origin"] == "prompt"][:2]:
                clause["role"] = "Issue description candidate"
        intents = task_intents(clauses)
        deliverables = task_deliverables(clauses)
        groups = grade_test_groups(record, config)
        rubrics = grade_rubrics(record, docs)
        other_rubric = grade_rubrics({}, docs) if kind == "atlas_qa" else []
        rubric_variant = None
        if other_rubric and rubrics:
            csv_criteria = {item["criterion"] for item in rubrics}
            git_criteria = {item["criterion"] for item in other_rubric}
            rubric_variant = "Same criterion text" if csv_criteria == git_criteria else "Different criterion text"
        rules = grade_contract(kind, docs, config)
        contract_evidence = reward_evidence(docs)
        checks = verifier_checks(docs, groups, rubrics, kind)
        inventory = verifier_inventory(files, config, (
            name for name in ("test_patch", "grader_py", "test_sh", "config_json", "rubric") if isinstance(record.get(name), str)
        ))
        boundary = verifier_boundary(config, kind, [f["path"] for f in files])
        units = grading_units(docs, groups, rubrics)
        matches = match_clauses(clauses, units, min_similarity)
        request_count = sum(clause["role"] in ACTION_ROLES for clause in clauses)
        mandatory = sum(row["importance"].lower() in {"must have", "mandatory", "required", "blocking"} for row in rubrics)
        literal_values = sorted({row["value"] for row in contract_evidence if row["signal"] == "Literal reward.txt write"})
        new = {
            "benchmark": benchmark, "task_id": task_id, "cohort": task["cohort"],
            "request_clauses": request_count, "prohibitions": sum(c["role"] == "Prohibition" for c in clauses),
            "explicit_paths": len(deliverables["paths"]),
            "intents": tuple(label for label, rows in intents.items() if rows),
            "deliverable_signals": tuple(label for label, rows in deliverables["signals"].items() if rows),
            "check_signals": tuple(checks),
            "boundary": boundary["boundary"], "grading_family": rules["grading_family"],
            "declared_artifacts": boundary["declared_artifacts"],
            "tests_build_definition": boundary["tests_build_definition"],
            "rubric_variant": rubric_variant,
            "file_evidence": bool(files), "test_patch_evidence": "test.patch" in docs,
            "verifier_entrypoint_text": "test.sh" in docs or "test.bat" in docs,
            "verifier_files": inventory["listed_files"], "verifier_text_files": inventory["readable_text_files"],
            "verifier_timeout_sec": inventory["verifier_timeout_sec"],
            "f2p_count": len(groups["f2p"]) if groups["f2p"] is not None else None,
            "p2p_count": len(groups["p2p"]) if groups["p2p"] is not None else None,
            "declared_cases": groups["declared_cases"],
            "rubric_criteria": len(rubrics) if rubrics else None,
            "mandatory_criteria": mandatory if rubrics else None,
            "visible_grading_units": len(units), "paired_clauses": len(matches),
            "pair_rate": len(matches) / request_count if request_count and units else None,
            "reward_channels": tuple(rules["reward_channels"]),
            "direct_literal_values": tuple(literal_values),
        }
        features.append(new)
        evidence[key] = {
            "task_prompt": task["prompt"], "supplement": task["supplemental_text"],
            "clauses": clauses, "intents": intents, "deliverables": deliverables,
            "inventory": inventory, "boundary": boundary, "checks": checks,
            "grading": rules, "reward_evidence": contract_evidence,
            "rubric": rubrics, "alternate_rubric": other_rubric, "rubric_variant": rubric_variant,
            "units": units, "matches": matches,
            "publisher_task_notes": str(meta.get("verification_explanation", "")),
            "files": [dict(row, url=grade_file_url(benchmark, task_id, row["path"])) for row in files],
        }
    result = pd.DataFrame(features).set_index(["benchmark", "task_id"], verify_integrity=True)
    assert result.index.symmetric_difference(keyed.index).empty, "Tasks and grading records do not match"
    return result, evidence


def inspect_task(evidence, benchmark, task_id):
    """Return prompt, extracted clauses, verifier evidence, rules and ranked pairs."""
    row = evidence[benchmark, task_id]
    return {
        "prompt": row["task_prompt"],
        "supplement": row["supplement"],
        "intent_labels": tuple(label for label, clauses in row["intents"].items() if clauses),
        "deliverables": row["deliverables"],
        "requests": pd.DataFrame([c for c in row["clauses"] if c["role"] in ACTION_ROLES]),
        "boundary": row["boundary"],
        "inventory": row["inventory"],
        "checks": row["checks"],
        "grading_rule": row["grading"],
        "publisher_task_notes": row["publisher_task_notes"],
        "reward_evidence": pd.DataFrame(row["reward_evidence"]),
        "rubric": pd.DataFrame(row["rubric"]),
        "alternate_rubric": pd.DataFrame(row["alternate_rubric"]),
        "rubric_variant": row["rubric_variant"],
        "grading_units": pd.DataFrame(row["units"]),
        "lexical_pairs": pd.DataFrame(row["matches"]),
        "files": pd.DataFrame(row["files"]),
    }
