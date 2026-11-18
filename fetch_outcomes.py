"""Download public grading artifacts only, without running any task."""

from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlencode
from xml.etree import ElementTree

from fetch_verifiers import get

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "outcomes"
GITHUB = "https://raw.githubusercontent.com"
API = "https://api.github.com/repos"
DEEP = "https://deepswe.datacurve.ai/artifacts/v1.1"
DEEP_CONFIGS = ("mini_swe_agent_claude_sonnet_5_high", "mini_swe_agent_gpt_5_6_sol_medium")
ATLAS_REV = "b2710531ada270d5a496cf5ae9b8d8b94d022d08"
PRO_REV = "66f92766bba642462d4bbe5479e83f91f9211862"


def cache(rel, url):
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        data = get(url)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return path


def json_cache(rel, url):
    return json.loads(cache(rel, url).read_bytes())


def plan():
    release = json_cache("deep/release.json", f"{DEEP}/release.json")
    trials = json_cache("deep/trials.json", f"{DEEP}/trials.json")["rows"]
    for config in DEEP_CONFIGS:
        best = {}
        for row in trials:
            if (row.get("source"), row.get("eval_scope"), row.get("config")) != ("deep-swe", "full", config):
                continue
            if "reward.json" not in (row.get("verifier_files") or []):
                continue
            trial = row["trial_name"]
            task = row["task_name"]
            choice = row.get("run_attempt") or 0, trial
            if task not in best or choice < best[task]:
                best[task] = choice
        for task, (_, trial) in best.items():
            template = release["artifact_patterns"]["verifier_file"]
            rel = template.format(trial_name=quote(trial, safe=""), file="reward.json")
            yield f"deep/files/{config}/{trial}/reward.json", f"{release['artifact_base_url'].rstrip('/')}/{rel}"

    atlas = "PhillipChaffee/big-pickle-swe-atlas"
    tree = json_cache("atlas/tree.json", f"{API}/{atlas}/git/trees/{ATLAS_REV}?recursive=1")
    for row in tree["tree"]:
        name = row["path"]
        if name == "results/per_task_results.csv" or name.startswith("results/verifier_logs/") and name.endswith(".txt"):
            yield f"atlas/{name}", f"{GITHUB}/{atlas}/{ATLAS_REV}/{name}"

    pro = "scaleapi/SWE-bench_Pro-os"
    tree = json_cache("pro-v1/tree.json", f"{API}/{pro}/git/trees/{PRO_REV}?recursive=1")
    for row in tree["tree"]:
        name = row["path"]
        if name.startswith("traj/") and name.endswith("/eval_results.json"):
            yield f"pro-v1/{name}", f"{GITHUB}/{pro}/{PRO_REV}/{name}"

    verified = "SWE-bench/experiments"
    rev = json_cache("verified/commit.json", f"{API}/{verified}/commits/main")["sha"]
    endpoint = f"{API}/{verified}/contents/evaluation/verified?ref={rev}"
    runs = json_cache("verified/submissions.json", endpoint)
    for run in runs:
        name = run["name"]
        if not name.startswith("20260217_mini-v2.0.0_"):
            continue
        yield f"verified/{name}/per_instance_details.json", f"{GITHUB}/{verified}/{rev}/{run['path']}/per_instance_details.json"


def main():
    sources = list(plan())
    manifest, failed = [], []

    def transfer(item):
        rel, url = item
        try:
            path = cache(rel, url)
            return {"path": rel, "source": url, "bytes": path.stat().st_size}
        except OSError as exc:
            return {"path": rel, "source": url, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=18) as pool:
        for row in pool.map(transfer, sources):
            (failed if "error" in row else manifest).append(row)
    (OUT / "manifest.json").write_text(json.dumps({"files": manifest, "unavailable": failed}, indent=2))
    print(f"{len(manifest)} public outcome files cached; {len(failed)} fetch failures")
    for failure in failed[:15]:
        print(failure["path"], failure["error"])



def public_terminal_receipts():
    """Read the trial rows embedded in Harbor Hub's public leaderboard pages."""
    origin = "https://hub.harborframework.com"
    for version, revision in (("4.0", "4"), ("4.1", "5")):
        key = version.replace(".", "-") + "-0"
        board_file = OUT / "terminal" / version / "board.json"
        board_file.parent.mkdir(parents=True, exist_ok=True)
        if not board_file.exists():
            board_file.write_bytes(subprocess.check_output([
                "harbor", "hub", "leaderboard", "show", f"terminal-bench/terminal-bench/{key}", "--json",
            ], timeout=30))
        board = json.loads(board_file.read_bytes())
        # Six score-ranked configurations. Keep all available pages of each.
        for row in [r for r in board["rows"] if r["n_trials"] >= 330][:6]:
            metadata = row["metadata"]
            label = " / ".join(str(x) for x in (
                metadata["model_display"]["label"], metadata["agent_display"]["label"],
                metadata.get("reasoning_effort") or "effort unspecified",
            ))
            url = f"{origin}/datasets/terminal-bench/terminal-bench/{revision}/leaderboards/{key}/rows/{row['id']}"

            def load_page(page):
                page_url = f"{url}?page={page}"
                html = get(page_url).decode()
                chunks = [json.loads(s) for s in re.findall(r"<script>self\.__next_f\.push\((.*?)\)</script>", html, re.S)]
                payload = "".join(str(c[1]) for c in chunks if len(c) > 1 and isinstance(c[1], str))
                start = payload.index('"initialTrials":') + len('"initialTrials":')
                result, _ = json.JSONDecoder().raw_decode(payload[start:])
                if result["page"] != page:
                    raise ValueError(f"Unexpected page from public Hub: {page_url}")
                return result

            output = board_file.parent / f"{row['id']}.json"
            if output.exists():
                continue
            first = load_page(1)
            with ThreadPoolExecutor(max_workers=6) as pool:
                pages = [first, *pool.map(load_page, range(2, first["total_pages"] + 1))]
            trials = [trial for page in pages for trial in page["items"]]
            if len({t["id"] for t in trials}) != first["total"]:
                raise ValueError(f"Missing or duplicate receipt IDs: {url}")
            output.write_text(json.dumps({"source": url, "config": label, "trials": trials}, ensure_ascii=False))
            print(f"Terminal-Bench {version}: {label}, {len(trials)} public receipts", flush=True)



def public_marathon_receipts():
    """Materialize the dashboard's public result rows, without evaluating its script."""
    base = "https://www.swe-marathon.org"
    html = cache("marathon/site.html", f"{base}/").read_text()
    script = re.search(r'src="(/assets/index[^"]+\.js)"', html)
    if not script:
        raise ValueError("Cannot locate the published Marathon dashboard bundle.")
    url = base + script.group(1)
    payload = cache("marathon/dashboard.js", url).read_text()
    tasks = sorted([p.parent.name for p in (ROOT / "data/raw/swe-marathon-1.1/tasks").glob("*/task.toml")],
                   key=len, reverse=True)
    decoder = json.JSONDecoder()
    trials = {}
    for match in re.finditer(r'\{"id":"[^"]+","trial":', payload):
        row, _ = decoder.raw_decode(payload, match.start())
        if "/trajectories/v1.1/" not in row.get("trajectoryUrl", ""):
            continue
        trial = row["id"]
        task = next((item for item in tasks if trial.startswith(item + "-")), None)
        if task is None:
            raise ValueError(f"Could not match published trial to a pinned Marathon task: {trial}")
        trials[trial] = {**row, "task_id": task}
    if set(tasks) != {row["task_id"] for row in trials.values()}:
        raise ValueError("Incomplete task coverage in Marathon dashboard.")
    (OUT / "marathon" / "receipts.json").write_text(json.dumps({"source": url, "trials": list(trials.values())}))
    print(f"SWE-Marathon v1.1: {len(trials):,} dashboard receipts across {len(tasks)} tasks", flush=True)



def public_pro_reports():
    """Fetch one public V1 run's per-task test reports and preserve raw JSON."""
    import pandas as pd

    model = "claude-45haiku-10222025"
    origin = "https://scaleapi-results.s3.amazonaws.com"
    prefix = f"swe-bench-pro/{model}/eval/"
    corpus = ROOT / "data" / "raw" / "swe-bench-pro" / "data" / "v1" / "test-00000-of-00001.parquet"
    wanted = set(pd.read_parquet(corpus, columns=["instance_id"])["instance_id"])
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = {}
    params = {"list-type": "2", "prefix": prefix}
    for page in range(1, 20):
        parsed = ElementTree.fromstring(cache(
            f"pro-v1/s3/{model}/keys-{page}.xml", f"{origin}/?{urlencode(params)}"
        ).read_bytes())
        for node in parsed.findall(".//s3:Contents/s3:Key", ns):
            key = node.text
            task, filename = key.removeprefix(prefix).split("/", 1)
            if filename == "_output.json" and task in wanted:
                keys[task] = key
        token = parsed.findtext("s3:NextContinuationToken", namespaces=ns)
        if not token:
            break
        params["continuation-token"] = token
    else:
        raise ValueError("S3 listing exceeded the expected page count.")

    def get_report(item):
        task, key = item
        return cache(f"pro-v1/s3/{model}/reports/{task}.json", f"{origin}/{quote(key)}")

    with ThreadPoolExecutor(max_workers=20) as pool:
        files = list(pool.map(get_report, keys.items()))
    print(f"SWE-bench Pro V1: cached {len(files)}/{len(wanted)} original test reports for {model}", flush=True)


if __name__ == "__main__":
    import sys
    if "--terminal" in sys.argv:
        public_terminal_receipts()
    elif "--marathon" in sys.argv:
        public_marathon_receipts()
    elif "--pro-reports" in sys.argv:
        public_pro_reports()
    else:
        main()
