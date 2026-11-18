"""Fetch public test and rubric text. No benchmark code is executed."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "raw" / "verification"

# GitHub paths and source revisions. The original prompt data may be mirrored on HF.
SOURCES = {
    "Terminal-Bench 4.1": ("harbor-framework/terminal-bench", "4070d6cef56a7d20cb5dc98d5e10ada88a6b9440", "tasks"),
    "Terminal-Bench 4.0": ("harbor-framework/terminal-bench", "452bf305c6daa62fc59061d22133a7cbc7c1572e", "tasks"),
    "SWE Atlas Q&A": ("scaleapi/SWE-Atlas", "b3cdddf44414c0ccca50c6e89fe978bb4d844d26", "data/qa"),
    "SWE Atlas Test Writing": ("scaleapi/SWE-Atlas", "b3cdddf44414c0ccca50c6e89fe978bb4d844d26", "data/tw"),
    "SWE Atlas Refactoring": ("scaleapi/SWE-Atlas", "b3cdddf44414c0ccca50c6e89fe978bb4d844d26", "data/rf"),
    "SWE-Marathon 1.1": ("abundant-ai/swe-marathon", "7bce7b29e1f2413633eb5ed6eb10dc2efb036712", "tasks"),
    "E2E-SWE": ("facebookresearch/E2E-SWE", "e696e96cb0f024d3231a9c1a41527c016d872f36", "tasks"),
}
CODE_SUFFIXES = set(".py .sh .bash .js .mjs .cjs .ts .tsx .go .rs .c .cpp .h .java .kt .cs .rb .php .swift .r .lua .sql .ps1 .bat .scm .clj .scala .zig".split())
CONFIG_NAMES = {"dockerfile", "rubrics.json", "rubric.json", "config.json", "reward.json",
                "grader_config.json", "criteria.json", "verifier_config.json"}
MAX_BYTES = 256 * 1024


def get(url):
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "swe-task-eda"})
            with urlopen(request, timeout=40) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def metadata(repo, revision):
    path = DEST / "trees" / f"{revision}.json"
    if not path.exists():
        path.write_bytes(get(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"))
    tree = json.loads(path.read_text())
    if tree.get("truncated"):
        raise ValueError(f"Incomplete GitHub tree for {repo}")
    return tree["tree"]


def is_text_candidate(name):
    path = PurePosixPath(name.lower())
    if any(part in {"fixtures", "fixture", "vendor", "node_modules", "testdata", "test_data"}
           for part in path.parts):
        return False
    return (
        path.suffix in CODE_SUFFIXES
        or path.name in CONFIG_NAMES
        or (
            path.suffix in {".txt", ".md", ".json", ".toml", ".yaml", ".yml"}
            and any(word in path.stem for word in ("rubric", "criteria", "threshold", "scorer", "grade", "reward", "checks"))
        )
    )


def fetch():
    (DEST / "trees").mkdir(parents=True, exist_ok=True)
    (DEST / "blobs").mkdir(exist_ok=True)
    tree_cache, pending, index = {}, {}, {}
    for benchmark, (repo, revision, prefix) in SOURCES.items():
        if revision not in tree_cache:
            tree_cache[revision] = metadata(repo, revision)
        rows = {}
        index[benchmark] = rows
        for item in tree_cache[revision]:
            if item["type"] != "blob" or not item["path"].startswith(prefix + "/"):
                continue
            relative = item["path"].removeprefix(prefix + "/")
            parts = relative.split("/", 2)
            if len(parts) != 3 or parts[1] != "tests":
                continue
            task, _, filename = parts
            task = task.removeprefix("qa-") if benchmark == "SWE Atlas Q&A" else task
            sha = item["sha"]
            row = {"path": filename, "sha": sha, "bytes": item["size"], "selected": False}
            rows.setdefault(task, []).append(row)
            if item["size"] <= MAX_BYTES and is_text_candidate(filename):
                url = f"https://raw.githubusercontent.com/{repo}/{revision}/{quote(item['path'])}"
                pending.setdefault(sha, url)
                row["selected"] = True

    def save(item):
        sha, url = item
        path = DEST / "blobs" / sha
        if path.exists():
            return
        body = get(url)
        blob_id = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if blob_id != sha:
            raise ValueError(f"Git blob checksum mismatch: {url}")
        path.write_bytes(body)

    print(f"Fetching {len(pending):,} unique public verifier text files", flush=True)
    with ThreadPoolExecutor(max_workers=20) as pool:
        for done, _ in enumerate(pool.map(save, pending.items()), 1):
            if done % 250 == 0:
                print(f"Verified {done:,}/{len(pending):,}", flush=True)
    (DEST / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    for benchmark, tasks in index.items():
        files = [f for files in tasks.values() for f in files]
        print(f"{benchmark}: {len(tasks)} tasks, {sum(f['selected'] for f in files)} text files")


if __name__ == "__main__":
    fetch()
