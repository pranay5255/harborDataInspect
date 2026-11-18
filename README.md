# SWE task EDA

The [executed notebook](swe_task_eda.ipynb) includes the analysis and saved outputs. The [DeFi benchmark design report](defi_benchmark_design.md) applies the findings to new evaluation tasks.

For a local rerun, restore `data/raw/` using the pinned revisions, formats, and paths in the [source manifest](data/sources.json). Downloaded data, verifier caches, and outcome archives are excluded from Git. Start JupyterLab from the repository root, open the notebook, and use **Run All Cells**.

```bash
uv sync --locked
uv run python -m jupyterlab --no-browser --ServerApp.ip=127.0.0.1
```

To execute the notebook and save its outputs from the command line:

```bash
uv run python -m nbconvert --to notebook --execute --inplace swe_task_eda.ipynb
```

The notebook loads the task records from `data/raw/`. The [source manifest](data/sources.json) records paths, expected row counts, upstream revisions, and sources. Hugging Face files came from the authenticated `hf` CLI. The three GitHub sources were fetched at pinned revisions using only task instructions and metadata. The DeepSWE Parquet is a tabular mirror of a pinned Datacurve release.

The main analysis covers 1,625 shortlisted records and 186 E2E-SWE tasks. Terminal-Bench 4.0 and SWE-bench Pro V1 appear only in the version comparison. Pro V2 Hard is checked against the full V2 and counted once. The analysis reads public task artifacts without launching benchmark environments. FrontierCode 2.0 is private.

The deeper analysis cells join prompts with public verifier source and rubrics. To refresh the pinned text artifacts, run `uv run python fetch_verifiers.py`. The downloader records the public tests tree, fetches selected text files with checksums, and skips fixture directories, binaries, and files over 256 KiB. The notebook uses static source signals and local TF-IDF. It needs no embedding model, API key, or benchmark execution.

Section 7 uses public run outcomes cached under `data/raw/outcomes/`. Use these commands to fetch reward JSONs, trial metadata, and result reports.

```bash
uv run python fetch_outcomes.py
uv run python fetch_outcomes.py --terminal
uv run python fetch_outcomes.py --marathon
uv run python fetch_outcomes.py --pro-reports
```
