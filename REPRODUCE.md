# Reproduction and verification

## Scope

The public repository supports three different levels of verification:

1. **CPU release verification** checks code, public artifacts, hashes, README numbers, links, and claim boundaries.
2. **Artifact re-analysis** recomputes the public result summary from the committed audited evidence.
3. **GPU experiment reproduction** requires separately downloaded upstream data and model assets and is not performed by the public-release CI.

## CPU setup

Use Python 3.10 or 3.11 in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the public checks:

```bash
python scripts/reproduce_public_summary.py --check
python scripts/verify_public_release.py
python scripts/build_public_release_manifest.py --check
python -m pytest -q
python -m ruff check .
```

Expected behavior:

- `reproduce_public_summary.py --check` prints `PASS` only when the canonical JSON, CSV, and Markdown table match the audited inputs;
- `verify_public_release.py` exits nonzero on missing files, hash drift, unsupported public claims, secrets, absolute local paths, restricted artifacts, or broken Markdown links;
- the full CPU test suite completes without skipped failures replacing current evidence;
- Ruff reports no violations.

## Canonical result flow

```text
audited evidence files
→ scripts/reproduce_public_summary.py
→ results/public_summary/main_results.json
→ generated CSV / Markdown table / figures / README checks
```

The public summary must not be hand-edited. If an audited input changes, regenerate into a new release and document why the evidence identity changed.

## GPU reproduction boundary

GPU reproduction requires the frozen model and dataset revisions recorded in `configs/`, sufficient storage, and acceptance of each upstream license. Large raw generations, checkpoints, adapter weights, and restricted dataset text are intentionally excluded from Git. Formal GPU runs must use immutable output directories and complete the repository's per-cell audit before being counted.

State Dependence v3 is not a completed GPU result. Its public artifacts document the frozen panel and CPU preflight only.
