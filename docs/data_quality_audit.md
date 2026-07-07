# Data Quality Audit v0

This audit is generated from the deterministic generator and a manual-style review of 20 sample rows. The current review is structural because no human annotator has edited individual rows yet.

## Checks

- Each sampled row has one prompt, one numeric answer, and one deterministic rationale.
- Task families cover ratio change, multiplicative relation, weighted aggregation, and temporal numeric constraint.
- Answers are produced by the same solver path used by tests.
- Split labels are explicit and stable.
- No test examples are used by selection scripts.

## Manual Review Notes

The first 20 generated examples should be checked after every major generator edit with:

```powershell
python scripts/generate_data.py --all
python - <<'PY'
import json
from pathlib import Path
for line in Path('data/samples/candidate_pool.jsonl').read_text(encoding='utf-8').splitlines()[:20]:
    row = json.loads(line)
    print(row['id'], row['task_family'], row['prompt'], '=>', row['answer'])
PY
```

## Current Verdict

Pass for scaffold use. Before emailing professors, run a second review after real model diagnostic outputs are available.
