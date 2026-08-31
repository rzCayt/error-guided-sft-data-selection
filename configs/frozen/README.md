# Frozen public configuration snapshots

This directory contains byte-identical release snapshots of the current execution configurations. The original paths under `configs/` remain the runtime source of truth because historical scripts and immutable manifests refer to them.

`MANIFEST.json` records the source path, snapshot path, byte count, and SHA-256 for every copy. Run:

```bash
python scripts/snapshot_public_configs.py --check
```

The check fails if either the execution source or its public snapshot changes without a newly generated release manifest.
