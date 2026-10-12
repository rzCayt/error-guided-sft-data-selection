"""Price a batch from measured instance wall time; never start paid compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()
from eg_sft.experiment.research_upgrade import plan_batch, read_object  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--units", type=int, required=True)
    parser.add_argument("--hourly-cny", required=True)
    parser.add_argument("--quote-at", required=True)
    parser.add_argument("--setup-hours", default="0")
    parser.add_argument("--storage-cny", default="0")
    parser.add_argument("--margin", default="1.25")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan_batch(ledger=read_object(args.ledger), profile=read_object(args.profile),
                        phase=args.phase, batch_id=args.batch_id, units=args.units,
                        hourly_cny=args.hourly_cny, quote_at=args.quote_at,
                        setup_hours=args.setup_hours, storage_cny=args.storage_cny,
                        margin=args.margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "reserve_cny": result["reserve_cny"],
                      "launch_authorized": False}))


if __name__ == "__main__":
    main()
