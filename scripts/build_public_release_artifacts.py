"""Build a sanitized, deterministic professor-facing evidence bundle.

The source artifacts remain untouched. The builder rewrites machine-specific
model/cache paths to a canonical Hugging Face model identifier and emits a
SHA-256 manifest for the copied files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "professor_package_validation"
DEFAULT_OUTPUT = ROOT / "results" / "public_release_v1"
CANONICAL_MODEL = "Qwen/Qwen3-1.7B"
PIPELINE_SOURCE = "qwen3_1_7b_exact_chat_pipeline_check_25"
PIPELINE_OUTPUT = "model_pipeline_check_25"
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:home|Users)/"),
)


def scalar_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from scalar_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from scalar_strings(nested)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing public bundle: {path}. "
            "Choose a new --output directory."
        )


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sanitize_jsonl(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            row["model"] = CANONICAL_MODEL
            writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_metadata(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["model"] = CANONICAL_MODEL
    payload["raw_outputs_path"] = (
        f"results/public_release_v1/{PIPELINE_OUTPUT}/scale_model_diagnostic_outputs.jsonl"
    )
    payload["public_artifact_note"] = (
        "Machine-specific cache paths were replaced by the canonical model ID; "
        "model and tokenizer revisions are unchanged."
    )
    write_json(destination, payload)


def sanitize_summary_csv(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as reader:
        rows = list(csv.DictReader(reader))
        fieldnames = list(rows[0]) if rows else []
    for row in rows:
        row["model"] = CANONICAL_MODEL
        if "raw_outputs_path" in row:
            row["raw_outputs_path"] = (
                f"results/public_release_v1/{PIPELINE_OUTPUT}/"
                "scale_model_diagnostic_outputs.jsonl"
            )
    with destination.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames, lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def copy_text(source: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())


def assert_no_absolute_paths(files: list[Path]) -> None:
    offenders: list[str] = []
    for path in files:
        strings: list[str] = []
        if path.suffix == ".json":
            strings.extend(scalar_strings(json.loads(path.read_text(encoding="utf-8-sig"))))
        elif path.suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        strings.extend(scalar_strings(json.loads(line)))
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.reader(handle):
                    strings.extend(row)
        else:
            strings.append(path.read_text(encoding="utf-8-sig", errors="replace"))
        if any(pattern.search(value) for value in strings for pattern in ABSOLUTE_PATH_PATTERNS):
            offenders.append(path.as_posix())
    if offenders:
        raise ValueError(f"Absolute paths remain in public artifacts: {offenders}")


def build_bundle(source_root: Path, output_root: Path) -> list[Path]:
    ensure_new_output(output_root)
    pipeline_source = source_root / PIPELINE_SOURCE
    pipeline_output = output_root / PIPELINE_OUTPUT
    selector_source = source_root / "selector_identifiability_rerun"
    selector_output = output_root / "selector_identifiability_rerun"
    pipeline_output.mkdir(parents=True)
    selector_output.mkdir(parents=True)

    generated: list[Path] = []

    outputs_name = "scale_model_diagnostic_outputs.jsonl"
    metadata_name = "scale_model_diagnostic_run_metadata.json"
    summary_name = "scale_model_diagnostic_summary.csv"
    sanitize_jsonl(pipeline_source / outputs_name, pipeline_output / outputs_name)
    sanitize_metadata(pipeline_source / metadata_name, pipeline_output / metadata_name)
    sanitize_summary_csv(pipeline_source / summary_name, pipeline_output / summary_name)
    generated.extend(
        [
            pipeline_output / outputs_name,
            pipeline_output / metadata_name,
            pipeline_output / summary_name,
        ]
    )

    for name in ("scale_model_error_profile.csv", "scale_model_error_profile_by_type.csv"):
        copy_text(pipeline_source / name, pipeline_output / name)
        generated.append(pipeline_output / name)

    for source in sorted(selector_source.iterdir()):
        if source.is_file():
            destination = selector_output / source.name
            copy_text(source, destination)
            generated.append(destination)

    assert_no_absolute_paths(generated)

    manifest_entries = []
    for path in sorted(generated):
        manifest_entries.append(
            {
                "path": path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "claim_boundary": (
            "This bundle supports deterministic selector-audit reproduction and "
            "a bounded model pipeline check only; it does not show selection or SFT effectiveness."
        ),
        "canonical_model": CANONICAL_MODEL,
        "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "files": manifest_entries,
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    generated.append(manifest_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generated = build_bundle(args.source.resolve(), args.output.resolve())
    print(f"Built {len(generated)} public artifacts under {args.output}")


if __name__ == "__main__":
    main()
