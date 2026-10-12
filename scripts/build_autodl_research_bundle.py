"""Build an immutable, hash-checked source bundle; no credentials, data download or GPU use."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from build_public_release_manifest import OUTPUT, ROOT, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("Bundle output must be outside the source tree to avoid recursive packaging")
    if output.exists():
        raise FileExistsError(f"Refusing to replace bundle output: {output}")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_public_release.py")],
                   cwd=ROOT, check=True)
    committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if payload() != committed:
        raise RuntimeError("Source tree changed after release verification")
    paths = [ROOT / entry["path"] for entry in committed["files"]] + [OUTPUT]
    members = []
    for path in paths:
        members.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"schema_version": "autodl-source-bundle-v1", "git_commit": git_commit,
                "content_identity": "Member SHA-256 values are authoritative for the packaged source",
                "cuda_qualified": False, "private_assets_included": False, "files": members}
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "autodl-research-upgrade-20261013.tar.gz"
    with archive_path.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for path, entry in zip(paths, members, strict=True):
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Source changed during packaging: {entry['path']}")
            info = tarfile.TarInfo("research-upgrade/" + entry["path"])
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
        metadata = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("research-upgrade/BUNDLE.json")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    with tarfile.open(archive_path) as archive:
        for entry in members:
            member = archive.extractfile("research-upgrade/" + entry["path"])
            if member is None or hashlib.sha256(member.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError("Packaged member verification failed")
    result = {"archive": archive_path.name,
              "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              "file_count": len(members), "cuda_qualified": False}
    with (output / "bundle_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump({**result, **manifest}, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
