"""Export frozen query/candidate similarity and near-duplicate cluster inputs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from eg_sft.data.public_gsm8k import normalize_text, word_ngrams
from eg_sft.selection.rds import cosine_similarity_matrix
from eg_sft.training.b500 import file_sha256, read_jsonl


def eligible_candidate_rows(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the frozen response-only-trainable subset in inventory order.

    The public full-pool inventory keeps all 10,000 audited rows, while RDS
    embeddings exist only for rows whose response survives max-length
    truncation. Older synthetic fixtures omit the eligibility field and are
    treated as already filtered.
    """

    has_field = ["response_only_trainable" in row for row in candidates]
    if any(has_field) and not all(has_field):
        raise ValueError("candidate eligibility field is only partially present")
    if not any(has_field):
        return list(candidates)
    invalid = [
        str(row.get("candidate_id", ""))
        for row in candidates
        if not isinstance(row["response_only_trainable"], bool)
    ]
    if invalid:
        raise ValueError("candidate eligibility values must be boolean")
    return [row for row in candidates if row["response_only_trainable"]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_embedding_chunks(
    run_dir: Path, *, kind: str, expected_ids: Sequence[str]
) -> torch.Tensor:
    """Load and verify immutable RDS embedding chunks in index order."""

    if kind not in {"query", "candidate"}:
        raise ValueError("kind must be query or candidate")
    chunk_dir = run_dir / "embedding_chunks" / kind
    manifests = sorted(chunk_dir.glob("chunk_*.json"))
    if not manifests:
        raise FileNotFoundError(f"no {kind} embedding chunk manifests")
    ids: list[str] = []
    tensors: list[torch.Tensor] = []
    expected_start = 0
    contract_hash: str | None = None
    representation: str | None = None
    for expected_index, manifest_path in enumerate(manifests):
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "COMPLETE":
            raise ValueError(f"incomplete embedding chunk: {manifest_path.name}")
        if int(manifest["chunk_index"]) != expected_index:
            raise ValueError("embedding chunk indices are not contiguous")
        if int(manifest["start_index"]) != expected_start:
            raise ValueError("embedding chunk ranges are not contiguous")
        artifact = chunk_dir / str(manifest["artifact_file"])
        if file_sha256(artifact) != manifest["artifact_sha256"]:
            raise ValueError(f"embedding chunk hash changed: {artifact.name}")
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        if payload.get("kind") != kind or int(payload["chunk_index"]) != expected_index:
            raise ValueError("embedding payload identity differs from manifest")
        chunk_ids = [str(value) for value in payload["ids"]]
        tensor = payload["embeddings"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError("embedding payload tensor is invalid")
        if tensor.shape[0] != len(chunk_ids):
            raise ValueError("embedding IDs and tensor rows differ")
        if not torch.isfinite(tensor).all():
            raise ValueError("embedding payload contains non-finite values")
        current_contract = str(payload["run_contract_sha256"])
        current_representation = str(payload["representation_version"])
        contract_hash = contract_hash or current_contract
        representation = representation or current_representation
        if current_contract != contract_hash or current_representation != representation:
            raise ValueError("embedding chunks do not share one frozen contract")
        ids.extend(chunk_ids)
        tensors.append(tensor.float())
        expected_start += len(chunk_ids)
    if ids != list(expected_ids):
        raise ValueError(f"{kind} embedding order differs from frozen inventory")
    return torch.cat(tensors, dim=0)


def export_similarity_artifact(
    *,
    run_dir: Path,
    query_inventory_path: Path,
    candidate_inventory_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    queries = read_jsonl(query_inventory_path)
    candidates = eligible_candidate_rows(read_jsonl(candidate_inventory_path))
    query_ids = [str(row["record_id"]) for row in queries]
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    query_embeddings = load_embedding_chunks(
        run_dir, kind="query", expected_ids=query_ids
    )
    candidate_embeddings = load_embedding_chunks(
        run_dir, kind="candidate", expected_ids=candidate_ids
    )
    similarity = cosine_similarity_matrix(query_embeddings, candidate_embeddings).cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(
            {
                "schema_version": "budget-equivalent-similarity-v3",
                "query_ids": query_ids,
                "candidate_ids": candidate_ids,
                "similarity": similarity,
                "source_query_inventory_sha256": file_sha256(query_inventory_path),
                "source_candidate_inventory_sha256": file_sha256(candidate_inventory_path),
            },
            handle,
        )
    return {
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "shape": list(similarity.shape),
        "dtype": str(similarity.dtype),
        "minimum": float(similarity.min().item()),
        "maximum": float(similarity.max().item()),
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _minhash_signature(grams: set[tuple[str, ...]], *, count: int) -> tuple[int, ...]:
    if not grams:
        return tuple(0 for _ in range(count))
    signature = []
    encoded = ["\x1f".join(gram).encode("utf-8") for gram in grams]
    for seed in range(count):
        seed_bytes = seed.to_bytes(2, "big")
        signature.append(
            min(
                int.from_bytes(
                    hashlib.blake2b(seed_bytes + gram, digest_size=8).digest(), "big"
                )
                for gram in encoded
            )
        )
    return tuple(signature)


def cluster_near_duplicate_prompts(
    candidate_ids: Sequence[str],
    prompt_texts: Sequence[str],
    *,
    ngram_size: int = 5,
    minhash_count: int = 64,
    band_size: int = 4,
    jaccard_threshold: float = 0.80,
    containment_threshold: float = 0.90,
    maximum_bucket_size: int = 500,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Cluster likely prompt near-duplicates using MinHash-LSH plus exact checks."""

    if len(candidate_ids) != len(prompt_texts) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs and prompts must be aligned and unique")
    if minhash_count % band_size:
        raise ValueError("minhash_count must be divisible by band_size")
    normalized = [normalize_text(text) for text in prompt_texts]
    grams = [word_ngrams(text, n=ngram_size) for text in normalized]
    signatures = [_minhash_signature(value, count=minhash_count) for value in grams]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(minhash_count // band_size):
            start = band * band_size
            buckets[(band, signature[start : start + band_size])].append(index)
    union = _UnionFind(len(candidate_ids))
    compared: set[tuple[int, int]] = set()
    oversized_bucket_count = 0
    for indices in buckets.values():
        if len(indices) > maximum_bucket_size:
            oversized_bucket_count += 1
            continue
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in compared:
                    continue
                compared.add(pair)
                left_grams, right_grams = grams[left], grams[right]
                if not left_grams or not right_grams:
                    is_match = normalized[left] == normalized[right]
                else:
                    intersection = len(left_grams & right_grams)
                    union_size = len(left_grams | right_grams)
                    jaccard_value = intersection / union_size
                    containment = intersection / min(len(left_grams), len(right_grams))
                    is_match = (
                        jaccard_value >= jaccard_threshold
                        or containment >= containment_threshold
                    )
                if is_match:
                    union.union(left, right)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidate_ids)):
        members[union.find(index)].append(index)
    cluster_ids: dict[int, str] = {}
    for root, indices in members.items():
        payload = "\n".join(sorted(candidate_ids[index] for index in indices)) + "\n"
        cluster_ids[root] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rows = [
        {
            "candidate_id": candidate_id,
            "near_duplicate_cluster_id": cluster_ids[union.find(index)],
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]
    sizes = [len(indices) for indices in members.values()]
    return rows, {
        "schema_version": "budget-equivalent-near-duplicate-clusters-v3",
        "candidate_count": len(candidate_ids),
        "cluster_count": len(members),
        "multi_member_cluster_count": sum(size > 1 for size in sizes),
        "maximum_cluster_size": max(sizes, default=0),
        "candidate_pair_checks": len(compared),
        "oversized_lsh_bucket_count": oversized_bucket_count,
        "ngram_size": ngram_size,
        "minhash_count": minhash_count,
        "band_size": band_size,
        "jaccard_threshold": jaccard_threshold,
        "containment_threshold": containment_threshold,
    }


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
