from eg_sft.data.public_gsm8k import (
    build_gsm8k_split_records,
    build_tulu_candidate_pool,
    build_ngram_reference_index,
    maximum_ngram_overlap,
    sha256_text,
)


def _gsm_rows(count: int, prefix: str) -> list[dict[str, str]]:
    return [
        {
            "question": f"{prefix} question {index}",
            "answer": f"work\n#### {index}",
        }
        for index in range(count)
    ]


def test_gsm8k_splits_are_deterministic_disjoint_and_complete() -> None:
    train = _gsm_rows(10, "train")
    test = _gsm_rows(3, "test")
    sizes = {"calibration": 2, "diagnostic": 3, "utility": 2, "development": 3}

    first_records, first_manifest = build_gsm8k_split_records(
        train_rows=train,
        test_rows=test,
        split_sizes=sizes,
        seed=20260722,
    )
    second_records, second_manifest = build_gsm8k_split_records(
        train_rows=train,
        test_rows=test,
        split_sizes=sizes,
        seed=20260722,
    )

    assert first_records == second_records
    assert first_manifest == second_manifest
    assert first_manifest["protocol_split_counts"] == sizes
    assert first_manifest["question_hash_overlap_train_test"] == 0
    assert len({record["record_id"] for record in first_records}) == 13


def test_candidate_pool_deduplicates_excludes_and_is_deterministic() -> None:
    rows = [
        {
            "dataset": "toy",
            "id": f"id-{index}",
            "index": index,
            "messages": [
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ],
        }
        for index in range(6)
    ]
    rows.append(
        {
            "dataset": "toy",
            "id": "duplicate",
            "index": 99,
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Different duplicate answer"},
            ],
        }
    )
    excluded = {sha256_text("Question 2")}

    first, first_manifest = build_tulu_candidate_pool(
        rows=rows,
        pool_size=4,
        seed=20260722,
        excluded_user_prompt_hashes=excluded,
    )
    second, second_manifest = build_tulu_candidate_pool(
        rows=reversed(rows),
        pool_size=4,
        seed=20260722,
        excluded_user_prompt_hashes=excluded,
    )

    assert first == second
    assert first_manifest == second_manifest
    assert first_manifest["unique_prompt_count_before_sampling"] == 5
    assert first_manifest["exact_gsm8k_user_prompt_exclusions"] == 1
    assert len({row["prompt_sha256"] for row in first}) == 4


def test_ngram_overlap_catches_embedded_reference_question() -> None:
    references = [
        "Natalia sold clips to 48 friends and half as many in May. How many total?"
    ]
    reference_sets, inverted = build_ngram_reference_index(references, n=3)
    reference_index, jaccard, containment = maximum_ngram_overlap(
        "Please solve carefully: Natalia sold clips to 48 friends and half as many "
        "in May. How many total? Explain your answer.",
        reference_sets=reference_sets,
        inverted_index=inverted,
        n=3,
    )

    assert reference_index == 0
    assert jaccard < 1.0
    assert containment == 1.0


def test_candidate_pool_excludes_fuzzy_gsm8k_match() -> None:
    reference = "A shop sold ten books on Monday and five on Tuesday. How many total?"
    rows = [
        {
            "dataset": "toy",
            "id": "fuzzy-match",
            "index": 0,
            "messages": [
                {
                    "role": "user",
                    "content": f"Please solve: {reference} Show all work.",
                },
                {"role": "assistant", "content": "15"},
            ],
        },
        *[
            {
                "dataset": "toy",
                "id": f"safe-{index}",
                "index": index,
                "messages": [
                    {"role": "user", "content": f"Unrelated prompt number {index}"},
                    {"role": "assistant", "content": f"Response {index}"},
                ],
            }
            for index in range(1, 5)
        ],
    ]

    selected, manifest = build_tulu_candidate_pool(
        rows=rows,
        pool_size=3,
        seed=20260722,
        excluded_user_prompt_hashes=set(),
        excluded_reference_texts=[reference],
        fuzzy_ngram_size=3,
        fuzzy_threshold=0.8,
    )

    assert "fuzzy-match" not in {row["source_id"] for row in selected}
    assert manifest["fuzzy_gsm8k_exclusion_count_before_pool_filled"] == 1
