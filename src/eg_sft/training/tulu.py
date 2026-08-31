"""Frozen response-only formatting for processed Tulu candidates."""

from __future__ import annotations

from collections.abc import Sequence


def tulu_response_only_parts(
    messages: Sequence[dict[str, str]],
    *,
    eos_token: str,
) -> tuple[str, str]:
    """Return prompt and final assistant response for response-only SFT."""

    if len(messages) < 2:
        raise ValueError("candidate needs prompt messages and a response")
    if messages[-1].get("role") != "assistant":
        raise ValueError("candidate must end with assistant")

    prompt_pieces: list[str] = []
    for message in messages[:-1]:
        role = message.get("role")
        content = message.get("content", "").strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        if not content and role != "system":
            raise ValueError(f"empty {role} message")
        suffix = eos_token if role == "assistant" else ""
        prompt_pieces.append(f"<|{role}|>\n{content}{suffix}\n")
    prompt_pieces.append("<|assistant|>\n")

    response = str(messages[-1].get("content", "")).strip()
    if not response:
        raise ValueError("empty final assistant response")
    return "".join(prompt_pieces), response
