"""OpenAI-compatible chat completions helper used by evaluation methods."""

from __future__ import annotations

from typing import Any, Dict, List

import openai

from utils.image_inputs import add_openai_images_to_messages


REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _completion_params(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if model.startswith(REASONING_MODEL_PREFIXES):
        params["max_completion_tokens"] = max_tokens
        return params

    params["max_tokens"] = max_tokens
    if temperature is not None and "reasoner" not in model.lower():
        params["temperature"] = temperature
    return params


def _message_text(message: Any) -> str:
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return f"{reasoning}\n\n---\n\n{content}"
    return content


def _completion_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, "completion_tokens", 0) or 0)


def generate_with_api(
    model: str,
    conversation: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    images: list[Any] | None = None,
) -> tuple[str, int]:
    """Generate a response through the configured OpenAI-compatible endpoint."""
    messages = add_openai_images_to_messages(conversation, images or [])
    params = _completion_params(model, messages, max_tokens, temperature)

    try:
        response = openai.chat.completions.create(**params)
        message = response.choices[0].message
        return _message_text(message).strip(), _completion_tokens(response)
    except Exception as exc:
        print(f"Error with OpenAI-compatible API ({model}): {exc}")
        return "", 0
