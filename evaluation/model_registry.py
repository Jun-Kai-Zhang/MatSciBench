"""User-editable model registry for evaluation endpoints.

Each entry is an OpenAI-compatible chat completions endpoint. Add or edit
entries here when evaluating a new hosted model.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import openai


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    endpoint_url: str
    api_key_env: str
    multimodal: bool


OPENAI_ENDPOINT = "https://api.openai.com/v1"
GEMINI_OPENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # OpenAI
    "gpt-4o": ModelConfig("gpt-4o", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "gpt-4.1": ModelConfig("gpt-4.1", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "gpt-4.1-2025-04-14": ModelConfig("gpt-4.1-2025-04-14", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "gpt-4.1-mini": ModelConfig("gpt-4.1-mini", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "o4-mini": ModelConfig("o4-mini", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "o4-mini-2025-04-16": ModelConfig("o4-mini-2025-04-16", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "o3": ModelConfig("o3", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),
    "gpt-5": ModelConfig("gpt-5", OPENAI_ENDPOINT, "OPENAI_API_KEY", True),

    # Gemini via its OpenAI-compatible endpoint.
    "gemini-2.5-flash": ModelConfig("gemini-2.5-flash", GEMINI_OPENAI_ENDPOINT, "GEMINI_API_KEY", True),
    "gemini-2.5-flash-lite": ModelConfig("gemini-2.5-flash-lite", GEMINI_OPENAI_ENDPOINT, "GEMINI_API_KEY", True),
    "gemini-2.5-pro": ModelConfig("gemini-2.5-pro", GEMINI_OPENAI_ENDPOINT, "GEMINI_API_KEY", True),
    "gemini-3-flash-preview": ModelConfig("gemini-3-flash-preview", GEMINI_OPENAI_ENDPOINT, "GEMINI_API_KEY", True),

    # Other OpenAI-compatible providers.
    "deepseek-chat": ModelConfig("deepseek-chat", DEEPSEEK_ENDPOINT, "DEEPSEEK_API_KEY", False),
    "deepseek-reasoner": ModelConfig("deepseek-reasoner", DEEPSEEK_ENDPOINT, "DEEPSEEK_API_KEY", False),
    "qwen3-235b-a22b": ModelConfig("qwen3-235b-a22b", DASHSCOPE_ENDPOINT, "QWEN_API_KEY", False),
    "qwen3-vl-32b-instruct": ModelConfig(
        "qwen/qwen3-vl-32b-instruct",
        OPENROUTER_ENDPOINT,
        "OPENROUTER_API_KEY",
        True,
    ),
    "qwen2.5-vl-72b-instruct": ModelConfig(
        "qwen/qwen2.5-vl-72b-instruct",
        OPENROUTER_ENDPOINT,
        "OPENROUTER_API_KEY",
        True,
    ),
    "gemini-2.5-pro-preview-05-06": ModelConfig(
        "google/gemini-2.5-pro-preview-05-06",
        OPENROUTER_ENDPOINT,
        "OPENROUTER_API_KEY",
        True,
    ),

    # OpenRouter.
    "llama-4-maverick": ModelConfig(
        "meta-llama/llama-4-maverick",
        OPENROUTER_ENDPOINT,
        "OPENROUTER_API_KEY",
        True,
    ),
    "moonshotai/kimi-k2": ModelConfig(
        "moonshotai/kimi-k2",
        OPENROUTER_ENDPOINT,
        "OPENROUTER_API_KEY",
        False,
    ),
}


FORMULA_JUDGE_MODEL = "gemini-2.5-flash-lite"


def get_model_config(model: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        known_models = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model {model!r}. Add it to evaluation/model_registry.py. "
            f"Known models: {known_models}"
        ) from exc


def get_api_key(config: ModelConfig) -> str:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key environment variable {config.api_key_env} for model "
            f"{config.model_name!r}."
        )
    return api_key


def configure_openai_compatible_client(config: ModelConfig) -> None:
    openai.api_key = get_api_key(config)
    openai.base_url = config.endpoint_url.rstrip("/") + "/"


def multimodal_model_names() -> list[str]:
    return [name for name, config in MODEL_REGISTRY.items() if config.multimodal]
