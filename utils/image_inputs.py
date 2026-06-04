"""Helpers for Hugging Face inline image entries used by evaluation."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image


def _open_image_bytes(image_bytes: bytes | bytearray) -> Image.Image:
    with BytesIO(image_bytes) as buffer:
        image = Image.open(buffer)
        image.load()
    return image


def normalize_image(image: Any) -> Image.Image:
    """Convert supported inline image payloads to a PIL image."""
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, (bytes, bytearray)):
        return _open_image_bytes(image)

    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return _open_image_bytes(image["bytes"])
        if image.get("path"):
            return Image.open(Path(image["path"]))

    raise TypeError(f"Unsupported image payload type: {type(image).__name__}")


def entry_images(entry: Dict[str, Any]) -> List[Image.Image]:
    images = entry.get("image") or []
    if not isinstance(images, list):
        raise TypeError("Expected entry['image'] to be a list of inline images")
    return [normalize_image(image) for image in images if image is not None]


def image_count(entry: Dict[str, Any]) -> int:
    return len(entry_images(entry))


def image_summary(entry: Dict[str, Any]) -> str:
    count = image_count(entry)
    return f"inline_images:{count}" if count else ""


def _normalize_format(image: Image.Image) -> str:
    fmt = (getattr(image, "format", None) or "PNG").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    return fmt


def encode_image(image: Image.Image) -> tuple[bytes, str]:
    image = normalize_image(image)
    fmt = _normalize_format(image)
    image_to_save = image
    if fmt == "JPEG" and image.mode in {"RGBA", "LA", "P"}:
        image_to_save = image.convert("RGB")

    buffer = BytesIO()
    image_to_save.save(buffer, format=fmt)
    mime_type = Image.MIME.get(fmt, "image/png")
    return buffer.getvalue(), mime_type


def data_url_for_image(image: Image.Image) -> str:
    image_bytes, mime_type = encode_image(image)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def openai_image_parts(images: Iterable[Image.Image]) -> List[Dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": data_url_for_image(image)}}
        for image in images
    ]


def add_openai_images_to_messages(messages: List[Dict[str, Any]], images: List[Image.Image]) -> List[Dict[str, Any]]:
    if not images:
        return messages

    updated = [dict(message) for message in messages]
    image_parts = openai_image_parts(images)
    for message in reversed(updated):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}, *image_parts]
        elif isinstance(content, list):
            message["content"] = [*content, *image_parts]
        else:
            raise TypeError("Unsupported OpenAI message content type for image insertion")
        return updated

    raise ValueError("No user message found for image insertion")
