"""
Message content utility functions.

Helpers for manipulating LLM message content that may be either
plain strings or multimodal lists (OpenAI-style content blocks).
Extracted from agent.py — no logic changes.
"""

import base64
import io
import logging
from typing import Any, List

from PIL import Image

logger = logging.getLogger(__name__)


def compress_frame_b64(b64_data: str, max_dim: int = 768, quality: int = 80) -> str:
    """Resize and recompress a base64-encoded image for memory context.

    Reduces 1280x720 JPEGs from ~270KB to ~60-80KB base64.
    Returns original data on failure.
    """
    try:
        raw = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw))
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        compressed = base64.b64encode(buf.getvalue()).decode()
        if len(compressed) < len(b64_data):
            return compressed
        return b64_data  # Original was smaller (already compressed)
    except Exception as e:
        logger.debug(f"Frame compression failed, using original: {e}")
        return b64_data


def append_to_content(content, text: str):
    """Append text to message content, handling both str and multimodal list formats."""
    if isinstance(content, list):
        # Find last text block and append, or add a new one
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] += text
                return content
        content.append({"type": "text", "text": text})
        return content
    return content + text


def replace_in_content(content, old: str, new: str):
    """Replace text in message content, handling both str and multimodal list formats."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = block["text"].replace(old, new)
        return content
    return content.replace(old, new)


def content_to_text(content) -> str:
    """Extract text from message content (str or multimodal list)."""
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


def strip_old_images(messages: list, keep_recent: int = 2) -> list:
    """Strip image content from older multimodal user messages.

    Keeps images only in the last *keep_recent* multimodal user messages
    to prevent token bloat. Returns a NEW list (does not mutate in-place).
    """
    # Find indices of user messages with multimodal (list) content
    mm_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    if keep_recent <= 0:
        # keep_recent=0 means "strip images from ALL multimodal user messages".
        # Note: Python list[:-0] returns [] (not all elements), so handle explicitly.
        strip_set = set(mm_indices)
        if not strip_set:
            return messages
    else:
        if len(mm_indices) <= keep_recent:
            return messages  # Nothing to strip
        strip_set = set(mm_indices[:-keep_recent])
    result = []
    for i, msg in enumerate(messages):
        if i in strip_set:
            # Replace multimodal content with text-only
            text_parts = [
                p.get("text", "") for p in msg["content"]
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            result.append({**msg, "content": "\n".join(text_parts)})
        else:
            result.append(msg)
    return result


def extract_text_from_result(result):
    """Extract text string from a tool result (may be str or multimodal list)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(
            p.get("text", "") for p in result if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(result)


def extract_images_from_result(result):
    """Extract visual content parts (images and video) from a multimodal tool result."""
    if not isinstance(result, list):
        return []
    return [p for p in result if isinstance(p, dict) and p.get("type") in ("image_url", "video_url")]


def build_log_content(text_result: str, image_parts: list) -> Any:
    """Build log content for trajectory, including visual metadata if present."""
    if not image_parts:
        return text_result
    attachments = []
    for p in image_parts:
        ptype = p.get("type", "")
        if ptype == "video_url":
            url = p.get("video_url", {}).get("url", "")
            fps = p.get("fps")
            attachments.append({"type": "video", "url": url, "fps": fps})
        elif ptype == "image_url":
            attachments.append({"type": "image"})
    return {"text": text_result, "attachments": attachments}
