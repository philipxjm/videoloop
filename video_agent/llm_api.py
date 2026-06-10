"""Shared LLM API helpers.

Provides:
- ``gemini_completion()`` — Gemini native ``generateContent`` endpoint
- ``openai_completion()`` — OpenAI-compatible ``/v1/chat/completions`` endpoint

Both return the same normalized dict format so callers can switch freely.
"""

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Official Google Gemini API endpoints. Any OpenAI-compatible endpoint can be
# substituted via the ``api_base`` argument / config field.
GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com"
GEMINI_OPENAI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

# Error-body substrings that mean "don't bother retrying".
_NON_TRANSIENT = (
    "not a valid model", "model not found", "authentication",
    "invalid api key", "unauthorized", "permission",
    "inappropriate content",
    "token count exceeds", "exceeds the maximum number of tokens",
    "too many images",
)

_EMPTY_RESULT = {"content": "", "thinking_content": "", "tool_calls": [],
                 "finish_reason": "stop", "usage": {}}


def strip_provider_prefix(model: str) -> str:
    """Strip a leading ``provider/`` routing prefix from a model name."""
    for prefix in ("openai/", "anthropic/", "google/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _backoff(attempt: int) -> float:
    return 2.0 * (2 ** attempt) + random.uniform(0, 2)


def _post_with_retry(url, headers, payload, timeout, max_retries, provider, parse):
    """POST ``payload`` with exponential-backoff retry, shared by both providers.

    ``parse(result_json) -> dict | None`` interprets a successful response:
    return the normalized dict, return ``None`` to retry (e.g. empty
    candidates/choices), or raise ``RuntimeError`` to abort without retrying.
    """
    for attempt in range(max_retries):
        is_last = attempt >= max_retries - 1
        try:
            resp = httpx.post(
                url, headers=headers, json=payload,
                timeout=httpx.Timeout(timeout, connect=30.0, read=timeout),
            )
            if resp.status_code >= 400:
                error_body = resp.text[:500]
                if any(kw in error_body.lower() for kw in _NON_TRANSIENT):
                    logger.error(f"{provider} non-transient error: {resp.status_code} {error_body}")
                    raise RuntimeError(f"{provider} API error {resp.status_code}: {error_body}")
                if not is_last:
                    logger.warning(f"{provider} retry {attempt+1}/{max_retries}: {resp.status_code} {error_body[:200]}")
                    time.sleep(_backoff(attempt))
                    continue
                raise RuntimeError(f"{provider} API error {resp.status_code} after {max_retries} retries: {error_body}")

            parsed = parse(resp.json())
            if parsed is not None:
                return parsed
            if not is_last:
                logger.warning(f"{provider} empty response, retry {attempt+1}/{max_retries}")
                time.sleep(_backoff(attempt))
                continue
            raise RuntimeError(f"{provider} returned no usable response after {max_retries} retries")

        except httpx.TimeoutException as e:
            # Only retry timeouts once — if the API hangs twice, it's not coming back.
            if attempt < 1:
                logger.warning(f"{provider} timeout retry {attempt+1}/1: {e}")
                time.sleep(5.0 + random.uniform(0, 5))
                continue
            logger.error(f"{provider} API timeout after {attempt+1} attempts — giving up")
            raise RuntimeError(f"{provider} API timeout after {timeout}s (tried {attempt+1} times)") from e
        except RuntimeError:
            raise
        except Exception as e:
            if not is_last:
                logger.warning(f"{provider} error retry {attempt+1}/{max_retries}: {e}")
                time.sleep(_backoff(attempt))
                continue
            raise
    return dict(_EMPTY_RESULT)


def gemini_completion(
    model: str,
    messages: List[Dict],
    api_base: str,
    api_key: str,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    timeout: float = 60.0,
    tools: Optional[List[Dict]] = None,
    max_retries: int = 8,
) -> Dict:
    """Call the Gemini native generateContent API.

    Converts OpenAI-style messages to Gemini format. ``api_base`` may be left
    empty to use the official endpoint. ``thinking_budget`` (token count) and
    ``thinking_level`` ("low"/"high") are mutually exclusive ways to control
    extended thinking, matching the two conventions across Gemini generations.

    Returns dict with keys:
        - content: str (response text)
        - thinking_content: str (reasoning trace, if thinking enabled)
        - tool_calls: list of tool call dicts
        - finish_reason: str (stop, length, content_filter)
        - usage: dict (promptTokenCount, candidatesTokenCount, thoughtsTokenCount, totalTokenCount)
    """
    from uuid import uuid4

    # ── Convert messages to Gemini format ──────────────────────────────
    system_instruction = None
    contents = []
    # Build tool_call_id → tool_name lookup
    tc_id_to_name = {}
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            if tc.get("id") and func.get("name"):
                tc_id_to_name[tc["id"]] = func["name"]

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content
            continue

        if role == "user":
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block["text"]})
                        elif block.get("type") == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                mime, _, b64_data = url.partition(";base64,")
                                mime = mime.replace("data:", "")
                                parts.append({"inline_data": {"mime_type": mime, "data": b64_data}})
                        elif block.get("type") == "video_url":
                            url = block.get("video_url", {}).get("url", "")
                            if url.startswith("data:"):
                                mime, _, b64_data = url.partition(";base64,")
                                mime = mime.replace("data:", "")
                                parts.append({"inline_data": {"mime_type": mime, "data": b64_data}})
                    elif isinstance(block, str):
                        parts.append({"text": block})
            else:
                parts = [{"text": content}] if content else []
            gem_role = "user"

        elif role == "assistant":
            parts = []
            if content:
                parts.append({"text": content})
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                fc_part = {"functionCall": {"name": func.get("name", ""), "args": args}}
                if tc.get("thought_signature"):
                    fc_part["thoughtSignature"] = tc["thought_signature"]
                parts.append(fc_part)
            gem_role = "model"

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "") or tc_id_to_name.get(tc_id, "unknown")
            raw = msg.get("content", "")
            if isinstance(raw, list):
                result_parts = []
                for block in raw:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            result_parts.append({"text": block["text"]})
                        elif block.get("type") == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                mime, _, b64_data = url.partition(";base64,")
                                mime = mime.replace("data:", "")
                                result_parts.append({"inline_data": {"mime_type": mime, "data": b64_data}})
                parts = [{"functionResponse": {"name": tool_name, "response": {"parts": result_parts}}}]
            else:
                try:
                    response_data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": raw}
                parts = [{"functionResponse": {"name": tool_name, "response": response_data}}]
            gem_role = "user"
        else:
            continue

        if not parts:
            continue

        # Merge consecutive same-role messages
        if contents and contents[-1]["role"] == gem_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gem_role, "parts": parts})

    # ── Build payload ──────────────────────────────────────────────────
    gen_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    thinking_config = {}
    if thinking_level is not None:
        thinking_config = {"includeThoughts": True, "thinkingLevel": thinking_level.upper()}
    elif thinking_budget is not None:
        thinking_config = {"includeThoughts": True, "thinkingBudget": thinking_budget}
    if thinking_config:
        gen_config["thinkingConfig"] = thinking_config

    # Strip /v1 suffix for native endpoint
    base = (api_base or GEMINI_NATIVE_BASE).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    bare_model = strip_provider_prefix(model)

    url = f"{base}/v1beta/models/{bare_model}:generateContent"

    payload = {"contents": contents, "generationConfig": gen_config}
    if tools:
        # Accept both OpenAI format and Gemini native format
        func_decls = []
        for t in tools:
            if "function" in t:
                # OpenAI format: {"type": "function", "function": {...}}
                func_decls.append(t["function"])
            elif "functionDeclarations" in t:
                # Gemini native format: {"functionDeclarations": [...]}
                func_decls.extend(t["functionDeclarations"])
            elif "function_declarations" in t:
                # Snake_case variant
                func_decls.extend(t["function_declarations"])
        if func_decls:
            payload["tools"] = [{"function_declarations": func_decls}]
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    # The official endpoint authenticates via the x-goog-api-key header.
    # OpenAI-style gateways generally expect a Bearer token; send both when a
    # custom api_base is configured.
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    if api_base:
        headers["Authorization"] = f"Bearer {api_key}"

    finish_map = {
        "STOP": "stop", "MAX_TOKENS": "length",
        "SAFETY": "content_filter", "RECITATION": "content_filter",
        "FINISH_REASON_UNSPECIFIED": "stop",
    }

    def parse(result):
        candidates = result.get("candidates", [])
        if not candidates:
            block_reason = result.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise RuntimeError(f"Gemini blocked request: {block_reason}")
            return None  # retry
        parts = candidates[0].get("content", {}).get("parts", [])
        finish_reason_raw = candidates[0].get("finishReason", "STOP")
        text_parts, thinking_parts, tool_calls = [], [], []
        for part in parts:
            if part.get("thought"):
                thinking_parts.append(part.get("text", ""))
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                    "thought_signature": part.get("thoughtSignature", ""),
                })
            elif "text" in part:
                text_parts.append(part["text"])
        return {
            "content": "\n".join(text_parts) if text_parts else "",
            "thinking_content": "\n".join(thinking_parts) if thinking_parts else "",
            "tool_calls": tool_calls,
            "finish_reason": finish_map.get(finish_reason_raw, "stop"),
            "usage": result.get("usageMetadata", {}),
        }

    return _post_with_retry(url, headers, payload, timeout, max_retries, "Gemini", parse)


def openai_completion(
    model: str,
    messages: List[Dict],
    api_base: str,
    api_key: str,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    timeout: float = 60.0,
    tools: Optional[List[Dict]] = None,
    max_retries: int = 8,
    reasoning_effort: Optional[str] = None,
) -> Dict:
    """Call an OpenAI-compatible /chat/completions endpoint.

    Messages should already be in OpenAI format (role/content/tool_calls).
    ``api_base`` may be left empty to use the Gemini API's OpenAI-compatible
    endpoint. ``reasoning_effort`` ("low"/"medium"/"high") controls extended
    thinking on endpoints that support it.

    Returns same dict format as gemini_completion:
        - content: str (response text)
        - thinking_content: str (reasoning, if present)
        - tool_calls: list of tool call dicts
        - finish_reason: str (stop, length, content_filter)
        - usage: dict (prompt_tokens, completion_tokens, total_tokens)
    """
    from uuid import uuid4

    bare_model = strip_provider_prefix(model)

    base = (api_base or GEMINI_OPENAI_COMPAT_BASE).rstrip("/")
    url = f"{base}/chat/completions"

    payload = {
        "model": bare_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if tools:
        # Normalize tool format
        oai_tools = []
        for t in tools:
            if "function" in t and "type" in t:
                oai_tools.append(t)
            elif "function" in t:
                oai_tools.append({"type": "function", "function": t["function"]})
            elif "functionDeclarations" in t:
                for fd in t["functionDeclarations"]:
                    oai_tools.append({"type": "function", "function": fd})
            elif "function_declarations" in t:
                for fd in t["function_declarations"]:
                    oai_tools.append({"type": "function", "function": fd})
        if oai_tools:
            payload["tools"] = oai_tools

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def parse(result):
        choices = result.get("choices", [])
        if not choices:
            return None  # retry
        message = choices[0].get("message", {})
        usage = result.get("usage", {})
        tool_calls = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append({
                "id": tc.get("id", f"call_{uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                },
                "thought_signature": "",
            })
        return {
            "content": message.get("content", "") or "",
            # Some endpoints expose reasoning as reasoning_content
            "thinking_content": message.get("reasoning_content", "") or "",
            "tool_calls": tool_calls,
            "finish_reason": choices[0].get("finish_reason", "stop") or "stop",
            # Normalize usage to match gemini_completion format
            "usage": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "thoughtsTokenCount": 0,  # Not reported separately in OpenAI format
                "totalTokenCount": usage.get("total_tokens", 0),
            },
        }

    return _post_with_retry(url, headers, payload, timeout, max_retries, "OpenAI", parse)
