"""
API call mixin for VideoUnderstandingAgent.

Provides normalized LLM API calling via Gemini native or OpenAI-compatible,
returning a uniform response dict regardless of provider.
"""

import json
import time
import random

from .logging_utils import get_logger

logger = get_logger(__name__)


def _normalize_tool_calls(raw_tool_calls):
    """Convert raw provider tool-call dicts into ``_NormalizedToolCall`` objects."""
    from .agent import _NormalizedToolCall
    out = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        out.append(_NormalizedToolCall(
            id=tc.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
            thought_signature=tc.get("thought_signature", ""),
        ))
    return out


class APICallMixin:
    """Mixin providing _dispatch_api_call and provider-specific helpers.

    Expects the host class to have:
      - self.model, self._bare_model, self.api_base, self.api_key
      - self._model_limits (.max_output_tokens)
      - self._temperature, self._use_thinking, self._thinking_budget
      - self._use_gemini_native, self._timeout
      - self.token_tracker  (TokenTracker instance)
      - self._parse_tool_call_from_content(content)
    """

    def _call_and_normalize_gemini_native(self, messages, tools):
        """Call Gemini native API via shared gemini_completion and normalize for agent loop."""
        from .llm_api import gemini_completion

        resp = gemini_completion(
            model=self.model,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=self._model_limits.max_output_tokens,
            temperature=self._temperature,
            thinking_budget=getattr(self, '_thinking_budget', None),
            thinking_level=getattr(self, '_thinking_level', None),
            timeout=self._timeout,
            tools=tools,
            max_retries=8,
        )

        usage = resp.get("usage", {})
        input_tokens = usage.get("promptTokenCount", 0) or 0
        output_tokens = (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0)
        thoughts_tokens = usage.get("thoughtsTokenCount", 0) or 0
        self.token_tracker.record(
            agent_type="main_agent",
            model=self._bare_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_site="gemini_native",
        )
        logger.info(
            f"[MainAgent] API call: prompt={input_tokens:,} output={output_tokens:,} "
            f"thinking={thoughts_tokens:,} candidates={output_tokens - thoughts_tokens:,}"
        )

        return {
            "text_content": resp.get("content") or None,
            "thinking_content": resp.get("thinking_content") or None,
            "tool_calls": _normalize_tool_calls(resp.get("tool_calls", [])),
            "finish_reason": resp.get("finish_reason", "stop"),
            "raw_message": None,
        }

    def _call_and_normalize_openai(self, messages, tools):
        """Call OpenAI-compatible API via shared openai_completion."""
        from .llm_api import openai_completion
        from .agent import _NormalizedToolCall

        resp = openai_completion(
            model=self.model,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=self._model_limits.max_output_tokens,
            temperature=self._temperature,
            timeout=self._timeout,
            tools=tools,
            max_retries=8,
        )

        usage = resp.get("usage", {})
        input_tokens = usage.get("promptTokenCount", 0) or 0
        output_tokens = usage.get("candidatesTokenCount", 0) or 0
        self.token_tracker.record(
            agent_type="main_agent",
            model=self._bare_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_site="openai_compat",
        )
        logger.info(
            f"[MainAgent] API call: prompt={input_tokens:,} output={output_tokens:,} "
            f"thinking=? candidates=?"
        )

        tool_calls = _normalize_tool_calls(resp.get("tool_calls", []))

        # Content-based tool call fallback
        if not tool_calls and resp.get("content"):
            parsed = self._parse_tool_call_from_content(resp["content"])
            if parsed:
                raw_args = getattr(parsed.function, 'arguments', '{}')
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                tool_calls = [_NormalizedToolCall(
                    id=getattr(parsed, 'id', ''),
                    name=getattr(parsed.function, 'name', ''),
                    arguments=args,
                )]

        return {
            "text_content": resp.get("content") or None,
            "thinking_content": resp.get("thinking_content") or None,
            "tool_calls": tool_calls,
            "finish_reason": resp.get("finish_reason", "stop"),
            "raw_message": None,
        }

    def _dispatch_api_call(self, messages, tools):
        """Route to the correct provider and return normalized response."""
        if self._use_gemini_native:
            return self._call_and_normalize_gemini_native(messages, tools)
        return self._call_and_normalize_openai(messages, tools)
