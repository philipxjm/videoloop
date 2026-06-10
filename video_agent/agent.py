"""
Main Video Understanding Agent

Native-multimodal agent loop for long-video question answering: the main
agent sees video frames directly, runs tools inside a Docker sandbox, and
persists evidence in LLM-managed working memory.
"""

import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import get_config, ModelLimits
from .config_loader import (
    get_system_prompt,
    get_task_prompt,
    get_no_tool_call_message,
    get_iteration_warning,
    get_forced_answer_message,
    get_early_submit_bounce,
    get_memory_preamble,
    get_focus_nudge_template,
    get_summarizer_prompts,
    get_orchestrator_prompts,
    get_orchestrated_preamble,
)
from .memory import MemoryOrchestrator, FilesystemContextAssembler, truncate_tool_output
from .file_memory import OrchestratedMemory
from .token_tracker import TokenTracker

# Tokenizer for the no-memory budget check.
# tiktoken cl100k_base is a close approximation for Gemini text-token rate
# (within ~10% across English/code prose); for image_url blocks we add a
# fixed Gemini per-image cost (258 tokens regardless of resolution, per
# https://ai.google.dev/gemini-api/docs/tokens).
try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken should be present
    _TIKTOKEN_ENC = None
_GEMINI_PER_IMAGE_TOKENS = 258


def _count_messages_tokens(llm_messages: List[Dict[str, Any]]) -> int:
    """Estimate prompt-side tokens for a multimodal message list.

    Counts tiktoken cl100k_base for text content and adds a fixed
    `_GEMINI_PER_IMAGE_TOKENS` per `image_url` block. Falls back to a
    chars/4 heuristic if tiktoken isn't available.
    """
    if _TIKTOKEN_ENC is None:
        return sum(len(str(m.get("content", ""))) for m in llm_messages) // 4
    text_tokens = 0
    n_images = 0
    for m in llm_messages:
        content = m.get("content")
        if isinstance(content, str):
            try:
                text_tokens += len(_TIKTOKEN_ENC.encode(content))
            except Exception:
                text_tokens += len(content) // 4
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                ptype = p.get("type")
                if ptype == "text":
                    try:
                        text_tokens += len(_TIKTOKEN_ENC.encode(p.get("text", "")))
                    except Exception:
                        text_tokens += len(p.get("text", "")) // 4
                elif ptype == "image_url":
                    n_images += 1
    return text_tokens + n_images * _GEMINI_PER_IMAGE_TOKENS
from .docker_runtime import DockerRuntime
from .transcript_cache import TranscriptCache
from .tools import format_tools_for_claude, format_tools_for_openai, format_tools_for_gemini
from .logging_utils import get_logger, console, IS_TTY
from .message_utils import (
    append_to_content,
    replace_in_content,
    content_to_text,
    strip_old_images,
    extract_text_from_result,
    extract_images_from_result,
    build_log_content,
)
from .api_calls import APICallMixin

# Base tools that don't consume an iteration step (bookkeeping/free actions)
FREE_TOOLS = {"complete_todo"}


def make_console(log_file: Optional[str] = None) -> Console:
    """Create a console, optionally with file logging."""
    if log_file:
        return Console(record=True, force_terminal=False, no_color=True, markup=False)
    return Console(force_terminal=IS_TTY, no_color=not IS_TTY)


logger = get_logger(__name__)


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory."""
    step_type: str  # "thought", "tool_use", "tool_result", "answer"
    content: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Result from running the agent."""
    answer: Optional[str]
    trajectory: List[TrajectoryStep]
    success: bool
    error: Optional[str] = None
    reasoning: Optional[str] = None
    token_usage: Optional[dict] = None


@dataclass
class _NormalizedToolCall:
    """A tool call normalized across all providers."""
    id: str
    name: str
    arguments: dict  # parsed, not JSON string
    thought_signature: str = ""  # Gemini thinking models require this echoed back


def _detect_repetition(text: str) -> bool:
    """Detect if text contains repetitive patterns (sign of thinking loop).

    Checks for:
    1. Token-level repetition: short tokens repeated many times (e.g., "oct0, oct0, oct0...")
    2. Sentence-level repetition: same sentence appears 3+ times
    """
    if not text or len(text) < 100:
        return False

    # 1. Token-level: find any word/token repeated 20+ times in a row
    #    Catches degenerate outputs like "oct0, oct0, oct0..."
    token_match = re.search(r'(\b\w{2,10}\b)(?:[,\s]+\1){19,}', text[:5000])
    if token_match:
        return True

    # 2. Sentence-level: same 30+ char sentence appears 3+ times
    sentences = re.split(r'[.!?\n]', text)
    seen = {}
    for s in sentences:
        s = s.strip().lower()
        if len(s) >= 30:
            seen[s] = seen.get(s, 0) + 1
            if seen[s] >= 3:
                return True

    # 3. Ratio check: if text is very long but has very low unique content
    if len(text) > 10000:
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        if total_words > 100 and unique_words / total_words < 0.05:
            return True

    return False


@dataclass
class _IterationState:
    """Mutable state carried across iterations of the main agent loop."""
    consecutive_no_tool_calls: int = 0
    consecutive_dedup_blocks: int = 0
    actual_step: int = 0
    recent_tools: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    reasoning: Optional[str] = None
    overthink_warnings: int = 0  # How many times we've injected an overthink nudge


# Load NO_TOOL_CALL_MSG from config (loaded lazily on first use)
_NO_TOOL_CALL_MSG_CACHE = None

def _get_no_tool_call_msg() -> str:
    """Get the no-tool-call reminder message (cached)."""
    global _NO_TOOL_CALL_MSG_CACHE
    if _NO_TOOL_CALL_MSG_CACHE is None:
        _NO_TOOL_CALL_MSG_CACHE = get_no_tool_call_message()
    return _NO_TOOL_CALL_MSG_CACHE


class VideoUnderstandingAgent(APICallMixin):
    """
    Main agent for video understanding tasks.

    Native multimodal: the agent reasons over video frames directly and
    orchestrates tools that execute in a Docker sandbox with video
    processing capabilities. Working memory is curated by an LLM
    orchestrator after every tool call.
    """

    @classmethod
    def get_system_prompt(cls, max_images: int = 32) -> str:
        """Generate system prompt with dynamic configuration values from configs/prompts.yaml."""
        return get_system_prompt(max_images=max_images)

    # Supported providers and their model prefixes
    PROVIDER_PREFIXES = {
        "anthropic": "claude",
        "openai": "gpt",
        "google": "gemini",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        docker_image: Optional[str] = None,
        max_iterations: int = 50,
        verbose: bool = True,
        debug: bool = False,
        api_base: Optional[str] = None,
        log_file: Optional[str] = None,
        step_callback: Optional[Callable[[str, Any, List], None]] = None,
        no_memory: bool = False,
        include_tools: list = None,  # Whitelist of tool names; None/empty = all tools
        orchestrator_model: Optional[str] = None,  # Override orchestrator/summarizer model
        multimodal: Optional[bool] = None,  # Override native multimodal mode
        max_ctx_tokens: Optional[int] = None,  # Override Layer 3 context budget (no_memory mode); default 500_000
        raw_context: bool = False,  # No-memory ablation: disable layered truncations so only max_ctx_tokens applies. Requires no_memory=True.
        early_submit_gate: bool = True,  # When False, _check_early_submit always returns None (transcript / min-iters / min-VLM-calls / open-todo bounces are all skipped).
    ):
        # Load config and apply defaults
        cfg = get_config()
        self.model = model or cfg.main_agent.model
        # Override orchestrator model if provided
        if orchestrator_model:
            cfg.memory.summarizer_model = orchestrator_model
        # Override multimodal mode if provided
        if multimodal is not None:
            cfg.main_agent.multimodal_agent = multimodal
        self.docker_image = docker_image or cfg.sandbox.full_image
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug
        self.api_base = api_base or cfg.main_agent.api_base
        self.log_file = log_file
        self.step_callback = step_callback  # Callback: (step_type, content, trajectory) -> None
        self._include_tools = include_tools or cfg.include_tools or []

        # Create console for this agent instance (with optional file recording)
        self._console = make_console(log_file)

        # Resolve token limits and temperature from config
        self._model_limits = ModelLimits(
            max_output_tokens=cfg.main_agent.max_output_tokens,
            temperature=cfg.main_agent.temperature if cfg.main_agent.temperature is not None else 0.0,
        )
        self._temperature = self._model_limits.temperature

        # --- API key/base resolution ---
        # Explicit arg > component env var > shared fallback keys.
        # When no api_base is set, Gemini models call the official
        # generativelanguage.googleapis.com endpoint; any OpenAI-compatible
        # endpoint can be supplied via api_base / MAIN_AGENT_API_BASE.
        if not self.api_base and os.environ.get("MAIN_AGENT_API_BASE"):
            self.api_base = os.environ["MAIN_AGENT_API_BASE"]
        self.api_key = (
            api_key
            or os.environ.get("MAIN_AGENT_API_KEY")
            or self._default_api_key()
        )

        # Memory orchestrator credentials (default to the main agent's)
        self._subagent_api_base = os.environ.get("SUMMARIZER_API_BASE") or self.api_base
        self._subagent_api_key = os.environ.get("SUMMARIZER_API_KEY") or self.api_key

        # Determine provider from model name
        self.provider = self._get_provider(self.model)

        # Strip provider prefix from model name for native API calls
        from .llm_api import strip_provider_prefix
        self._bare_model = strip_provider_prefix(self.model)
        # Extended thinking comes from config
        self._use_thinking = bool(getattr(cfg.main_agent, 'thinking', False))

        # Timeout and retry config
        self._timeout = cfg.main_agent.timeout  # seconds
        self._max_retries = cfg.main_agent.max_retries

        # Docker runtime (created per-run)
        self._docker: Optional[DockerRuntime] = None

        # Max images per visual-analysis call (bounds tool schema + payload)
        self.max_images = cfg.main_agent.max_images
        # Token usage tracking
        self.token_tracker = TokenTracker()

        # Trajectory tracking
        self.trajectory: List[TrajectoryStep] = []
        
        # Hierarchical memory (VideoARM-inspired)
        self._memory_enabled = cfg.memory.enabled and not no_memory
        # Layer 3 context budget for no_memory (LLM-in-Sandbox) mode. Default 500K tokens
        # to match Gemini's 1M context. Override for ablations that test tighter budgets.
        self._max_ctx_tokens = max_ctx_tokens if max_ctx_tokens is not None else 500_000
        # raw_context ablation: when on AND no_memory is on, disable layered truncations
        # (Layer 1 tool-output cap, Layer 2 image strip, Layer 4 overthink trim) so the
        # only ceiling is _max_ctx_tokens. Has no effect when memory is enabled — keeps
        # configs B/C byte-identical.
        self._raw_no_trunc = bool(raw_context) and (not self._memory_enabled)
        # Early-submit gate toggle. Default True (production behavior).
        # When False, _check_early_submit returns None unconditionally — used in
        # ablations that test how the gate shapes trajectory length.
        self._early_submit_gate = bool(early_submit_gate)
        self._memory_config = cfg.memory
        self._visual_planner_config = cfg.visual_planner
        self._project_root = Path(__file__).resolve().parent.parent

        # Temp directory for frame transfers
        self._temp_dir: Optional[str] = None
        
        # Use native Gemini generateContent API for Google models (supports thinking + tool calling)
        self._use_gemini_native = (self.provider == "google")
        # Optional thinking budget / level from config (None = let model decide)
        self._thinking_budget = getattr(cfg.main_agent, 'thinking_budget', None)
        self._thinking_level = getattr(cfg.main_agent, 'thinking_level', None)

        # Native multimodal: images to agent + agent's analysis for memory
        self._multimodal_agent = getattr(cfg.main_agent, 'multimodal_agent', True)
        if not self._multimodal_agent:
            raise ValueError(
                "multimodal_agent=false (the VLM sub-agent pipeline) was removed "
                "in the public release; only the native multimodal agent is supported."
            )

        _format = "gemini-native" if self._use_gemini_native else "openai-compat"
        _mm = ", multimodal_agent=on" if self._multimodal_agent else ""
        logger.info(f"Agent initialized: model={self._bare_model}, provider={self.provider}, "
                     f"format={_format}{_mm}")
    
    def _log_print(self, *args, **kwargs):
        """Print to console and optionally record for log file export."""
        # Print to instance console (displays on terminal and records if log_file set)
        self._console.print(*args, **kwargs)
    
    def _flush_log(self):
        """Write recorded console output to log file."""
        if self.log_file and hasattr(self._console, 'export_text'):
            try:
                text = self._console.export_text()
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(text)
            except Exception:
                pass
    
    def _get_provider(self, model: str) -> str:
        """Determine the provider from model name."""
        model_lower = model.lower()
        # Explicit provider prefixes first
        if model_lower.startswith("openai/"):
            return "openai"  # Any OpenAI-compatible API
        elif model_lower.startswith("anthropic/"):
            return "anthropic"
        elif model_lower.startswith("gemini/") or model_lower.startswith("google/"):
            return "google"
        # Fallback to model name heuristics
        elif "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "google"
        else:
            return "unknown"

    @staticmethod
    def _default_api_key() -> str:
        """Shared fallback key resolution when MAIN_AGENT_API_KEY is not set."""
        return (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )

    def _get_tools_for_provider(self) -> list:
        """Get tools formatted for the current provider."""
        inc = self._include_tools or None
        if self._use_gemini_native:
            return format_tools_for_gemini(max_images=self.max_images, include_tools=inc)
        elif self.provider == "anthropic":
            return format_tools_for_claude(max_images=self.max_images, include_tools=inc)
        else:
            # OpenAI-style format works for most providers including Ollama
            return format_tools_for_openai(max_images=self.max_images, include_tools=inc)

    def _parse_tool_call_from_content(self, content: str) -> Optional[Any]:
        """
        Parse a tool call from text content.
        Some models (Ollama/Qwen/Gemini) output tool calls as JSON or
        Python function-call syntax in text instead of using proper tool_calls.
        """
        import re
        from types import SimpleNamespace

        content = content.strip()

        # 1. Try Python function-call syntax: submit_answer(answer="C", reasoning="...")
        #    This is common with Gemini via OpenAI-compatible APIs.
        func_match = re.search(
            r'(\w+)\s*\(\s*((?:\w+\s*=\s*(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,)]+)\s*,?\s*)*)\)',
            content,
        )
        if func_match:
            func_name = func_match.group(1)
            # Only parse known tool names to avoid false positives
            known_tools = {
                "submit_answer",
                "transcribe_audio", "analyze_frames", "analyze_clip",
                "execute_bash", "create_file",
                "list_files", "read_file",
                "pin_memory", "complete_todo",
            }
            if func_name in known_tools:
                args_str = func_match.group(2)
                tool_args = {}
                # Parse keyword arguments: key="value" or key='value'
                for kv in re.finditer(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|([^,)]+))', args_str):
                    key = kv.group(1)
                    val = kv.group(2) if kv.group(2) is not None else (kv.group(3) if kv.group(3) is not None else kv.group(4).strip())
                    tool_args[key] = val
                if tool_args:
                    return SimpleNamespace(
                        id=f"call_{func_name}",
                        function=SimpleNamespace(
                            name=func_name,
                            arguments=json.dumps(tool_args),
                        )
                    )

        # 2. Try JSON object format: {"function": {"name": ..., "arguments": ...}}
        json_match = re.search(r'\{[^{}]*"function"\s*:\s*\{[^{}]*"name"\s*:[^{}]*\}[^{}]*\}', content)
        if not json_match:
            json_match = re.search(r'\{.*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*\}.*\}', content, re.DOTALL)

        if not json_match:
            return None

        try:
            tool_json = json.loads(json_match.group())

            if "function" in tool_json:
                func = tool_json["function"]
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                tool_id = tool_json.get("id", f"call_{tool_name}")
            else:
                tool_name = tool_json.get("name", "")
                tool_args = tool_json.get("arguments", {})
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                tool_id = f"call_{tool_name}"

            if not tool_name:
                return None

            return SimpleNamespace(
                id=tool_id,
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _log_step(self, step_type: str, content: Any):
        """Log a trajectory step with llm-in-sandbox style output."""
        step = TrajectoryStep(step_type=step_type, content=content)
        self.trajectory.append(step)
        
        # Fire callback for live updates
        if self.step_callback:
            try:
                self.step_callback(step_type, content, self.trajectory)
            except Exception:
                pass  # Don't fail the agent if callback fails
        
        if self.verbose:
            if step_type == "iteration":
                # Step header with rule line (llm-in-sandbox style)
                self._console.print()
                self._console.rule(
                    f"[bold blue]Step {content['number']}/{content['max']}[/bold blue]",
                    style="blue"
                )
            elif step_type == "thought":
                # Reasoning/thinking content (magenta panel with 🧠 or 💭)
                thought_display = str(content)[:2000] + "..." if len(str(content)) > 2000 else str(content)
                self._console.print(Panel(
                    thought_display,
                    title="[bold magenta]💭 THOUGHT[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                ))
            elif step_type == "tool_use":
                # Action panel (yellow with ⚡)
                action_text = f"[bold]{content['name']}[/bold]"
                if content.get('input'):
                    params_str = json.dumps(content['input'], indent=2, ensure_ascii=False)
                    if len(params_str) > 500:
                        params_str = params_str[:500] + "..."
                    action_text += f"\n{params_str}"
                self._console.print(Panel(
                    action_text,
                    title="[bold yellow]⚡ ACTION[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                ))
            elif step_type == "tool_result":
                # Observation panel (green with 👁)
                obs_display = str(content)[:800] + "..." if len(str(content)) > 800 else str(content)
                self._console.print(Panel(
                    obs_display,
                    title="[bold green]👁 OBSERVATION[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                ))
            elif step_type == "memory":
                # Memory state panel (orange)
                self._console.print(Panel(
                    str(content),
                    title="[bold #ffb347]🧠 MEMORY[/bold #ffb347]",
                    border_style="#ffb347",
                    padding=(0, 1),
                ))
            elif step_type == "answer":
                # Final answer panel (cyan with 📄)
                answer_text = f"[bold]{content['answer']}[/bold]"
                if content.get('reasoning'):
                    reasoning_display = content['reasoning'][:500] + "..." if len(content.get('reasoning', '')) > 500 else content.get('reasoning', '')
                    answer_text += f"\n\n{reasoning_display}"
                self._console.print()
                self._console.print(Panel(
                    answer_text,
                    title="[bold cyan]📄 ANSWER[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                ))

    def _copy_frame_to_host(self, container_path: str) -> Optional[str]:
        """Copy a frame from container to host for VLM processing."""
        import tempfile
        
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="video_agent_")
        
        filename = os.path.basename(container_path)
        local_path = os.path.join(self._temp_dir, filename)
        
        if self._docker.copy_from_container(container_path, local_path):
            # The file is extracted with original name
            extracted_path = os.path.join(self._temp_dir, filename)
            if os.path.exists(extracted_path):
                return extracted_path
        
        return None

    def _fix_video_path(self, path: str) -> str:
        """Convert host paths to container paths.
        
        Some models ignore instructions and use host paths instead of container paths.
        This method fixes common patterns like /home/*/videos/*.mp4 -> /videos/*.mp4
        """
        if not path:
            return path
        # If path looks like a host path (contains /home/ or /work/ etc), extract just the filename
        if "/home/" in path or "/work/" in path or "/tmp/" in path:
            filename = os.path.basename(path)
            fixed = f"/videos/{filename}"
            logger.debug(f"Fixed video path: {path} -> {fixed}")
            return fixed
        return path

    def _build_memory_messages(
        self,
        system_prompt: str,
        user_task: str,
        full_messages: List[Dict],
        memory,
        iteration: int,
    ) -> List[Dict]:
        """Build compact message list with memory context injected.

        Replaces the growing raw message history with:
        system → user_task → <MEMORY> block → [last N raw message pairs]
        """
        # Render memory context (may be str or list of content blocks with images)
        memory_content = memory.render_context(include_working=True)

        # Add focus nudge if there are unexplored intervals
        unexplored = memory.get_unexplored_intervals()
        focus_nudge = ""
        if unexplored and memory.focus_intervals:
            nudge_template = get_focus_nudge_template()
            if nudge_template:
                focus_nudge = "\n" + nudge_template.format(
                    focus_intervals=memory.focus_intervals,
                    unexplored=unexplored,
                )

        task_suffix = (
            f"\n\n[TASK — Do NOT echo or repeat the MEMORY JSON above. "
            f"Reason about your next action and call a tool.]\n"
            f"{user_task}"
        )

        # Merge memory + task into a SINGLE user message.
        # Handle both plain text and multimodal content blocks.
        question_images = getattr(self, '_question_images', [])
        needs_multimodal = isinstance(memory_content, list) or question_images

        if needs_multimodal:
            if isinstance(memory_content, list):
                combined_user = list(memory_content)
            else:
                combined_user = [{"type": "text", "text": memory_content}]
            combined_user.append({
                "type": "text",
                "text": focus_nudge + task_suffix,
            })
            # Re-inject question images if memory didn't already include them
            # (OrchestratedMemory includes them in render_context; others don't)
            memory_has_images = isinstance(memory_content, list) and any(
                b.get("type") == "image_url" for b in memory_content
            )
            if question_images and not memory_has_images:
                combined_user.append({
                    "type": "text",
                    "text": (
                        "[QUESTION IMAGE(S) — These are part of the QUESTION, not from the video. "
                        "Analyze them directly. Do NOT search for them in the video.]"
                    ),
                })
                for img_b64 in question_images:
                    combined_user.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
        else:
            # Plain text (no key frames, no question images)
            combined_user = memory_content + focus_nudge + task_suffix

        result = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_user},
            # Synthetic assistant ack to maintain user/assistant alternation
            {"role": "assistant", "content": "Understood. Continuing analysis based on my memory."},
        ]

        # Recent window: last N "groups" from full history.
        # A group = one assistant message + ALL consecutive tool/user messages
        # that follow it (there can be multiple tool results when the model
        # calls several tools in one turn, e.g. get_video_info + pin_memory).
        window_size = self._memory_config.recent_messages_window

        # Build groups by walking forward through messages (skip system[0], user[1])
        groups: List[List[Dict]] = []
        current_group: List[Dict] = []
        for msg in full_messages[2:]:
            if msg.get("role") == "assistant":
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
            elif current_group:
                # tool or user message belongs to current group
                current_group.append(msg)
            # else: stray message before first assistant — skip
        if current_group:
            groups.append(current_group)

        # Take the last N groups
        for group in groups[-window_size:]:
            result.extend(group)

        # Telemetry: log message sizes
        memory_chars = sum(len(str(b.get("text", ""))) for b in memory_content if isinstance(b, dict)) if isinstance(memory_content, list) else len(str(memory_content))
        img_count = sum(1 for b in memory_content if isinstance(b, dict) and b.get("type") == "image_url") if isinstance(memory_content, list) else 0
        recent_msgs = len(result) - 3  # subtract system + user + ack
        logger.info(
            f"[MainAgent] context: memory={memory_chars} chars, {img_count} images, "
            f"recent_msgs={recent_msgs}, total_msgs={len(result)}"
        )

        return result

    # Delegate to module-level functions from message_utils
    _append_to_content = staticmethod(append_to_content)
    _replace_in_content = staticmethod(replace_in_content)
    _content_to_text = staticmethod(content_to_text)
    _strip_old_images = staticmethod(strip_old_images)
    _extract_text_from_result = staticmethod(extract_text_from_result)
    _extract_images_from_result = staticmethod(extract_images_from_result)
    _build_log_content = staticmethod(build_log_content)

    def _execute_tool(self, name: str, input_data: Dict[str, Any]):
        """Execute a tool and return the result (str or multimodal list)."""
        from . import tool_handlers
        try:
            # Fix video paths that models might hallucinate
            if "video_path" in input_data:
                original_path = input_data["video_path"]
                input_data["video_path"] = self._fix_video_path(input_data["video_path"])
                if original_path != input_data["video_path"]:
                    logger.debug(f"PATH FIX: {original_path} -> {input_data['video_path']}")

            return tool_handlers.dispatch(self, name, input_data)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing {name}: {str(e)}"

    @staticmethod
    def _extract_answer_from_text(text: str, open_ended: bool = False) -> Optional[str]:
        """Try to extract an answer from free-form text.

        For multiple-choice: extracts a letter (A-D).
        For open-ended: extracts a number or short value.
        """
        import re
        if not text:
            return None
        text = text.strip()

        if open_ended:
            # Try to extract a numeric answer
            for pat in [
                r"(?:answer|result|value)\s*(?:is|=|:)\s*([^\s,.\n]+)",
                r"(?:submit_answer|submitting)\s*\(?[^)]*answer[\"':\s]+([^\s,\"'}\n]+)",
                r"=\s*(\d+(?:\.\d+)?)\s*$",
            ]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    return m.group(1)
            return None

        # Multiple-choice extraction
        # Direct single letter
        if text.upper() in ("A", "B", "C", "D", "E"):
            return text.upper()
        # Common patterns
        for pat in [
            r"(?:answer|submit|choice|select)[:\s]+([A-E])\b",
            r"\b([A-E])\.",
            r"\b([A-E])\)",
            r"^([A-E])\b",
            r"\b([A-E])\s*$",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        # Last resort: first standalone A-E
        m = re.search(r"\b([A-E])\b", text)
        if m:
            return m.group(1).upper()
        return None

    # _call_and_normalize_gemini_native, _call_and_normalize_openai,
    # and _dispatch_api_call are inherited from APICallMixin (api_calls.py)

    # ------------------------------------------------------------------
    # Unified response processing (replaces 3 duplicated paths)
    # ------------------------------------------------------------------

    def _process_response(self, resp, messages, memory, free_tools, state, iteration,
                          original_user_task="", tools=None):
        """Process a normalized LLM response. Returns 'break', 'continue', or 'next'."""
        text_content = resp["text_content"]
        thinking_content = resp["thinking_content"]
        all_tool_calls = resp["tool_calls"]
        finish_reason = resp["finish_reason"]
        raw_message = resp.get("raw_message")

        # Truncate degenerate outputs — hard 8K if repetition detected, 32K otherwise
        all_response = (text_content or "") + (thinking_content or "")
        if _detect_repetition(all_response):
            cap = 8000
            logger.warning(f"[Safety] Repetition detected — hard truncating to {cap} chars")
        else:
            cap = 32000
        if text_content and len(text_content) > cap:
            logger.warning(f"[Safety] Truncating text_content from {len(text_content)} to {cap} chars")
            text_content = text_content[:cap]
        if thinking_content and len(thinking_content) > cap:
            logger.warning(f"[Safety] Truncating thinking_content from {len(thinking_content)} to {cap} chars")
            thinking_content = thinking_content[:cap]

        # Log thinking
        if thinking_content and not all_tool_calls:
            self._log_step("thought", thinking_content)
        elif text_content and not all_tool_calls:
            self._log_step("thought", text_content)

        # Detect overthinking: repetitive text or finish_reason=length (hit output cap)
        all_text = (text_content or "") + (thinking_content or "")
        is_overthinking = False
        if finish_reason == "length" and len(all_text) > 5000:
            is_overthinking = True
            logger.warning(f"[Overthink] Response hit output token limit (finish_reason=length)")
        elif _detect_repetition(all_text):
            is_overthinking = True
            logger.warning(f"[Overthink] Repetitive sentences detected in response")

        if is_overthinking:
            state.overthink_warnings += 1
            nudge = (
                "⚠️ You appear to be overthinking this. The question may contain inconsistencies "
                "or ambiguities — that's OK. Do NOT try to resolve every detail perfectly. "
                "Focus on what you DO know, pick the closest answer, and call submit_answer. "
                "If you need more evidence, make ONE targeted tool call — do not re-analyze "
                "intervals you've already examined."
            )
            messages.append({"role": "user", "content": nudge})
            logger.info(f"[Overthink] Injected nudge (warning #{state.overthink_warnings})")

            # After 3 overthink warnings, force submit on next no-tool response
            if state.overthink_warnings >= 3:
                logger.warning(f"[Overthink] 3+ warnings — will force answer on next opportunity")

        # Separate free vs step-consuming tools
        free_calls = [tc for tc in all_tool_calls if tc.name in free_tools]
        step_calls = [tc for tc in all_tool_calls if tc.name not in free_tools]

        # Build assistant message with ALL tool calls
        assistant_msg = {"role": "assistant", "content": text_content or ""}
        if raw_message and hasattr(raw_message, 'reasoning_content') and raw_message.reasoning_content:
            assistant_msg["reasoning_content"] = raw_message.reasoning_content
        if all_tool_calls:
            tc_list = []
            for tc in all_tool_calls:
                tc_dict = {"id": tc.id, "type": "function",
                           "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                if tc.thought_signature:
                    tc_dict["thought_signature"] = tc.thought_signature
                tc_list.append(tc_dict)
            assistant_msg["tool_calls"] = tc_list
        messages.append(assistant_msg)

        # Execute free tool calls silently
        _pending_images = []  # Collect images from multimodal results
        for fc in free_calls:
            self._log_step("tool_use", {"name": fc.name, "input": fc.arguments, "free": True})
            result = self._execute_tool(fc.name, fc.arguments)
            text_result = self._extract_text_from_result(result)
            fc_images = self._extract_images_from_result(result)
            self._log_step("tool_result", self._build_log_content(text_result, fc_images))
            logger.info(f"{fc.name}: {text_result[:200]}")
            # Inline truncation when memory is disabled (LLM-in-Sandbox mode).
            # Gated off when raw_context ablation is on so only Layer 3 (max_ctx_tokens) applies.
            if not memory and not self._raw_no_trunc:
                text_result = re.sub(
                    r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                    '[image-stripped-in-no-memory-mode]',
                    text_result,
                )
                text_result = truncate_tool_output(text_result)
                if len(text_result) > 12000:
                    text_result = text_result[:6000] + "\n...[tool output truncated]...\n" + text_result[-6000:]
            messages.append({"role": "tool", "tool_call_id": fc.id, "content": text_result})
            _pending_images.extend(fc_images)

        # Stub responses for extra step-consuming tools (beyond first)
        for extra_tc in step_calls[1:]:
            messages.append({
                "role": "tool", "tool_call_id": extra_tc.id,
                "content": "Skipped: only one tool executed per step. Call this tool again if needed.",
            })

        # Free-tools-only: continue without consuming iteration
        if free_calls and not step_calls:
            # Inject pending images as a user message (multimodal visual context)
            if _pending_images:
                img_msg_content = [{"type": "text", "text": "[Visual context: keyframes from the analyzed content]"}]
                img_msg_content.extend(_pending_images)
                messages.append({"role": "user", "content": img_msg_content})
            return "continue"

        # No-tool-call handling
        if not step_calls and finish_reason in ("end_turn", "stop", "completed", "length"):
            if finish_reason == "length":
                logger.warning("Response truncated (finish_reason=length)")
            state.consecutive_no_tool_calls += 1

            if state.answer is not None:
                return "break"

            all_text = (text_content or "") + " " + (thinking_content or "")
            is_open_ended = not getattr(self, '_current_options', None)

            if is_open_ended:
                # For open-ended questions, look for numeric or value-based answers
                explicit = re.search(r'(?:answer\s+is|result\s+is|value\s+is|equals?)[:\s]+([^\s,.]+)', all_text, re.IGNORECASE)
            else:
                explicit = re.search(r'(?:answer\s+is|submit|choose|select)\s+([A-D])\b', all_text, re.IGNORECASE)
                if not explicit:
                    explicit = re.search(r'\bSo\s+([A-D])[\.\s]', all_text)

            # Only auto-extract after multiple no-tool turns — give the model a chance to call submit_answer.
            # Also enforce the same min-iteration / min-VLM-calls gates as submit_answer, so "append-only"
            # runs can't short-circuit by emitting the answer as plain text instead of calling the tool.
            if explicit and state.consecutive_no_tool_calls >= 3:
                bounce_reason = self._check_early_submit(state, iteration)
                if bounce_reason:
                    logger.info(f"Auto-extract suppressed at iter {iteration}: {bounce_reason}")
                    messages.append({"role": "user", "content": (
                        f"Stating an answer in text is not enough. {bounce_reason}"
                    )})
                    state.consecutive_no_tool_calls = 0  # Reset so we give the model another chance
                    return "continue"
                state.answer = explicit.group(1).upper() if not is_open_ended else explicit.group(1)
                state.reasoning = "Auto-extracted from model response (model stated answer but did not call submit_answer)"
                logger.info(f"Auto-extracting answer '{state.answer}' from model text")
                self._log_step("answer", {"answer": state.answer, "reasoning": state.reasoning})
                return "break"

            # Escalating nudges to recover the agent
            if state.consecutive_no_tool_calls <= 3:
                # Gentle nudge — remind to call a tool
                messages.append({"role": "user", "content": _get_no_tool_call_msg()})
            elif state.consecutive_no_tool_calls == 4:
                # Clean slate — strip the no-tool assistant messages that went nowhere
                # Keep only messages with tool calls or tool results
                cleaned = []
                stripped = 0
                for msg in messages:
                    role = msg.get("role", "")
                    # Keep system, user, tool messages
                    if role in ("system", "user", "tool"):
                        cleaned.append(msg)
                    elif role == "assistant":
                        # Keep assistant messages that had tool calls
                        has_tools = msg.get("tool_calls") or msg.get("function_call")
                        content = msg.get("content", "")
                        # Also truncate any large assistant content (overthink residue).
                        # Gated off when raw_context ablation is on so only Layer 3 applies.
                        if has_tools:
                            if isinstance(content, str) and len(content) > 4000 and not self._raw_no_trunc:
                                msg = dict(msg)
                                msg["content"] = content[:4000] + "\n[truncated]"
                            cleaned.append(msg)
                        elif isinstance(content, str) and len(content) > 8000 and not self._raw_no_trunc:
                            # Skip bloated no-tool responses entirely
                            stripped += 1
                        else:
                            cleaned.append(msg)
                    else:
                        cleaned.append(msg)
                if stripped > 0:
                    logger.info(f"[Recovery] Stripped {stripped} bloated no-tool messages from context")
                messages.clear()
                messages.extend(cleaned)
                nudge = ("Your previous thinking didn't lead to action. The context has been cleaned up. "
                         "Look at your memory and decide: analyze_frames, execute_bash, transcribe_audio, or submit_answer.")
                messages.append({"role": "user", "content": nudge})
            elif state.consecutive_no_tool_calls <= 7:
                # Stronger nudge with specific suggestions
                nudge = ("You have not called a tool in several turns. You MUST call a tool now. "
                         "Options: analyze_frames to look at video frames, transcribe_audio for the transcript, "
                         "execute_bash to extract frames or run calculations, or submit_answer if you're ready.")
                messages.append({"role": "user", "content": nudge})
            else:
                # Last resort — insist on submit_answer
                force_msg = ("🚨 You have been thinking for too many turns without acting. "
                             "Call submit_answer NOW with your best answer based on what you know."
                             if not is_open_ended else
                             "🚨 You have been thinking for too many turns without acting. "
                             "Call submit_answer NOW with your best computed answer.")
                messages.append({"role": "user", "content": force_msg})
            return "continue"
        else:
            state.consecutive_no_tool_calls = 0

        # Execute first step-consuming tool
        if not step_calls:
            return "next"

        tc = step_calls[0]
        tool_name = tc.name
        tool_input = tc.arguments

        # Dedup check
        dedup_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        if dedup_key in self._tool_call_history and tool_name != "submit_answer":
            state.consecutive_dedup_blocks += 1
            logger.warning(f"Dedup blocked: {tool_name} (iter {iteration}, consecutive={state.consecutive_dedup_blocks})")
            if state.consecutive_dedup_blocks >= 10:
                logger.warning(f"Dedup loop stuck ({state.consecutive_dedup_blocks} blocks) — breaking loop")
                return "break"
            dedup_msg = (
                f"DUPLICATE CALL BLOCKED: You already called {tool_name} with these exact arguments. "
                f"Use different arguments or a different tool."
            )
            if state.consecutive_dedup_blocks >= 5:
                dedup_msg = (
                    "DUPLICATE CALL BLOCKED: You are stuck repeating the same tool calls. "
                    "You MUST call submit_answer NOW with your best guess. Choose A, B, C, or D."
                )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": dedup_msg})
            return "continue"
        self._tool_call_history[dedup_key] = True
        state.consecutive_dedup_blocks = 0

        # --- Early submit bounce (before consuming a step) ---
        if tool_name == "submit_answer":
            bounce_reason = self._check_early_submit(state, iteration)
            if bounce_reason:
                logger.info(f"Early submit bounced at iter {iteration}: {bounce_reason}")
                bounce_msg = get_early_submit_bounce(bounce_reason)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": bounce_msg})
                return "continue"

        state.recent_tools.append(tool_name)
        state.actual_step += 1
        self._log_step("iteration", {"number": state.actual_step, "max": self.max_iterations})
        self._log_step("tool_use", {"name": tool_name, "input": tool_input})

        result = self._execute_tool(tool_name, tool_input)
        text_result = self._extract_text_from_result(result)
        image_parts = self._extract_images_from_result(result)

        # Intercept mp4-in-analyze_frames errors: inject correction and retry iteration
        if tool_name == "analyze_frames" and "No valid image files" in text_result:
            logger.warning(f"[mp4 intercept] Agent passed video file to analyze_frames — injecting correction")
            self._log_step("tool_result", text_result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text_result})
            messages.append({"role": "user", "content": (
                "CRITICAL ERROR: You passed a .mp4 VIDEO file to analyze_frames. This will NEVER work. "
                "You must FIRST extract frames using execute_bash with ffmpeg, THEN pass the .jpg files:\n"
                "Step 1: execute_bash({\"command\": \"ffmpeg -ss 120 -i /videos/video.mp4 -frames:v 1 /outputs/frame_t120.jpg -y\"})\n"
                "Step 2: analyze_frames({\"image_paths\": [\"/outputs/frame_t120.jpg\"], \"question\": \"...\"})\n"
                "Do NOT pass /videos/*.mp4 to analyze_frames. Call execute_bash NOW to extract frames."
            )})
            state.actual_step -= 1  # Don't count this as a step
            return "continue"

        self._log_step("tool_result", self._build_log_content(text_result, image_parts))
        # Inline truncation when memory is disabled (LLM-in-Sandbox mode).
        # Strip any embedded base64 data URLs first — some tools embed them in the
        # text blob (analyze_clip wrapping clip frames, etc.) and they balloon the
        # context by millions of tokens, triggering a premature _force_answer.
        # Gated off when raw_context ablation is on so only Layer 3 (max_ctx_tokens) applies.
        if not memory and not self._raw_no_trunc:
            text_result = re.sub(
                r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                '[image-stripped-in-no-memory-mode]',
                text_result,
            )
            text_result = truncate_tool_output(text_result)
            # Hard cap: even after truncate, cap each tool result to 12K chars.
            if len(text_result) > 12000:
                text_result = text_result[:6000] + "\n...[tool output truncated]...\n" + text_result[-6000:]
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": text_result})

        # Native multimodal: second LLM call to analyze images in same iteration
        text_result_for_memory = text_result
        if image_parts and self._multimodal_agent and tool_name in ("analyze_frames", "analyze_clip", "describe_frames"):
            # Inject images as user message
            img_msg_content = [{"type": "text", "text": "[Frames from your analysis request. Describe what you observe in detail.]"}]
            img_msg_content.extend(image_parts)
            messages.append({"role": "user", "content": img_msg_content})

            # Rebuild context with memory + images, strip ALL old images
            # (keep only the current frame set to bound context growth)
            if memory:
                llm_messages = self._build_memory_messages(
                    system_prompt=messages[0]["content"],
                    user_task=original_user_task,
                    full_messages=messages,
                    memory=memory,
                    iteration=iteration,
                )
                llm_messages = self._strip_old_images(llm_messages, keep_recent=1)
            else:
                llm_messages = messages

            logger.info(f"[Multimodal] Second LLM call for image analysis ({len(image_parts)} images)")
            # No tools — force the agent to describe what it sees rather than act
            resp2 = self._dispatch_api_call(llm_messages, None)

            # Extract the agent's analysis (no tools = text-only response)
            analysis_text = ""
            thinking_text = ""
            if resp2:
                if hasattr(resp2, 'choices') and resp2.choices:
                    analysis_text = getattr(resp2.choices[0].message, 'content', '') or ''
                elif isinstance(resp2, dict):
                    analysis_text = resp2.get('text_content', '') or resp2.get('text', '') or ''
                    thinking_text = resp2.get('thinking_content', '') or ''

            # Fallback: if content is empty but thinking has analysis, use thinking
            if not analysis_text and thinking_text:
                logger.warning(f"[Multimodal] Empty content but thinking has {len(thinking_text)} chars — using thinking as analysis")
                analysis_text = thinking_text

            if analysis_text:
                messages.append({"role": "assistant", "content": analysis_text})
                text_result_for_memory = analysis_text
                logger.info(f"[Multimodal] Agent analysis: {analysis_text[:200]}")
            else:
                logger.warning("[Multimodal] Agent returned empty analysis for images")
        elif image_parts:
            # Legacy/standard: inject keyframes as user message for next iteration
            img_msg_content = [{"type": "text", "text": "[Visual context: keyframes from the analyzed content]"}]
            img_msg_content.extend(image_parts)
            messages.append({"role": "user", "content": img_msg_content})

        # Update hierarchical memory
        if memory and tool_name != "submit_answer":
            thought = text_content or thinking_content or ""
            memory.update_from_tool(
                tool_name=tool_name, tool_input=tool_input,
                tool_output=text_result_for_memory, iteration=iteration, thought=thought,
            )
            # Sync frame cache from tool handler to orchestrator for render_context
            if hasattr(self, '_frame_b64_cache') and self._frame_b64_cache:
                if hasattr(memory, 'orchestrator') and hasattr(memory.orchestrator, '_frame_cache'):
                    memory.orchestrator._frame_cache.update(self._frame_b64_cache)

        # Process pending frame batches from split analyze_frames calls
        if hasattr(self, '_pending_frame_batches') and self._pending_frame_batches:
            batch_num = 2
            total_batches = len(self._pending_frame_batches) + 1  # +1 for already-processed first batch
            while self._pending_frame_batches:
                batch = self._pending_frame_batches.pop(0)
                batch_urls = batch['image_urls']
                batch_paths = batch['frame_paths']
                batch_question = batch['question']

                logger.info(f"[FrameBatch] Processing batch {batch_num}/{total_batches}: {len(batch_urls)} frames")

                # Build image message for this batch
                img_msg_content = [{"type": "text", "text": f"[Batch {batch_num}/{total_batches}: {len(batch_urls)} more frames. {batch_question}]"}]
                for url in batch_urls:
                    img_msg_content.append({"type": "image_url", "image_url": {"url": url}})
                messages.append({"role": "user", "content": img_msg_content})

                # Second LLM call for this batch
                if memory:
                    llm_messages = self._build_memory_messages(
                        system_prompt=messages[0]["content"],
                        user_task=original_user_task,
                        full_messages=messages,
                        memory=memory,
                        iteration=iteration,
                    )
                    llm_messages = self._strip_old_images(llm_messages, keep_recent=1)
                else:
                    llm_messages = messages

                logger.info(f"[Multimodal] Second LLM call for batch {batch_num} ({len(batch_urls)} images)")
                resp_batch = self._dispatch_api_call(llm_messages, None)

                batch_analysis = ""
                if resp_batch and isinstance(resp_batch, dict):
                    batch_analysis = resp_batch.get('text_content', '') or ''

                if batch_analysis:
                    messages.append({"role": "assistant", "content": batch_analysis})
                    logger.info(f"[Multimodal] Batch {batch_num} analysis: {batch_analysis[:200]}")

                    # Update memory with this batch's analysis
                    if memory and tool_name != "submit_answer":
                        memory.update_from_tool(
                            tool_name="analyze_frames",
                            tool_input={"image_paths": batch_paths, "question": batch_question},
                            tool_output=batch_analysis,
                            iteration=iteration,
                            thought=f"Batch {batch_num}/{total_batches} frame analysis",
                        )

                batch_num += 1

        # Log memory state to trajectory for dashboard
        if memory and tool_name != "submit_answer":
            memory_ctx = memory.render_context()
            # render_context may return list of content blocks (multimodal)
            # Preserve image data URIs for dashboard display
            if isinstance(memory_ctx, list):
                memory_log = ""
                for block in memory_ctx:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        memory_log += block.get("text", "")
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url:
                            memory_log += f'\n{{{{IMAGE:{url}}}}}\n'
                        else:
                            memory_log += "[image]"
            else:
                memory_log = memory_ctx
            self._log_step("memory", memory_log)
            if self.debug:
                self._log_print(f"[dim]{memory.render_stats()}[/dim]")
        elif not memory and tool_name != "submit_answer":
            # No-memory mode: log conversation state as <MEMORY> JSON
            # so the dashboard can render it with the existing renderer.
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            msg_counts: dict = {}
            for m in messages:
                r = m.get("role", "?")
                msg_counts[r] = msg_counts.get(r, 0) + 1
            conv_state = {
                "mode": "no-memory (LLM-in-Sandbox)",
                "conversation": {
                    "total_messages": len(messages),
                    "by_role": msg_counts,
                    "total_chars": total_chars,
                    "est_tokens": total_chars // 4,
                },
                "stats": (
                    f"No-memory mode: {len(messages)} msgs, "
                    f"{total_chars} chars, ~{total_chars // 4} tokens"
                ),
            }
            self._log_step("memory", f"<MEMORY>\n{json.dumps(conv_state, separators=(',', ':'))}\n</MEMORY>")

        if tool_name == "submit_answer":
            state.answer = tool_input.get("answer")
            state.reasoning = tool_input.get("reasoning", "")
            self._log_step("answer", {"answer": state.answer, "reasoning": state.reasoning})
            return "break"

        return "next"

    def _check_early_submit(self, state: _IterationState, iteration: int) -> Optional[str]:
        """Check if the agent should be bounced from an early submit_answer.

        Returns a reason string if the submit should be rejected, or None if it's OK.
        Only enforced when memory config thresholds are set and we're not near the
        iteration budget (last 30% of budget is always allowed to submit).
        """
        # Ablation toggle: when the gate is disabled, every submit_answer goes through.
        if not self._early_submit_gate:
            return None
        cfg = self._memory_config
        iterations_remaining = self.max_iterations - iteration

        # Check for uncompleted visual planner todos — enforced until 5 iterations remain
        # (keeps its own gate so it fires even past the 70% early-submit threshold)
        if iterations_remaining > 5:
            memory = getattr(self, '_current_memory', None)
            todo_list = getattr(memory, '_todo_list', None) if memory else None
            if todo_list:
                open_todos = [t for t in todo_list if not t.get('done')]
                if open_todos:
                    pending_ids = ', '.join(f"#{t['id']}" for t in open_todos)
                    return (
                        f"You have {len(open_todos)} uncompleted visual investigation todo(s): {pending_ids}. "
                        f"Investigate each timestamp_hint using analyze_frames or analyze_clip, "
                        f"then call complete_todo(id=N, finding='what you saw') for each one."
                    )

        # Check if transcript has been called
        has_transcript = "transcribe_audio" in state.recent_tools

        # Count visual-inspection calls (any tool that actually looks at frames/clips).
        _VISUAL_TOOLS = {"analyze_clip", "analyze_frames"}
        vlm_calls = sum(1 for t in state.recent_tools if t in _VISUAL_TOOLS)

        # Build list of unmet requirements
        reasons = []

        if cfg.require_transcript_before_submit and not has_transcript:
            reasons.append(
                "No transcript obtained yet. Call transcribe_audio to get the video transcript — "
                "it provides crucial context (names, timestamps, descriptions) that visual analysis alone cannot."
            )

        if getattr(cfg, "min_iterations_before_submit", 0) and iteration < cfg.min_iterations_before_submit:
            reasons.append(
                f"You are at iteration {iteration} but the pipeline requires at least "
                f"{cfg.min_iterations_before_submit} iterations of investigation before submitting. "
                f"Use analyze_clip / analyze_frames / execute_bash to gather more evidence across the video."
            )

        if getattr(cfg, "min_vlm_calls_before_submit", 0) and vlm_calls < cfg.min_vlm_calls_before_submit:
            reasons.append(
                f"You have made {vlm_calls} visual-analysis call(s) so far "
                f"(analyze_clip / analyze_frames / describe_frames), but at least "
                f"{cfg.min_vlm_calls_before_submit} are required before submitting. "
                f"Inspect more segments of the video first."
            )

        if reasons:
            return " | ".join(reasons)

        return None

    def _force_answer(self, messages, tools, state):
        """Force a final answer when max iterations reached. Works with any provider."""
        logger.warning("Forcing answer — max iterations reached")
        is_open_ended = not getattr(self, '_current_options', None)
        if is_open_ended:
            force_prompts = [
                "CRITICAL: You MUST call submit_answer NOW with your best computed answer. Do not explain — just call submit_answer with the value.",
                "You did NOT call submit_answer. Call it RIGHT NOW. Submit your computed answer (number, value, or short text). Call submit_answer(answer=<value>, reasoning='forced').",
                "FINAL WARNING: If you do not call submit_answer this turn, your answer will be recorded as wrong. Call submit_answer immediately.",
            ]
        else:
            force_prompts = [
                "CRITICAL: You MUST call submit_answer NOW with your best guess. Choose A, B, C, or D. Do not explain — just call submit_answer.",
                "You did NOT call submit_answer. Call it RIGHT NOW. Pick one option letter (one of A, B, C, D, or E if present). Call submit_answer(answer=<letter>, reasoning='forced').",
                "FINAL WARNING: If you do not call submit_answer this turn, your answer will be recorded as wrong. Call submit_answer immediately.",
            ]
        last_text = ""
        for attempt, fprompt in enumerate(force_prompts):
            messages.append({"role": "user", "content": fprompt})
            try:
                resp = self._dispatch_api_call(messages, tools)
                for tc in resp["tool_calls"]:
                    if tc.name == "submit_answer":
                        state.answer = tc.arguments.get("answer")
                        state.reasoning = tc.arguments.get("reasoning", "")
                        self._log_step("answer", {"answer": state.answer, "reasoning": state.reasoning})
                        return
                last_text = resp["text_content"] or last_text
                messages.append({"role": "assistant", "content": last_text or "I'm not sure."})
            except Exception as e:
                logger.warning(f"Force answer attempt {attempt+1}/{len(force_prompts)} failed: {e}")

        if not state.answer:
            extracted = self._extract_answer_from_text(last_text, open_ended=is_open_ended)
            if extracted:
                state.answer = extracted
                state.reasoning = f"[FORCED] Extracted from model response after {len(force_prompts)} prompts"
                logger.warning(f"Extracted answer '{state.answer}' from model text")
            else:
                state.answer = "0" if is_open_ended else "A"
                state.reasoning = f"[FORCED] Model refused to submit. Defaulting to {'0' if is_open_ended else 'A'}."
                logger.warning(f"Could not extract answer — defaulting to {'0' if is_open_ended else 'A'}")
            self._log_step("answer", {"answer": state.answer, "reasoning": state.reasoning})

    def _verify_todo(self, todo_item: dict, finding: str) -> tuple:
        """Delegate todo verification to MemoryOrchestrator."""
        question = getattr(self, '_current_question', '') or ''
        options = getattr(self, '_current_options', []) or []
        cfg = self._visual_planner_config
        _verifier = MemoryOrchestrator(
            question=question,
            options="\n".join(options) if options else "",
            model=cfg.model or self.model,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        _verifier.token_tracker = self.token_tracker
        return _verifier.verify_todo(
            todo_item, finding,
            question=question,
            options=options,
            planner_model=cfg.model or self.model,
        )

    def _run_agent_loop(
        self,
        messages: List[Dict],
        tools: list,
        memory,
        free_tools: set,
        original_user_task: str,
    ) -> _IterationState:
        """Execute the main agent reasoning loop.

        Reusable by subclasses that build their own messages/tools but share
        the same LLM-calling and response-processing infrastructure.

        Returns the final _IterationState (with answer/reasoning populated).
        """
        state = _IterationState()

        # Pre-populate recent_tools with transcript if seeded from cache during init.
        # This prevents the submit guard from bouncing answers for "no transcript".
        if getattr(self, '_transcript_seeded', False):
            state.recent_tools.append("transcribe_audio")

        for iteration in range(self.max_iterations):
            self.token_tracker.set_iteration(iteration)
            steps_remaining = self.max_iterations - iteration
            step_msg = get_iteration_warning(steps_remaining, self.max_iterations)

            # Append step counter to the last user/tool message
            if messages and messages[-1]["role"] in ("user", "tool"):
                last_content = messages[-1].get("content", "")
                if isinstance(last_content, list):
                    for part in last_content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = part["text"] + f"\n{step_msg}"
                            break
                else:
                    messages[-1]["content"] = str(last_content) + f"\n{step_msg}"

            # Build compact messages when memory is active
            if memory and iteration > 0:
                llm_messages = self._build_memory_messages(
                    system_prompt=messages[0]["content"],
                    user_task=original_user_task,
                    full_messages=messages,
                    memory=memory,
                    iteration=iteration,
                )
                if self.debug:
                    self._log_print(
                        f"[dim]Memory: {len(messages)} full msgs -> "
                        f"{len(llm_messages)} compact msgs[/dim]"
                    )
            else:
                llm_messages = messages

            # Strip images from older multimodal messages to control token usage.
            # Gated off when raw_context ablation is on so the only ceiling is max_ctx_tokens.
            if self._multimodal_agent and not self._raw_no_trunc:
                llm_messages = self._strip_old_images(llm_messages, keep_recent=0)

            # Token limit check (LLM-in-Sandbox style): when memory is disabled,
            # the raw conversation history grows unbounded.  If it exceeds the
            # context budget, force the agent to submit its best answer now.
            if not memory:
                # Tokenizer-accurate count: tiktoken cl100k_base for text +
                # fixed Gemini per-image cost (replaces the previous chars/4
                # heuristic which severely undercounted image-laden contexts
                # and let the cap go un-triggered even at 240K-token prompts).
                try:
                    _ctx_tokens = _count_messages_tokens(llm_messages)
                except Exception:
                    _ctx_tokens = sum(
                        len(str(m.get("content", ""))) for m in llm_messages
                    ) // 4
                # Layer 3 context budget. Default 500K tokens to match Gemini's 1M
                # context. Configurable via max_ctx_tokens on the constructor — ablations
                # may override this to study how tighter budgets affect thrashing.
                _max_ctx = self._max_ctx_tokens
                if _ctx_tokens > _max_ctx:
                    # No-memory mode budget guard. Always aggressively trim and
                    # continue iterating — never force-answer on a budget hit.
                    # Force-answer was removed because mixing tight budgets with
                    # an early-commit policy confounds the truncation ablation:
                    # the agent's submission-time accuracy reflects both the
                    # bounded-memory effect AND the early-commit pressure.
                    trimmed_tool = 0
                    stripped_imgs = 0
                    stripped_b64 = 0
                    for m in llm_messages:
                        content = m.get("content")
                        # Tool or assistant string content: hard-cap and strip base64.
                        if isinstance(content, str) and len(content) > 3000:
                            new = re.sub(
                                r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                                '[img-stripped]', content,
                            )
                            if new != content:
                                stripped_b64 += 1
                            if len(new) > 3000:
                                new = new[:2000] + "\n...[aggressively trimmed]...\n"
                                trimmed_tool += 1
                            m["content"] = new
                        # Multimodal user/assistant content: drop image_url blocks entirely.
                        elif isinstance(content, list):
                            new_parts = [
                                p for p in content
                                if not (isinstance(p, dict) and p.get("type") == "image_url")
                            ]
                            if len(new_parts) != len(content):
                                stripped_imgs += 1
                            m["content"] = new_parts if new_parts else ""
                    logger.warning(
                        f"Context tokens ({_ctx_tokens}) exceed limit ({_max_ctx}) at iter {iteration}; "
                        f"trimmed {trimmed_tool} tool_results, stripped_images={stripped_imgs}, "
                        f"stripped_base64={stripped_b64} (truncation-only path; force-answer disabled)"
                    )

            if self.debug:
                _api_format = "gemini-native" if self._use_gemini_native else "openai-compat"
                self._log_print(f"[dim]Calling LLM API ({_api_format}) with {len(llm_messages)} messages, {len(tools)} tools...[/dim]")

            # Unified API call + response processing
            resp = self._dispatch_api_call(llm_messages, tools)
            action = self._process_response(resp, messages, memory, free_tools, state, iteration,
                                               original_user_task=original_user_task, tools=tools)

            if action == "break":
                break
            if action == "continue":
                continue

            # Force answer on last iteration if no answer yet
            if iteration == self.max_iterations - 1 and state.answer is None:
                self._force_answer(messages, tools, state)
                break

        # Post-loop fallback: if answer is still None, scan conversation history
        if state.answer is None:
            is_open_ended = not getattr(self, '_current_options', None)
            text_parts = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    if isinstance(msg.get("content"), str):
                        text_parts.append(msg["content"])
                    if isinstance(msg.get("reasoning_content"), str):
                        text_parts.append(msg["reasoning_content"])
            all_text = " ".join(text_parts)
            extracted = self._extract_answer_from_text(all_text, open_ended=is_open_ended)
            if extracted:
                state.answer = extracted
                state.reasoning = "[FORCED] Extracted from conversation history"
                logger.warning(f"Post-loop extraction: '{state.answer}'")
            else:
                state.answer = "0" if is_open_ended else "A"
                state.reasoning = f"[FORCED] No answer found anywhere. Defaulting to {'0' if is_open_ended else 'A'}."
                logger.warning(f"Post-loop: defaulting to {'0' if is_open_ended else 'A'}")
            self._log_step("answer", {"answer": state.answer, "reasoning": state.reasoning})

        # Completion banner
        if self.verbose:
            step_count = len([s for s in self.trajectory if s.step_type == "iteration"])
            if state.answer:
                self._console.print()
                self._console.print(Panel.fit(
                    f"[bold green]✅ Agent completed in {step_count} steps[/bold green]",
                    border_style="green",
                ))
            else:
                self._console.print()
                self._console.print(Panel.fit(
                    f"[bold red]❌ Agent failed after {step_count} steps[/bold red]",
                    border_style="red",
                ))

        return state

    def run(
        self,
        video_path: str,
        question: str,
        options: Optional[List[str]] = None,
        question_images: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        sub_category: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the agent on a video understanding task.
        
        Args:
            video_path: Path to the video file
            question: Question to answer
            options: Optional list of answer options (for multiple choice)
            
        Returns:
            AgentResult with answer and trajectory
        """
        self.trajectory = []
        self._tool_call_history = {}  # Reset dedup tracking for new run
        self.token_tracker.reset()
        self._current_question = question
        self._current_options = options
        self._question_images = question_images or []

        # Build genre context for orchestrator and prompts
        self._genre_context = ""
        if domain == "Film & Television" and sub_category in ("Movie & TV Show", "Animation"):
            self._genre_context = (
                "SCRIPTED FICTION — characters may state false reasons or deceive each other. "
                "Distinguish stated motivations from actual motivations."
            )

        # Format the task from prompts.yaml (thinking models get step-by-step prompts)
        use_thinking = getattr(self, '_use_thinking', False)
        if options:
            # Check if options already have letter prefixes (e.g., "A. ...")
            if options and options[0].strip()[:2] in ["A.", "A "]:
                options_text = "\n".join(options)
            else:
                options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            task = get_task_prompt(video_path=video_path, question=question, options_text=options_text, thinking=use_thinking)
        else:
            task = get_task_prompt(video_path=video_path, question=question, thinking=use_thinking, open_ended=True)

        # Inject adaptation guidance when the question tests knowledge transfer
        if task_type == "Adaptation":
            adaptation_hint = (
                "\n\n<KNOWLEDGE_TRANSFER>\n"
                "This is a KNOWLEDGE TRANSFER question. The video teaches a method, concept, or technique. "
                "The question asks you to APPLY that knowledge to a NEW scenario with different parameters. "
                "The video and question may involve different numbers, setups, or contexts — this is intentional. "
                "Your job: (1) Learn the METHOD from the video, (2) Apply it to the problem in the question. "
                "Do NOT dismiss the question because it doesn't match the video exactly.\n"
                "</KNOWLEDGE_TRANSFER>"
            )
            task += adaptation_hint

        # Inject narrative reasoning hint for scripted fiction content
        if domain == "Film & Television" and sub_category in ("Movie & TV Show", "Animation"):
            narrative_hint = (
                "\n\n<NARRATIVE_REASONING>\n"
                "This video is SCRIPTED FICTION (movie/TV show/animation). Characters may lie, "
                "deceive, or state false reasons to other characters. When the question asks WHY "
                "a character did something, consider whether their STATED reason is their ACTUAL "
                "motivation. Look for evidence of hidden agendas, cover stories, or dramatic irony "
                "where the audience knows more than the characters.\n"
                "</NARRATIVE_REASONING>"
            )
            task += narrative_hint

        # Initialize messages with system prompt (using dynamic max_images)
        # If question has images, build multimodal user content
        if self._question_images:
            question_image_hint = (
                "\n\n<QUESTION_IMAGE>\n"
                "The image(s) below are part of the QUESTION, not from the video. "
                "They may contain diagrams, tables, graphs, music notation, formulas, or other visual data "
                "that you need to interpret directly. Do NOT look for these images in the video — "
                "analyze them here as provided.\n"
                "</QUESTION_IMAGE>"
            )
            user_content = [{"type": "text", "text": task + question_image_hint}]
            for img_b64 in self._question_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
        else:
            user_content = task
        messages = [
            {"role": "system", "content": self.get_system_prompt(max_images=self.max_images)},
            {"role": "user", "content": user_content}
        ]

        try:
            with DockerRuntime(self.docker_image) as docker:
                self._docker = docker

                # Copy video to container
                video_path = Path(video_path)
                if video_path.exists():
                    container_video_path = f"/videos/{video_path.name}"
                    docker.copy_to_container(str(video_path), container_video_path)
                else:
                    container_video_path = str(video_path)
                    logger.warning(f"Video file not found locally: {video_path}")
                self._container_video_path = container_video_path

                # Update task with container path
                messages[1]["content"] = self._replace_in_content(
                    messages[1]["content"], str(video_path), container_video_path
                )

                # Auto-fetch video info (saves an iteration — model sees it from turn 1)
                _video_info_summary = {}
                try:
                    raw_info = docker.get_video_info(container_video_path)
                    if "format" in raw_info:
                        fmt = raw_info["format"]
                        _video_info_summary["duration_seconds"] = float(fmt.get("duration", 0))
                        _video_info_summary["size_mb"] = round(int(fmt.get("size", 0)) / 1048576, 1)
                        _video_info_summary["format"] = fmt.get("format_long_name", "")
                    for s in raw_info.get("streams", []):
                        if s.get("codec_type") == "video":
                            _video_info_summary["video"] = {
                                "codec": s.get("codec_name"),
                                "width": s.get("width"),
                                "height": s.get("height"),
                                "fps": s.get("r_frame_rate"),
                            }
                        elif s.get("codec_type") == "audio":
                            _video_info_summary["audio"] = {
                                "codec": s.get("codec_name"),
                                "sample_rate": s.get("sample_rate"),
                                "channels": s.get("channels"),
                            }
                    # Inject metadata into the task prompt
                    dur = _video_info_summary.get("duration_seconds", "?")
                    vid = _video_info_summary.get("video", {})
                    res = f"{vid.get('width', '?')}x{vid.get('height', '?')}" if vid else "?"
                    fps = vid.get("fps", "?") if vid else "?"
                    codec = vid.get("codec", "?") if vid else "?"
                    meta_line = f"\nVideo metadata: {dur}s duration, {res}, {codec}, {fps}fps"
                    messages[1]["content"] = self._append_to_content(messages[1]["content"], meta_line)
                    logger.info(f"Auto-fetched video info:{meta_line.strip()}")
                except Exception as e:
                    logger.warning(f"Auto-fetch video info failed: {e}")

                # Hierarchical memory setup
                memory = None
                if self._memory_enabled:
                    if self._memory_config.orchestrated and self._memory_config.summarizer_model:
                        # Orchestrated memory: LLM-managed working memory (Docker-native)
                        orchestrator_prompts = get_orchestrator_prompts()
                        options_text = "\n".join(
                            f"  ({chr(65 + i)}) {opt}" for i, opt in enumerate(options)
                        ) if options else ""
                        orchestrator = MemoryOrchestrator(
                            question=question,
                            options=options_text,
                            model=self._memory_config.summarizer_model,
                            max_tokens=self._memory_config.summarizer_max_tokens,
                            api_base=self._subagent_api_base,
                            api_key=self._subagent_api_key,
                            timeout=self._memory_config.summarizer_timeout,
                            max_working_memory_chars=self._memory_config.max_context_chars,
                            prompts=orchestrator_prompts,
                            manifest_max_detailed=self._memory_config.manifest_max_detailed,
                            manifest_compression_batch=self._memory_config.manifest_compression_batch,
                            max_loop_iterations=self._memory_config.orchestrator_max_iterations,
                            docker_runtime=docker,
                            genre_context=self._genre_context,
                            reasoning_effort=getattr(self._memory_config, 'summarizer_reasoning_effort', None),
                        )
                        orchestrator.token_tracker = self.token_tracker
                        memory = OrchestratedMemory(
                            docker_runtime=docker,
                            orchestrator=orchestrator,
                            max_context_chars=self._memory_config.max_context_chars,
                            question=question,
                            max_key_frames=self._memory_config.max_key_frames,
                        )
                        # Pass question images to memory and orchestrator
                        memory._question_images = self._question_images or []
                        orchestrator._question_images = self._question_images or []
                        if memory._question_images:
                            logger.info(f"[Multimodal] Injected {len(memory._question_images)} question image(s) into memory + orchestrator")
                        # Wire filesystem context assembler
                        fs_assembler = FilesystemContextAssembler(
                            activity_log=memory.activity_log,
                            docker_runtime=docker,
                            max_chars=self._memory_config.fs_context_budget,
                        )
                        orchestrator._fs_assembler = fs_assembler
                        logger.info("Orchestrated memory (Docker-native) initialized")
                    self._current_memory = memory  # For pin_memory tool access
                    # Seed memory with auto-fetched video info
                    if _video_info_summary:
                        memory._absorb_video_info(json.dumps(_video_info_summary))
                    # Always seed transcript into memory if available in cache.
                    # This creates a pinned entry so the agent knows the transcript
                    # exists without needing to call transcribe_audio explicitly.
                    _cached_transcript = ""
                    try:
                        cache = TranscriptCache()
                        t_result = cache.get(str(video_path), whisper_fallback=None)
                        _cached_transcript = t_result.get("transcript", "")
                        if _cached_transcript and _cached_transcript not in ("(no transcript available)", "(no speech detected)"):
                            memory._absorb_transcript(json.dumps(t_result), iteration=0)
                            self._transcript_seeded = True
                            logger.info("Seeded transcript into memory from cache")
                        else:
                            _cached_transcript = ""
                    except Exception as e:
                        logger.debug(f"No cached transcript to seed: {e}")

                    # Narrative timeline: seed neutral per-video descriptions as full story arc
                    try:
                        nt_dir = getattr(self._visual_planner_config, 'narrative_dir', '')
                        if nt_dir and hasattr(memory, '_absorb_narrative_timeline'):
                            nt_dir = Path(nt_dir)
                            if not nt_dir.is_absolute():
                                nt_dir = self._project_root / nt_dir
                            vid_stem = Path(str(video_path)).stem
                            nt_path = nt_dir / f"{vid_stem}.json"
                            if nt_path.exists():
                                memory._absorb_narrative_timeline(nt_path.read_text(encoding="utf-8"))
                                logger.info(f"Seeded narrative timeline from {nt_path.name}")
                    except Exception as e:
                        logger.debug(f"No narrative timeline to seed: {e}")

                    logger.info(
                        f"Orchestrated memory enabled "
                        f"(model={self._memory_config.summarizer_model})"
                    )

                # Visual planner: seed visual descriptions into memory
                if self._visual_planner_config.enabled and memory:
                    qid = getattr(self, '_current_question_id', None)
                    if qid:
                        if hasattr(memory, '_absorb_visual_plan'):
                            # Orchestrated memory: LLM creates warm-start working memory
                            # from visual descriptions + transcript
                            # Try question-specific first, then video-level (neutral)
                            cfg = self._visual_planner_config
                            desc_dir = Path(cfg.descriptions_dir)
                            if not desc_dir.is_absolute():
                                desc_dir = self._project_root / desc_dir
                            vid_id = getattr(self, '_current_video_id', None) or qid.rsplit('-', 1)[0]
                            desc_path = None
                            for candidate in [desc_dir / f"{qid}.json", desc_dir / f"{vid_id}.json"]:
                                if candidate.exists():
                                    desc_path = candidate
                                    break
                            if desc_path:
                                try:
                                    desc_json = desc_path.read_text(encoding="utf-8")
                                    transcript_summary = _cached_transcript[:3000] if _cached_transcript else ""
                                    if len(_cached_transcript or "") > 3000:
                                        transcript_summary += "\n... [truncated]"
                                    memory._absorb_visual_plan(desc_json, transcript_summary)
                                    logger.info(f"Seeded visual plan from {desc_path.name}")
                                except Exception as e:
                                    logger.warning(f"Failed to seed visual plan: {e}")
                            else:
                                logger.info(f"No visual descriptions for {qid}/{vid_id}, using cold-start")
                        else:
                            # Non-orchestrated memory: generate todo list the old way
                            cfg = self._visual_planner_config
                            _planner = MemoryOrchestrator(
                                question=question,
                                options="\n".join(options) if options else "",
                                model=cfg.model or self.model,
                                api_base=self.api_base,
                                api_key=self.api_key,
                            )
                            _planner.token_tracker = self.token_tracker
                            todos = _planner.call_visual_planner(
                                qid, question, options,
                                transcript=_cached_transcript,
                                descriptions_dir=cfg.descriptions_dir,
                                project_root=self._project_root,
                                planner_model=cfg.model or self.model,
                                max_tokens=cfg.max_tokens,
                            )
                            if todos:
                                memory._todo_list = [dict(t, done=False) for t in todos]
                                self._log_step("visual_planner", {
                                    "question_id": qid,
                                    "todos": len(todos),
                                    "items": todos,
                                })

                # Post-skill memory setup: preamble
                if memory:
                    if isinstance(memory, OrchestratedMemory):
                        preamble = get_orchestrated_preamble()
                    else:
                        preamble = get_memory_preamble()
                    if preamble:
                        messages[0]["content"] += "\n\n" + preamble

                # Effective free tools
                free_tools = FREE_TOOLS

                # Build tool list
                tools = self._get_tools_for_provider()

                # Store original user task for memory-aware message construction
                # Extract text only — question images are re-injected via self._question_images
                original_user_task = self._content_to_text(messages[1]["content"])

                # Print startup banners (llm-in-sandbox style)
                if self.verbose:
                    system_prompt = messages[0]["content"]
                    user_prompt = self._content_to_text(messages[1]["content"])
                    self._console.print(Panel(
                        system_prompt[:1000] + "..." if len(system_prompt) > 1000 else system_prompt,
                        title="[bold cyan]SYSTEM PROMPT[/bold cyan]",
                        border_style="cyan",
                        padding=(0, 1),
                    ))
                    self._console.print(Panel(
                        user_prompt[:1000] + "..." if len(user_prompt) > 1000 else user_prompt,
                        title="[bold cyan]USER PROMPT[/bold cyan]",
                        border_style="cyan",
                        padding=(0, 1),
                    ))

                state = self._run_agent_loop(messages, tools, memory, free_tools, original_user_task)

                return AgentResult(
                    answer=state.answer,
                    trajectory=self.trajectory,
                    success=state.answer is not None,
                    reasoning=state.reasoning,
                    token_usage=self.token_tracker.to_dict(),
                )
                
        except Exception as e:
            import traceback
            logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
            return AgentResult(
                answer=None,
                trajectory=self.trajectory,
                success=False,
                error=str(e),
                token_usage=self.token_tracker.to_dict(),
            )
        finally:
            self._docker = None
            # Cleanup temp directory
            if self._temp_dir and os.path.exists(self._temp_dir):
                import shutil
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir = None
