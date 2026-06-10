"""
Configuration loader for Video Understanding Agent.
Loads prompts, tools, and settings from YAML files in the configs directory.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Default configs directory (relative to this file's parent)
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the configs directory."""
    filepath = CONFIGS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


# Cache loaded configs
_prompts_cache: Optional[Dict[str, Any]] = None
_config_cache: Optional[Dict[str, Any]] = None


def get_prompts() -> Dict[str, Any]:
    """Load and cache the prompts YAML.

    Reads ``configs/prompts.yaml`` by default. Set the ``PROMPTS_FILE``
    environment variable to load an alternate file (e.g.
    ``PROMPTS_FILE=prompts_no_clip.yaml``) for ablations.
    """
    global _prompts_cache
    if _prompts_cache is None:
        filename = os.environ.get("PROMPTS_FILE", "prompts.yaml")
        _prompts_cache = _load_yaml(filename)
    return _prompts_cache


def get_raw_config() -> Dict[str, Any]:
    """Load and cache config.yaml as raw dict (use config.get_config() for typed access)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = _load_yaml("config.yaml")
    return _config_cache


def reload_configs():
    """Force reload all configs (useful for testing)."""
    global _prompts_cache, _config_cache
    _prompts_cache = None
    _config_cache = None


# ==============================================================================
# PROMPT ACCESSORS
# ==============================================================================

def get_system_prompt(max_images: int = 32) -> str:
    """Get the main agent system prompt with max_images substituted."""
    prompts = get_prompts()
    template = prompts.get("main_agent", {}).get("system_prompt", "")
    return template.replace("{max_images}", str(max_images))


def get_task_prompt(video_path: str, question: str, options_text: str = None, thinking: bool = False, open_ended: bool = False) -> str:
    """Get the formatted task prompt for the agent.

    Args:
        thinking: If True, use thinking-model prompt that encourages step-by-step reasoning.
        open_ended: If True, use open-ended prompt (no multiple-choice options).
    """
    prompts = get_prompts()
    task_prompts = prompts.get("task_prompt", {})
    suffix = "_thinking" if thinking else ""
    if open_ended:
        template = task_prompts.get(f"open_ended{suffix}", task_prompts.get("open_ended", ""))
        return template.format(video_path=video_path, question=question)
    template = task_prompts.get(f"multiple_choice{suffix}", task_prompts.get("multiple_choice", ""))
    return template.format(video_path=video_path, question=question, options_text=options_text or "")


def get_no_tool_call_message() -> str:
    """Get the reminder message when model forgets to use a tool."""
    prompts = get_prompts()
    return prompts.get("reminders", {}).get("no_tool_call", "")


def get_iteration_warning(steps_remaining: int, max_iterations: int = 50) -> str:
    """Get the appropriate iteration warning based on steps remaining.
    
    R3: More aggressive thresholds — warnings start at 50% budget used,
    not just the last 10 steps. Uses percentage-based logic so it scales
    with any max_iterations value.
    """
    prompts = get_prompts()
    warnings = prompts.get("iteration_warnings", {})
    
    pct_remaining = steps_remaining / max_iterations if max_iterations > 0 else 0
    
    if pct_remaining > 0.50:
        # More than 50% budget left — normal
        template = warnings.get("normal", "Steps Remaining: {steps_remaining}")
    elif pct_remaining > 0.30:
        # 30-50% budget left — start nudging
        template = warnings.get("soon", "Steps Remaining: {steps_remaining}")
    elif pct_remaining > 0.15:
        # 15-30% budget left — warning
        template = warnings.get("warning", "Steps Remaining: {steps_remaining}")
    elif pct_remaining > 0.05:
        # 5-15% budget left — urgent
        template = warnings.get("urgent", "Steps Remaining: {steps_remaining}")
    else:
        # Last 5% — final
        template = warnings.get("final", "FINAL STEP")
    
    return template.format(steps_remaining=steps_remaining)


def get_tool_loop_nudge(loop_tool: str, used_tools: List[str], recent_tools: List[str]) -> str:
    """Get the nudge message when model is stuck in a tool loop."""
    prompts = get_prompts()

    nudge_template = prompts.get("tool_loop_nudge", "")
    nudge = nudge_template.format(loop_tool=loop_tool, used_tools=list(set(used_tools)))

    suggestions = prompts.get("tool_loop_suggestions", {})

    if "execute_bash" not in " ".join(recent_tools):
        nudge += suggestions.get("bash", "")
    if "analyze_clip" not in " ".join(recent_tools) and "analyze_frames" not in " ".join(recent_tools):
        nudge += suggestions.get("vlm", "")
    if "transcribe" not in " ".join(recent_tools):
        nudge += suggestions.get("transcript", "")
    # Don't suggest submit_answer when it IS the looping tool
    if loop_tool != "submit_answer":
        nudge += suggestions.get("submit", "")

    return nudge


def get_memory_preamble() -> str:
    """Get the memory system preamble for hierarchical memory mode."""
    prompts = get_prompts()
    return prompts.get("memory_agent", {}).get("memory_preamble", "")


def get_focus_nudge_template() -> str:
    """Get the focus nudge template for unexplored intervals."""
    prompts = get_prompts()
    return prompts.get("memory_agent", {}).get("focus_nudge", "")


def get_summarizer_prompts() -> Dict[str, str]:
    """Get the memory summarizer prompt templates."""
    prompts = get_prompts()
    return prompts.get("memory_summarizer", {})


def get_orchestrator_prompts() -> Dict[str, str]:
    """Get the memory orchestrator prompt templates."""
    prompts = get_prompts()
    return prompts.get("memory_orchestrator", {})


def get_orchestrated_preamble() -> str:
    """Get the orchestrated memory preamble (Markdown-based memory)."""
    prompts = get_prompts()
    return prompts.get("memory_agent", {}).get("orchestrated_preamble", "")


def get_early_submit_bounce(reason: str) -> str:
    """Get the early submit bounce message."""
    prompts = get_prompts()
    template = prompts.get("memory_agent", {}).get("early_submit_bounce", "")
    return template.format(reason=reason)


def get_forced_answer_message(answer: Optional[str] = None) -> str:
    """Get the forced answer message for thought loop detection."""
    prompts = get_prompts()
    forced = prompts.get("forced_answer", {})
    
    if answer:
        return forced.get("extracted", "").format(answer=answer)
    else:
        return forced.get("fallback", "")
