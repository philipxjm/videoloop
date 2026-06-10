"""
Tool definitions for the Video Understanding Agent
"""

from typing import Dict, Any, List

# ==============================================================================
# VIDEO ANALYSIS TOOLS
# ==============================================================================

transcribe_audio_tool = {
    "type": "function",
    "function": {
        "name": "transcribe_audio",
        "description": """Transcribe speech from video/audio.

Returns the full transcript as timestamped segments with speaker diarization:
- transcript: Compact text like "[6s-9s] SPEAKER_00: spoken text here"
- segments_count: Number of segments
- duration: Total audio duration in seconds
- Speaker labels (e.g. SPEAKER_00, SPEAKER_01) identify different speakers

Results are cached — calling again with the same video_path returns instantly.""",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Path to the video or audio file",
                },
                "model_size": {
                    "type": "string",
                    "description": "Whisper model size: tiny, base, small, medium, large",
                    "enum": ["tiny", "base", "small", "medium", "large"],
                    "default": "base",
                },
            },
            "required": ["video_path"],
        },
    },
}

# ==============================================================================
# VISUAL ANALYSIS TOOLS
# ==============================================================================

analyze_clip_tool = {
    "type": "function",
    "function": {
        "name": "analyze_clip",
        "description": """Analyze a video clip extracted from the video by asking a question about it.

This extracts an ACTUAL VIDEO CLIP (with motion, audio, temporal dynamics) and sends it to a VLM for analysis.
PREFER THIS over analyze_frames for questions involving:
- Motion, actions, gestures, or continuous events
- Temporal ordering ("what happened first/next/before")
- Counting actions or events over time
- Comparing two time periods
- Understanding continuous processes or transformations

Use short clips (15-30s) for focused analysis, medium clips (30-60s) for actions in context, longer clips (1-3 min) for broader understanding.

Example: analyze_clip(start_time=120.0, end_time=180.0, question="How many shell swaps does the host perform?")""",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "number",
                    "description": "Start timestamp in seconds",
                },
                "end_time": {
                    "type": "number",
                    "description": "End timestamp in seconds",
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask about the video clip",
                },
            },
            "required": ["start_time", "end_time", "question"],
        },
    },
}

def get_analyze_frames_tool(max_images: int = 32) -> dict:
    """Get analyze_frames tool with dynamic max_images in description."""
    return {
        "type": "function",
        "function": {
            "name": "analyze_frames",
            "description": f"""Analyze multiple frames by asking a specific question about them.

Provide multiple frames to analyze together, allowing understanding of
temporal relationships, actions across time, and video context.

Supports up to {max_images} images per call - use this for comprehensive video analysis.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"List of paths to images/frames to analyze together (up to {max_images})",
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the frames",
                    },
                },
                "required": ["image_paths", "question"],
            },
        },
    }

# ==============================================================================
# FILE SYSTEM TOOLS
# ==============================================================================

execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": """Execute a bash command in the sandbox container.

Use this for custom video/image processing, file operations, or running
scripts not covered by the specialized tools.

IMPORTANT for ffmpeg: Always put -ss BEFORE -i for fast seeking:
  FAST: ffmpeg -ss 1200 -i video.mp4 -vframes 1 out.jpg
  SLOW: ffmpeg -i video.mp4 -ss 1200 -vframes 1 out.jpg (decodes from start!)""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
            },
            "required": ["command"],
        },
    },
}

list_files_tool = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to directory to list",
                },
            },
            "required": ["directory"],
        },
    },
}

read_file_tool = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read contents of a text file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to read",
                },
            },
            "required": ["file_path"],
        },
    },
}

# ==============================================================================
# ANSWER SUBMISSION TOOL
# ==============================================================================

submit_answer_tool = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": """Submit the final answer to a question about a video.

Call this when you have analyzed the video and determined the best answer.
For multiple-choice questions, provide the answer letter (A, B, C, D, etc.).
For open-ended questions, provide the actual computed value (number, expression, or short text).""",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer: a letter for multiple-choice, or the computed value for open-ended questions",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this answer is correct",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level from 0 to 1",
                },
            },
            "required": ["answer", "reasoning"],
        },
    },
}

create_file_tool = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": """Create a file in the container. Use for writing Python scripts for visual analysis (colors, motion, etc.).

IMPORTANT:
- Use \\n for newlines in the content
- Keep scripts SHORT (under 15 lines)
- DO NOT use this to save transcripts or large text - analyze them directly
- After creating, run with execute_bash(command="python /tmp/script.py")

Example: create_file(path="/tmp/colors.py", content="import cv2\\nimport numpy as np\\nimg = cv2.imread('/outputs/frames/frame_000.jpg')\\nprint('Mean:', img.mean(axis=(0,1)))")""",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path for the file, e.g. /tmp/analyze.py",
                },
                "content": {
                    "type": "string",
                    "description": "File content. Use \\n for newlines. Keep it SHORT.",
                },
            },
            "required": ["path", "content"],
        },
    },
}

# ==============================================================================
# MEMORY PINNING TOOL
# ==============================================================================

pin_memory_tool = {
    "type": "function",
    "function": {
        "name": "pin_memory",
        "description": """Pin a memory entry so it is ALWAYS loaded in your context (L0 Pinned).

Use this when you discover a critical observation that you'll need throughout your analysis,
regardless of where your temporal focus moves. Pinned entries always appear in "pinned_entries".

You can pin by entry_id (shown as "id" in focal_entries) or by time range.
This tool does NOT consume a step — you can call it alongside other tools.

Examples:
  pin_memory(entry_id="temporal_003")
  pin_memory(start_time=120.0, end_time=180.0)""",
        "parameters": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "ID of the entry to pin (e.g. 'temporal_005', 'frame_003')",
                },
                "start_time": {
                    "type": "number",
                    "description": "Start of time range (seconds) to find entries to pin",
                },
                "end_time": {
                    "type": "number",
                    "description": "End of time range (seconds) to find entries to pin",
                },
            },
        },
    },
}

# ==============================================================================
# COUNTING AGENT TOOLS (complete_todo used by visual planner)
# ==============================================================================

complete_todo_tool = {
    "type": "function",
    "function": {
        "name": "complete_todo",
        "description": """Submit evidence that you've completed a visual investigation todo item.
A planner will verify your justification is sufficient before marking it done.
If rejected, you'll receive feedback on what's still needed and the todo stays open.
The todo list is visible as "todo_list" in your MEMORY block.
This tool does NOT consume a step — call it alongside other tools.""",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "ID of the todo item to complete",
                },
                "finding": {
                    "type": "string",
                    "description": "Specific evidence: what you saw, at what timestamp, and how it relates to the question",
                },
            },
            "required": ["id", "finding"],
        },
    },
}

# ==============================================================================
# TOOL COLLECTIONS
# ==============================================================================

# All available tools
ALL_TOOLS = [
    # Video analysis
    transcribe_audio_tool,
    # Visual analysis
    analyze_clip_tool,
    get_analyze_frames_tool(),
    # File system
    execute_bash_tool,
    list_files_tool,
    read_file_tool,
    create_file_tool,
    # Memory
    pin_memory_tool,
    complete_todo_tool,
    # Answer
    submit_answer_tool,
]

# Alias for backwards compatibility
TOOLS = ALL_TOOLS


def get_all_tools(max_images: int = 32, exclude_tools: list = None, include_tools: list = None) -> list:
    """Get all tools with dynamic configuration applied.

    Args:
        include_tools: If non-empty, only these tool names are included (whitelist).
        exclude_tools: Tool names to exclude (ignored when include_tools is set).
    """
    include = set(include_tools) if include_tools else None
    exclude = set(exclude_tools or [])
    tools = []
    for tool in ALL_TOOLS:
        name = tool.get("function", {}).get("name") or tool.get("name", "")
        if include is not None:
            if name not in include:
                continue
        elif name in exclude:
            continue
        if name == "analyze_frames":
            # Use dynamic version with max_images
            tools.append(get_analyze_frames_tool(max_images))
        else:
            tools.append(tool)
    return tools


def _normalize_tool(tool: dict) -> tuple:
    """Extract (name, description, parameters) from a tool in any format."""
    if tool.get("type") == "function" and "function" in tool:
        func = tool["function"]
        return (func["name"], func["description"],
                func.get("parameters", {"type": "object", "properties": {}}))
    return (tool["name"], tool["description"],
            tool.get("input_schema", {"type": "object", "properties": {}}))


def format_tools_for_openai(max_images: int = 32, exclude_tools: list = None, include_tools: list = None) -> list:
    """Format tools for OpenAI-style APIs (OpenAI, Ollama, DeepSeek, etc.)."""
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": p}}
            for n, d, p in (_normalize_tool(t) for t in get_all_tools(max_images, exclude_tools=exclude_tools, include_tools=include_tools))]


def format_tools_for_claude(max_images: int = 32, exclude_tools: list = None, include_tools: list = None) -> list:
    """Format tools for Claude's native tool calling API."""
    return [{"name": n, "description": d, "input_schema": p}
            for n, d, p in (_normalize_tool(t) for t in get_all_tools(max_images, exclude_tools=exclude_tools, include_tools=include_tools))]


def _gemini_schema(schema: dict) -> dict:
    """Recursively convert JSON Schema types to uppercase for Gemini API.

    Gemini requires type values as uppercase enums: STRING, NUMBER, INTEGER,
    BOOLEAN, ARRAY, OBJECT. Also strips unsupported keys like 'additionalProperties'.
    """
    if not isinstance(schema, dict):
        return schema
    out = {}
    # Keys that Gemini doesn't support in function parameter schemas
    _skip_keys = {"additionalProperties", "$schema", "definitions", "$defs"}
    for k, v in schema.items():
        if k in _skip_keys:
            continue
        if k == "type" and isinstance(v, str):
            out[k] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _gemini_schema(v)
        else:
            out[k] = v
    return out


def format_tools_for_gemini(max_images: int = 32, exclude_tools: list = None, include_tools: list = None) -> list:
    """Format tools for Gemini native API (functionDeclarations)."""
    declarations = [
        {"name": n, "description": d, "parameters": _gemini_schema(p)}
        for n, d, p in (_normalize_tool(t) for t in get_all_tools(max_images, exclude_tools=exclude_tools, include_tools=include_tools))
    ]
    return [{"functionDeclarations": declarations}]
