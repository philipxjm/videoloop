"""
Memory implementations: OrchestratedMemory and ActivityLog.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .memory import MemoryOrchestrator, TemporalInterval
from .message_utils import compress_frame_b64

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActivityLog
# ---------------------------------------------------------------------------

class ActivityLog:
    """Condensed log of agent work. Merges consecutive similar actions.

    Unlike WorkLedger which stores every event as separate dicts in flat lists,
    ActivityLog renders a succinct summary where consecutive similar work
    (e.g., multiple frame analyses on adjacent time ranges) gets merged.
    """

    # Patterns for parsing frame extraction from bash commands
    _SEQ_PATTERN = re.compile(r'seq\s+(\d+)\s+(\d+)\s+(\d+)')
    _FRAME_PATH_PATTERN = re.compile(
        r'(\/\S*frames\S*\/frame_t[^\s"\']*\.(?:jpg|png))'
    )

    def __init__(self, docker_runtime=None):
        self.entries: List[dict] = []
        self._docker = docker_runtime

    def record(self, tool_name: str, tool_input: Dict[str, Any],
               tool_output: str, iteration: int) -> None:
        """Record an action, merging with the previous entry if similar."""
        entry = self._parse_action(tool_name, tool_input, tool_output, iteration)
        if not entry:
            return

        # Try to merge with last entry of same type
        if self.entries and self._can_merge(self.entries[-1], entry):
            self._merge(self.entries[-1], entry)
        else:
            self.entries.append(entry)

    def render(self) -> str:
        """Render as a succinct multi-line string."""
        if not self.entries:
            return "(no activity yet)"
        lines = []
        for e in self.entries:
            lines.append(f"- {e['summary']}")
        return "\n".join(lines)

    def get_available_frames(self) -> List[dict]:
        """Return all frame paths with visual analysis for the orchestrator."""
        frames = []
        for e in self.entries:
            if e["type"] == "visual_analysis" and e.get("frame_paths"):
                for path in e["frame_paths"]:
                    # Extract timestamp from path like frame_t120.jpg
                    t_match = re.search(r'frame_t(\d+(?:\.\d+)?)', path)
                    timestamp = float(t_match.group(1)) if t_match else None
                    frames.append({
                        "path": path,
                        "timestamp": timestamp,
                        "visual_analysis": e.get("visual_analysis", ""),
                    })
        return frames

    def _parse_action(self, tool_name: str, tool_input: Dict[str, Any],
                      tool_output: str, iteration: int) -> Optional[dict]:
        """Parse a tool call into a structured activity entry."""
        if tool_name in ("submit_answer", "pin_memory"):
            return None

        if tool_name in ("analyze_frames", "analyze_clip", "describe_frames"):
            return self._parse_visual_analysis(tool_name, tool_input, tool_output, iteration)
        elif tool_name == "transcribe_audio":
            return self._parse_transcript(tool_output, iteration)
        elif tool_name == "execute_bash":
            return self._parse_bash(tool_input, tool_output, iteration)
        elif tool_name == "create_file":
            return self._parse_file_create(tool_input, iteration)
        else:
            # Generic tool
            return {
                "type": "other",
                "tool": tool_name,
                "iterations": [iteration],
                "summary": f"{tool_name} (iter {iteration})",
            }

    def _parse_visual_analysis(self, tool_name: str, tool_input: Dict[str, Any],
                               tool_output: str, iteration: int) -> dict:
        """Parse a visual analysis tool call (agent's own analysis or VLM output)."""
        frame_paths = (
            tool_input.get("image_paths", [])
            or tool_input.get("frame_paths", [])
        )
        frame_count = len(frame_paths)

        # Determine time interval from frame paths
        timestamps = []
        for p in frame_paths:
            t_match = re.search(r'frame_t(\d+(?:\.\d+)?)', str(p))
            if t_match:
                timestamps.append(float(t_match.group(1)))

        if timestamps:
            interval = [min(timestamps), max(timestamps)]
        else:
            interval = None

        # Fallback for analyze_clip / similar: pull start_time/end_time directly.
        if interval is None:
            ts_start = tool_input.get("start_time")
            ts_end = tool_input.get("end_time")
            if ts_start is not None and ts_end is not None:
                try:
                    interval = [float(ts_start), float(ts_end)]
                except (TypeError, ValueError):
                    interval = None
                # Approximate frame count if not already known.
                if frame_count == 0 and interval is not None:
                    fps = tool_input.get("fps", 1.0)
                    try:
                        frame_count = max(1, int((interval[1] - interval[0]) * float(fps)))
                    except (TypeError, ValueError):
                        pass

        # Store full analysis text — no truncation
        visual_analysis = tool_output.strip() if tool_output else ""

        interval_str = f"{interval[0]:.0f}-{interval[1]:.0f}s" if interval else "unknown range"
        summary = f"Analyzed frames {interval_str}, {frame_count} frames (iter {iteration})"

        # Persist to JSON log in Docker — read, append, write back
        if self._docker:
            try:
                new_entry = {
                    "iteration": iteration,
                    "interval": interval,
                    "frames": list(frame_paths),
                    "analysis": visual_analysis,
                }
                # Use python3 inside Docker to safely append to JSON array
                entry_b64 = __import__('base64').b64encode(json.dumps(new_entry).encode()).decode()
                self._docker.execute(
                    f"python3 -c \"\nimport json, base64\ntry:\n    with open('/outputs/visual_analysis_log.json') as f:\n        log = json.load(f)\nexcept (FileNotFoundError, json.JSONDecodeError):\n    log = []\nentry = json.loads(base64.b64decode('{entry_b64}').decode())\nlog.append(entry)\nwith open('/outputs/visual_analysis_log.json', 'w') as f:\n    json.dump(log, f, indent=2)\n\"",
                    timeout=5,
                )
            except Exception:
                pass  # Non-critical

        return {
            "type": "visual_analysis",
            "tool": tool_name,
            "interval": interval,
            "frame_count": frame_count,
            "frame_paths": list(frame_paths),
            "visual_analysis": visual_analysis,
            "iterations": [iteration],
            "summary": summary,
        }

    def _parse_transcript(self, tool_output: str, iteration: int) -> dict:
        """Parse a transcript retrieval."""
        segments = 0
        duration = 0.0
        try:
            parsed = json.loads(tool_output)
            segments = parsed.get("segments_count", 0)
            duration = parsed.get("duration", 0)
        except (json.JSONDecodeError, TypeError):
            pass

        summary = f"Retrieved transcript ({segments} segments, {duration:.0f}s) (iter {iteration})"
        return {
            "type": "transcript",
            "segments": segments,
            "duration": duration,
            "iterations": [iteration],
            "summary": summary,
        }

    def _parse_bash(self, tool_input: Dict[str, Any],
                    tool_output: str, iteration: int) -> dict:
        """Parse a bash command execution."""
        cmd = tool_input.get("command", "")

        # Check if this is a frame extraction
        seq_match = self._SEQ_PATTERN.search(cmd)
        if seq_match and ("ffmpeg" in cmd or "frame_t" in cmd):
            start, step, end = int(seq_match.group(1)), int(seq_match.group(2)), int(seq_match.group(3))
            count = len(range(start, end + 1, step))
            summary = f"Extracted frames {start}-{end}s at {step}s intervals, {count} frames (iter {iteration})"
            return {
                "type": "frame_extraction",
                "interval": [start, end],
                "step": step,
                "count": count,
                "iterations": [iteration],
                "summary": summary,
            }

        # Generic bash command — truncate to 60 chars
        cmd_short = cmd[:60].strip()
        if len(cmd) > 60:
            cmd_short += "..."
        summary = f"Ran bash: {cmd_short} (iter {iteration})"
        return {
            "type": "bash",
            "command": cmd[:200],
            "iterations": [iteration],
            "summary": summary,
        }

    def _parse_file_create(self, tool_input: Dict[str, Any], iteration: int) -> dict:
        """Parse a file creation."""
        path = tool_input.get("path", "unknown")
        summary = f"Created file: {path} (iter {iteration})"
        return {
            "type": "file_create",
            "path": path,
            "iterations": [iteration],
            "summary": summary,
        }

    def _can_merge(self, existing: dict, new: dict) -> bool:
        """Check if two entries can be merged."""
        if existing["type"] != new["type"]:
            return False

        if existing["type"] == "visual_analysis":
            # Never merge — each analysis has its own frames, question, and description
            return False

        if existing["type"] == "frame_extraction":
            # Merge if same step and overlapping/adjacent intervals
            if existing.get("step") == new.get("step"):
                e_int = existing.get("interval", [])
                n_int = new.get("interval", [])
                if e_int and n_int:
                    # Overlapping or adjacent
                    return n_int[0] <= e_int[1] + existing.get("step", 1)
            return False

        # transcript, bash, file_create, other — never merge
        return False

    def _merge(self, existing: dict, new: dict) -> None:
        """Merge new entry into existing one."""
        existing["iterations"].extend(new["iterations"])
        iter_range = f"{existing['iterations'][0]}-{existing['iterations'][-1]}"

        if existing["type"] == "frame_extraction":
            # Extend interval
            if existing.get("interval") and new.get("interval"):
                existing["interval"] = [
                    min(existing["interval"][0], new["interval"][0]),
                    max(existing["interval"][1], new["interval"][1]),
                ]
            existing["count"] = existing.get("count", 0) + new.get("count", 0)
            interval_str = f"{existing['interval'][0]}-{existing['interval'][1]}s"
            existing["summary"] = (
                f"Extracted frames {interval_str} at {existing['step']}s intervals, "
                f"{existing['count']} frames (iter {iter_range})"
            )


# (FileBackedMemory removed — legacy code path, recoverable from git)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OrchestratedMemory — LLM-managed working memory
# ---------------------------------------------------------------------------

class OrchestratedMemory:
    """Memory system where an LLM orchestrator manages working memory.

    Standalone class with a minimal interface matching what agent.py expects.
    """

    def __init__(
        self,
        docker_runtime=None,
        orchestrator: "MemoryOrchestrator" = None,
        max_context_chars: int = 32000,
        question: str = "",
        max_key_frames: int = 3,  # Deprecated — kept for backward compat
        memory_dir: Optional[Path] = None,  # Legacy fallback
    ):
        self.docker = docker_runtime
        self.orchestrator = orchestrator
        self.max_context_chars = max_context_chars
        self.question = question
        self.max_key_frames = max_key_frames
        self.memory_path = "/testbed/memory"  # Inside container

        # Create memory dirs in Docker container
        if self.docker:
            self.docker.execute(f"mkdir -p {self.memory_path}/reasoning", timeout=5)
        elif memory_dir:
            # Legacy fallback: host filesystem
            self.memory_dir = Path(memory_dir)
            for sub in ("reasoning",):
                (self.memory_dir / sub).mkdir(parents=True, exist_ok=True)

        # Activity log (replaces WorkLedger)
        self.activity_log = ActivityLog(docker_runtime=docker_runtime)

        # Stats
        self.total_absorbed_chars: int = 0
        self.total_rendered_chars: int = 0
        self.video_duration: Optional[float] = None

        # Guardrails tracking (agent.py checks these)
        self.vlm_call_count: int = 0
        self.transcript_retrieved: bool = False

        # Full transcript text (injected alongside memory so agent doesn't need to grep)
        self._transcript_text: Optional[str] = None

        # Key frame cache (path -> base64 data) to avoid re-reading from Docker
        self._frame_cache: Dict[str, str] = {}

        # Question images (base64) — always injected at top of memory block
        self._question_images: List[str] = []

        # Interface stubs for compatibility with _build_memory_messages
        self.focus_intervals: List[TemporalInterval] = []
        self.explored_intervals: List[TemporalInterval] = []

    def update_from_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: str,
        iteration: int,
        thought: str = "",
    ) -> None:
        """Process a tool result through the orchestrator."""
        self.total_absorbed_chars += len(tool_output)

        # Track guardrails
        if tool_name in ("analyze_frames", "analyze_clip", "describe_frames"):
            self.vlm_call_count += 1
        elif tool_name == "transcribe_audio":
            self.transcript_retrieved = True
            # Store full transcript for context injection
            try:
                parsed = json.loads(tool_output)
                self._transcript_text = parsed.get("transcript", "")
                if self._transcript_text:
                    logger.info(
                        f"[OrchestratedMemory] Stored transcript: "
                        f"{len(self._transcript_text)} chars for context injection"
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        # Record in activity log (replaces WorkLedger)
        self.activity_log.record(tool_name, tool_input, tool_output, iteration)

        # Delegate to orchestrator
        self.orchestrator.process(
            tool_name, tool_input, tool_output, iteration, thought
        )

        # Persist to Docker
        self._persist(iteration, tool_name, tool_input, thought)
        logger.info(
            f"[OrchestratedMemory] Persisted step {iteration}: "
            f"wm={len(self.orchestrator.working_memory)} chars, "
            f"manifest={len(self.orchestrator.manifest.entries)} entries"
        )

    def _persist(self, iteration: int, tool_name: str,
                 tool_input: Dict[str, Any], thought: str) -> None:
        """Write working memory, manifest, and per-step reasoning to Docker container."""
        memory_content = self.orchestrator.working_memory
        # Replicate StepManifest.save() serialization as JSON string
        manifest_data = []
        for e in self.orchestrator.manifest.entries:
            manifest_data.append({
                "iteration": e.iteration,
                "tool_name": e.tool_name,
                "tool_args_summary": e.tool_args_summary,
                "result_summary": e.result_summary,
                "importance": e.importance,
                "timestamp_range": list(e.timestamp_range),
                "thought_snippet": e.thought_snippet,
                "is_compressed": e.is_compressed,
            })
        manifest_json = json.dumps(manifest_data, ensure_ascii=False, indent=2)
        step_data = json.dumps({
            "iteration": iteration,
            "tool_name": tool_name,
            "tool_args_summary": self.orchestrator._summarize_args(tool_name, tool_input),
            "thought": (thought[:500] if thought else ""),
            "working_memory_len": len(memory_content),
            "manifest_entries": len(self.orchestrator.manifest.entries),
        }, ensure_ascii=False, indent=2)

        if self.docker:
            # Batched Docker exec: write all 3 files in one call
            # Escape single quotes in content for heredoc safety
            safe_memory = memory_content.replace("'", "'\\''")
            safe_manifest = manifest_json.replace("'", "'\\''")
            safe_step = step_data.replace("'", "'\\''")
            script = (
                f"cat > {self.memory_path}/working_memory.md << 'WM_EOF'\n"
                f"{memory_content}\n"
                f"WM_EOF\n"
                f"cat > {self.memory_path}/manifest.json << 'MF_EOF'\n"
                f"{manifest_json}\n"
                f"MF_EOF\n"
                f"cat > {self.memory_path}/reasoning/step_{iteration:03d}.json << 'ST_EOF'\n"
                f"{step_data}\n"
                f"ST_EOF"
            )
            self.docker.execute(script, timeout=10)
        elif hasattr(self, 'memory_dir'):
            # Legacy fallback: host filesystem
            (self.memory_dir / "working_memory.md").write_text(memory_content, encoding="utf-8")
            self.orchestrator.manifest.save(self.memory_dir / "manifest.json")
            step_path = self.memory_dir / "reasoning" / f"step_{iteration:03d}.json"
            step_path.write_text(step_data, encoding="utf-8")

    def render_context(self, include_working: bool = True):
        """Return working memory with question images as multimodal content blocks.

        Returns:
            str if no question images, or list of content blocks with images.
        """
        wm = self.orchestrator.get_working_memory()
        if not wm:
            wm = (
                f"# Video Analysis Memory\n\n"
                f"## Video Info\n"
                f"- Question: {self.question}\n\n"
                f"## Current Understanding\n"
                f"Analysis has not yet begun.\n"
            )
        # Strip any remaining {{IMAGE:...}} base64 blobs — should never be in working memory
        wm = re.sub(r'\{\{IMAGE:data:image/[^}]+\}\}', '', wm)
        # Strip invalid {{FRAME:...}} tags where the path isn't a Docker path
        wm = re.sub(r'\{\{FRAME:(?!/)[^}]*\}\}', '', wm)

        self.total_rendered_chars = len(wm)

        # Append transcript if available
        transcript_section = ""
        if self._transcript_text:
            transcript_section = f"\n<TRANSCRIPT>\n{self._transcript_text}\n</TRANSCRIPT>"

        # Detect frame references: markdown images ![...](/path) or {{FRAME:/path}}
        has_frame_tags = bool(re.search(r'\{\{FRAME:/', wm))
        has_md_images = bool(re.search(r'!\[.*?\]\(/(?:outputs|tmp|question_image).*?\.(?:jpg|png)\)', wm))
        if not self._question_images and not has_frame_tags and not has_md_images:
            return f"<MEMORY>\n{wm}\n</MEMORY>{transcript_section}"

        # Build multimodal content blocks
        content_blocks = []

        # Question images at the top (as proper image blocks, not text)
        if self._question_images:
            content_blocks.append({
                "type": "text",
                "text": (
                    "[QUESTION IMAGE(S) — These are part of the QUESTION, not from the video. "
                    "Analyze them directly. Do NOT search for them in the video.]"
                ),
            })
            for img_b64 in self._question_images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

        # Memory block (clean text, no base64)
        content_blocks.append({"type": "text", "text": f"<MEMORY>\n{wm}\n</MEMORY>"})

        # Resolve frame references in memory to actual images
        # Supports both {{FRAME:/path}} tags and markdown ![alt](/path) syntax
        memory_text = f"<MEMORY>\n{wm}\n</MEMORY>"

        # Collect all frame references: {{FRAME:/path}}, ![...](/path.jpg), ![...](/question_image.png)
        frame_pattern = re.compile(
            r'\{\{FRAME:(/[^}]+)\}\}'           # {{FRAME:/outputs/frame_t120.jpg}}
            r'|!\[[^\]]*\]\((/(?:outputs|tmp)/[^)]*\.(?:jpg|png))\)'  # ![caption](/outputs/frame_t120.jpg)
            r'|!\[[^\]]*\]\((/question_image\.png)\)'  # ![question image](/question_image.png)
        )
        matches = list(frame_pattern.finditer(memory_text))

        if matches:
            resolved_count = 0
            resolved_qimg = 0  # question images
            resolved_frames = 0  # video frames
            failed_count = 0
            content_blocks.pop()  # Remove the plain text memory block
            frame_cache = getattr(self.orchestrator, '_frame_cache', {})
            last_end = 0

            for match in matches:
                # Text before this match
                if match.start() > last_end:
                    content_blocks.append({"type": "text", "text": memory_text[last_end:match.start()]})

                # Extract the path from whichever group matched
                frame_path = match.group(1) or match.group(2) or match.group(3)
                last_end = match.end()

                # Resolve path to image data
                if frame_path == "/question_image.png" and self._question_images:
                    # Map to the actual question image
                    b64 = self._question_images[0]
                    mime = "image/png"
                else:
                    # Read frame from cache or Docker
                    b64 = frame_cache.get(frame_path)
                    if not b64:
                        # Fuzzy match: try filename only (orchestrator may use different dir)
                        fname = os.path.basename(frame_path)
                        for cached_path, cached_b64 in frame_cache.items():
                            if os.path.basename(cached_path) == fname:
                                b64 = cached_b64
                                break
                    if not b64:
                        # Try reading from Docker (frame may still be there)
                        b64 = self._read_frame_base64(frame_path)
                        if b64:
                            frame_cache[frame_path] = b64
                    mime = "image/jpeg"
                if b64:
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                    resolved_count += 1
                    if frame_path == "/question_image.png":
                        resolved_qimg += 1
                    else:
                        resolved_frames += 1
                else:
                    # Frame not found — silently skip (likely hallucinated path)
                    # Don't insert placeholder text as it pollutes the agent's context
                    failed_count += 1

            if resolved_count or failed_count:
                # Estimate total image payload size
                img_payload_kb = sum(
                    len(b.get("image_url", {}).get("url", "")) * 3 // 4 // 1024
                    for b in content_blocks if isinstance(b, dict) and b.get("type") == "image_url"
                )
                logger.info(
                    f"[FrameEmbed] render_context: {resolved_qimg} question img, "
                    f"{resolved_frames} frame(s), {failed_count} skipped, ~{img_payload_kb}KB payload"
                )

            # Remaining text after last match
            if last_end < len(memory_text):
                content_blocks.append({"type": "text", "text": memory_text[last_end:]})

        # 3. Transcript at the end
        if self._transcript_text:
            content_blocks.append({
                "type": "text",
                "text": f"\n<TRANSCRIPT>\n{self._transcript_text}\n</TRANSCRIPT>",
            })

        return content_blocks

    def _read_frame_base64(self, frame_path: str) -> Optional[str]:
        """Read a frame from Docker container and return base64-encoded data."""
        # Check cache first
        if frame_path in self._frame_cache:
            return self._frame_cache[frame_path]

        if not self.docker:
            return None

        try:
            exit_code, output = self.docker.execute(
                f"base64 -w0 {frame_path} 2>/dev/null", timeout=5
            )
            if exit_code == 0 and output and not output.startswith("[STDERR]"):
                data = compress_frame_b64(output.strip())
                self._frame_cache[frame_path] = data
                return data
        except Exception:
            pass
        return None

    def render_stats(self) -> str:
        """Render memory stats for logging."""
        orch_stats = self.orchestrator.render_stats()
        rendered = self.total_rendered_chars
        ratio = self.total_absorbed_chars / max(rendered, 1)
        return (
            f"{orch_stats} | "
            f"activity: {len(self.activity_log.entries)} entries | "
            f"compression={ratio:.1f}x"
        )

    def pin_entry(
        self,
        content: Optional[str] = None,
        entry_id: Optional[str] = None,
        start_t: Optional[float] = None,
        end_t: Optional[float] = None,
    ) -> str:
        """Deprecated — pin_memory tool removed. Key Evidence section replaces Pinned."""
        return "pin_memory is no longer supported. Use Key Evidence section in working memory."

    def get_unexplored_intervals(self) -> List[TemporalInterval]:
        """No temporal cone — orchestrator handles coverage tracking."""
        return []

    # --- Seeding methods (called during agent initialization) ---

    def _absorb_video_info(self, output: str) -> None:
        """Seed video metadata into orchestrator."""
        self.orchestrator.seed_video_info(output)
        try:
            info = json.loads(output)
            self.video_duration = info.get("duration_seconds")
        except (json.JSONDecodeError, TypeError):
            pass
        self._persist_working_memory()

    def _absorb_transcript(self, output: str, iteration: int = 0) -> None:
        """Seed transcript info into orchestrator."""
        self.transcript_retrieved = True
        self.orchestrator.seed_transcript(output)
        self._persist_working_memory()

    def _absorb_visual_plan(self, descriptions_json: str, transcript_summary: str = "") -> None:
        """Seed visual descriptions into orchestrator for warm-start planning."""
        self.orchestrator.seed_visual_plan(descriptions_json, transcript_summary)
        self._persist_working_memory()

    def _absorb_narrative_timeline(self, neutral_vd_json: str) -> None:
        """Seed a neutral narrative timeline into working memory."""
        self.orchestrator.seed_narrative_timeline(neutral_vd_json)
        self._persist_working_memory()

    def _persist_working_memory(self) -> None:
        """Persist just the working memory file."""
        content = self.orchestrator.working_memory
        if self.docker:
            self.docker.execute(
                f"cat > {self.memory_path}/working_memory.md << 'WM_EOF'\n"
                f"{content}\n"
                f"WM_EOF",
                timeout=5,
            )
        elif hasattr(self, 'memory_dir'):
            (self.memory_dir / "working_memory.md").write_text(content, encoding="utf-8")

    # --- Helpers ---

    def _get_interval_for_tool(
        self, tool_name: str, tool_input: Dict[str, Any],
    ) -> Optional[TemporalInterval]:
        """Extract temporal interval from tool input."""
        if tool_name == "analyze_clip":
            start = float(tool_input.get("start_time", 0))
            end = float(tool_input.get("end_time", 0))
            if end > start:
                return TemporalInterval(start, end)
            return None

        frame_paths = (
            tool_input.get("image_paths", [])
            or tool_input.get("frame_paths", [])
        )
        if frame_paths:
            timestamps = []
            for fp in frame_paths:
                m = re.search(r"_t(\d+\.?\d*)", fp)
                if m:
                    timestamps.append(float(m.group(1)))
            if timestamps:
                return TemporalInterval(min(timestamps), max(timestamps))
        return None
