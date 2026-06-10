"""
Orchestrated Memory: ManifestEntry, StepManifest, FilesystemContextAssembler, MemoryOrchestrator.

Contains:
- ManifestEntry, StepManifest: Structured step log with progressive compression.
- FilesystemContextAssembler: Pre-assembles filesystem context for orchestrator.
- MemoryOrchestrator: LLM-powered memory manager that curates working memory.
"""

import json
import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..message_utils import compress_frame_b64

logger = logging.getLogger(__name__)


# LLM completion: use OpenAI format for reliable output capping
def _get_completion():
    """Lazy import of openai_completion."""
    from ..llm_api import openai_completion
    return openai_completion


@dataclass
class ManifestEntry:
    """One step in the orchestrator's manifest."""
    iteration: Optional[int]
    tool_name: str
    tool_args_summary: str
    result_summary: str
    importance: float
    timestamp_range: Tuple[Optional[float], Optional[float]]
    thought_snippet: str
    is_compressed: bool = False


class StepManifest:
    """Structured log of every agent step with progressive compression."""

    def __init__(self, max_detailed: int = 15, compression_batch: int = 5):
        self.entries: List[ManifestEntry] = []
        self.max_detailed = max_detailed
        self.compression_batch = compression_batch

    def add(self, entry: ManifestEntry) -> None:
        self.entries.append(entry)

    @property
    def detailed_entries(self) -> List[ManifestEntry]:
        return [e for e in self.entries if not e.is_compressed]

    @property
    def compressed_entries(self) -> List[ManifestEntry]:
        return [e for e in self.entries if e.is_compressed]

    def maybe_compress(self, model: str = "", api_base: str = "",
                       api_key: str = "", timeout: int = 15,
                       token_tracker: Any = None) -> None:
        """Compress oldest detailed entries when count exceeds max_detailed."""
        detailed = self.detailed_entries
        if len(detailed) <= self.max_detailed:
            return

        # Take oldest batch
        batch = detailed[:self.compression_batch]
        if len(batch) < 2:
            return

        # Try LLM compression
        summary = self._compress_batch_llm(batch, model, api_base, api_key,
                                           timeout, token_tracker)
        if not summary:
            # Fallback: rule-based compression
            summary = self._compress_batch_rule(batch)

        # Replace batch with single compressed entry
        first_iter = batch[0].iteration
        last_iter = batch[-1].iteration
        ts_starts = [e.timestamp_range[0] for e in batch if e.timestamp_range[0] is not None]
        ts_ends = [e.timestamp_range[1] for e in batch if e.timestamp_range[1] is not None]

        compressed = ManifestEntry(
            iteration=None,
            tool_name="_summary_",
            tool_args_summary=f"steps {first_iter}-{last_iter}",
            result_summary=summary,
            importance=max(e.importance for e in batch),
            timestamp_range=(
                min(ts_starts) if ts_starts else None,
                max(ts_ends) if ts_ends else None,
            ),
            thought_snippet="",
            is_compressed=True,
        )

        # Remove the batch entries, insert compressed entry in their place
        batch_set = set(id(e) for e in batch)
        new_entries = []
        inserted = False
        for e in self.entries:
            if id(e) in batch_set:
                if not inserted:
                    new_entries.append(compressed)
                    inserted = True
            else:
                new_entries.append(e)
        self.entries = new_entries

    def _compress_batch_llm(self, batch: List[ManifestEntry], model: str,
                            api_base: str, api_key: str, timeout: int,
                            token_tracker: Any) -> Optional[str]:
        """Compress a batch of entries via LLM."""
        if not model:
            return None

        _completion = _get_completion()

        lines = []
        for e in batch:
            ts = f"[{e.timestamp_range[0]}-{e.timestamp_range[1]}s]" if e.timestamp_range[0] is not None else ""
            lines.append(f"  Step {e.iteration}: {e.tool_name}({e.tool_args_summary}) {ts} → {e.result_summary}")
        steps_text = "\n".join(lines)

        try:
            resp = _completion(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Compress these agent steps into 2-3 sentences. "
                        "Include timestamps, tool names, and key findings. "
                        "Format: 'Steps N-M: [narrative]'"
                    )},
                    {"role": "user", "content": steps_text},
                ],
                api_base=api_base,
                api_key=api_key,
                max_tokens=200,
                temperature=1.0,
                timeout=timeout if isinstance(timeout, (int, float)) else 30,
            )
            usage = resp.get("usage", {})
            if token_tracker and usage:
                token_tracker.record(
                    "summarizer", model,
                    usage.get("promptTokenCount", 0) or 0,
                    (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0),
                    call_site="manifest.compress",
                )
            result = (resp.get("content", "") or "").strip()
            return result if result else None
        except Exception as e:
            logger.warning(f"Manifest compression LLM call failed: {e}")
            return None

    def _compress_batch_rule(self, batch: List[ManifestEntry]) -> str:
        """Rule-based fallback compression."""
        first_iter = batch[0].iteration
        last_iter = batch[-1].iteration
        tools = sorted(set(e.tool_name for e in batch))
        return f"Steps {first_iter}-{last_iter}: Used {', '.join(tools)}."

    def render_for_orchestrator(self, max_chars: int = 8000) -> str:
        """Render manifest as text for the orchestrator's prompt context."""
        lines = []
        for e in self.entries:
            if e.is_compressed:
                lines.append(f"[compressed] {e.result_summary}")
            else:
                ts = ""
                if e.timestamp_range[0] is not None and e.timestamp_range[1] is not None:
                    ts = f" [{e.timestamp_range[0]:.0f}-{e.timestamp_range[1]:.0f}s]"
                imp = f" imp={e.importance:.1f}" if e.importance != 0.5 else ""
                lines.append(
                    f"Step {e.iteration}: {e.tool_name}({e.tool_args_summary}){ts}{imp}"
                    f"\n  → {e.result_summary}"
                )

        text = "\n".join(lines)
        if len(text) > max_chars:
            # Truncate from the top (oldest entries)
            while len(text) > max_chars and len(lines) > 2:
                lines.pop(0)
                text = "\n".join(lines)
            text = "[earlier steps omitted]\n" + text
        return text

    def save(self, path) -> None:
        """Persist manifest to JSON."""
        import json as _json
        data = []
        for e in self.entries:
            data.append({
                "iteration": e.iteration,
                "tool_name": e.tool_name,
                "tool_args_summary": e.tool_args_summary,
                "result_summary": e.result_summary,
                "importance": e.importance,
                "timestamp_range": list(e.timestamp_range),
                "thought_snippet": e.thought_snippet,
                "is_compressed": e.is_compressed,
            })
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path) -> None:
        """Load manifest from JSON."""
        import json as _json
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            for item in data:
                self.entries.append(ManifestEntry(
                    iteration=item.get("iteration"),
                    tool_name=item["tool_name"],
                    tool_args_summary=item.get("tool_args_summary", ""),
                    result_summary=item.get("result_summary", ""),
                    importance=item.get("importance", 0.5),
                    timestamp_range=tuple(item.get("timestamp_range", [None, None])),
                    thought_snippet=item.get("thought_snippet", ""),
                    is_compressed=item.get("is_compressed", False),
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load manifest: {e}")


class FilesystemContextAssembler:
    """Pre-assembles filesystem context for the orchestrator prompt.

    Reads from in-memory state (activity log, working memory history) and
    optionally from Docker for sandbox file listings. Produces a structured
    context string injected into the orchestrator's LLM call.
    """

    def __init__(self, activity_log=None, docker_runtime=None,
                 max_chars: int = 2000):
        self.activity_log = activity_log  # In-memory ref to ActivityLog
        self.docker = docker_runtime  # For reading sandbox files
        self.max_chars = max_chars
        self._memory_history: List[Tuple[int, str]] = []  # Ring buffer
        self._max_history = 3

    def snapshot_memory(self, iteration: int, memory: str) -> None:
        """Save a working memory snapshot for delta detection."""
        self._memory_history.append((iteration, memory))
        if len(self._memory_history) > self._max_history:
            self._memory_history.pop(0)

    def assemble(self, mode: str, tool_name: str, tool_input: Dict,
                 current_memory: str, iteration: int) -> str:
        """Assemble filesystem context for the orchestrator prompt.

        Args:
            mode: Orchestrator mode (bootstrap, edit)
            tool_name: Name of the tool that just ran
            tool_input: Tool input dict
            current_memory: Current working memory string
            iteration: Current iteration number

        Returns:
            Structured context string (activity log always included; delta/sandbox for edit mode)
        """
        parts = []

        # 1. Activity log (always)
        if self.activity_log:
            rendered = self.activity_log.render()
            if rendered:
                parts.append(f"ACTIVITY LOG:\n{rendered}")

        # 2. Available frames (from activity log)
        if self.activity_log:
            avail = self.activity_log.get_available_frames()
            logger.info(f"[FSContext] activity_log entries={len(self.activity_log.entries)}, available_frames={len(avail)}")
            frames_section = self._render_available_frames()
            if frames_section:
                parts.append(frames_section)

        # 3. Working memory delta + recovery hints (edit only)
        if mode == "edit" and self._memory_history:
            delta = self._render_memory_delta(current_memory)
            if delta:
                parts.append(delta)

        # 4. Sandbox file index (edit only, reads from Docker)
        if mode == "edit" and self.docker:
            sandbox = self._render_sandbox_files()
            if sandbox:
                parts.append(sandbox)

        result = "\n\n".join(parts)
        return result[:self.max_chars] if result else ""

    def _render_available_frames(self) -> str:
        """Render available frames with visual analysis for key frame tagging."""
        if not self.activity_log:
            return ""
        frames = self.activity_log.get_available_frames()
        if not frames:
            return ""

        # Deduplicate by path and limit to most recent 20 frames
        seen = set()
        unique_frames = []
        for f in reversed(frames):
            if f["path"] not in seen:
                seen.add(f["path"])
                unique_frames.append(f)
        unique_frames.reverse()
        unique_frames = unique_frames[-20:]  # Keep last 20

        lines = ["AVAILABLE FRAMES:"]
        for f in unique_frames:
            ts = f"[{f['timestamp']:.0f}s]" if f["timestamp"] is not None else "[?s]"
            analysis = f["visual_analysis"] if f["visual_analysis"] else "no description"
            lines.append(f"- {f['path']} {ts} \"{analysis}\"")
        lines.append(
            "Use inspect_frames(start, end) to visually inspect frames from a specific interval."
        )
        return "\n".join(lines)

    def _render_memory_delta(self, current_memory: str) -> str:
        """Compare current memory with previous snapshot, report changes."""
        if not self._memory_history:
            return ""

        prev_iter, prev_memory = self._memory_history[-1]
        prev_len = len(prev_memory)
        curr_len = len(current_memory)

        if prev_len == 0:
            return ""

        change_pct = ((curr_len - prev_len) / prev_len) * 100
        shrinkage = change_pct < -30

        # Detect lost sections
        prev_sections = set(re.findall(r'^## (.+)$', prev_memory, re.MULTILINE))
        curr_sections = set(re.findall(r'^## (.+)$', current_memory, re.MULTILINE))
        lost_sections = prev_sections - curr_sections

        parts = []
        parts.append(
            f"WORKING MEMORY DELTA (iter {prev_iter}→current): "
            f"{prev_len}→{curr_len} chars ({change_pct:+.0f}%)."
            + (f" Lost: [{', '.join(lost_sections)}]." if lost_sections else "")
        )

        # Recovery hint if significant shrinkage or section loss
        if (shrinkage or len(lost_sections) >= 2) and lost_sections:
            recovery_lines = ["RECOVERY: Previous content of lost sections:"]
            for section_name in lost_sections:
                # Extract the lost section content from previous memory
                pattern = rf'^## {re.escape(section_name)}\n(.*?)(?=\n## |\Z)'
                match = re.search(pattern, prev_memory, re.MULTILINE | re.DOTALL)
                if match:
                    content = match.group(1).strip()[:300]
                    recovery_lines.append(f"## {section_name}\n{content}")
            recovery_lines.append("Restore accidentally dropped sections.")
            parts.append("\n".join(recovery_lines))

        return "\n".join(parts)

    def _render_sandbox_files(self) -> str:
        """List files in /testbed/ and /outputs/ via Docker exec."""
        if not self.docker:
            return ""
        try:
            exit_code, output = self.docker.execute(
                "{ find /testbed -maxdepth 2 \\( -name '*.txt' -o -name '*.json' -o -name '*.py' -o -name '*.csv' -o -name '*.jsonl' \\) "
                "2>/dev/null | grep -v /testbed/memory/; "
                "find /outputs -maxdepth 1 \\( -name '*.jsonl' -o -name '*.json' -o -name '*.txt' \\) 2>/dev/null; } | head -20",
                timeout=5,
            )
            if exit_code == 0 and output.strip():
                files = output.strip().split("\n")
                lines = ["SANDBOX FILES:"]
                for f in files:
                    lines.append(f"- {f.strip()}")
                return "\n".join(lines)
        except Exception:
            pass
        return ""


class MemoryOrchestrator:
    """LLM-powered memory manager that actively curates working memory.

    Replaces the mechanical temporal-cone assembly with an LLM that reads
    the step manifest and composes a human-readable Markdown working memory.
    """

    # Frame extraction patterns (for importance scoring)
    _FFMPEG_RE = re.compile(r"ffmpeg.*-ss|frame_t\d+", re.IGNORECASE)

    def __init__(
        self,
        question: str,
        options: str,
        model: str,
        max_tokens: int = 600,
        api_base: str = "",
        api_key: str = "",
        timeout: int = 15,
        max_working_memory_chars: int = 32000,
        prompts: Optional[Dict[str, str]] = None,
        manifest_max_detailed: int = 15,
        manifest_compression_batch: int = 5,
        fs_assembler: Optional["FilesystemContextAssembler"] = None,
        max_loop_iterations: int = 0,
        docker_runtime: Any = None,
        genre_context: str = "",
        tracer: Any = None,  # MemoryTracer | None; see video_agent/memory/tracer.py
        reasoning_effort: Optional[str] = None,
    ):
        self.question = question
        self.genre_context = genre_context
        self.options = options
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        # Empty api_base = the official Gemini endpoint (resolved in llm_api)
        self.api_base = api_base or os.environ.get("SUMMARIZER_API_BASE", "")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._raw_timeout = timeout
        self.timeout = timeout
        self.max_working_memory_chars = max_working_memory_chars
        self.prompts = prompts or {}

        self.working_memory: str = ""
        self._question_images: List[str] = []  # Base64 question images for multimodal context
        self._frame_cache: Dict[str, str] = {}  # path -> base64 cache for frame resolution
        self._last_viewed_frames: List[str] = []  # paths from most recent inspect_frames
        self._inspected_frames: set = set()  # frames already auto-inspected
        self.manifest = StepManifest(
            max_detailed=manifest_max_detailed,
            compression_batch=manifest_compression_batch,
        )
        self.calls_made: int = 0
        self.calls_failed: int = 0
        self.total_raw_chars: int = 0
        self.cumulative_prompt_tokens: int = 0  # Track total orchestrator input tokens
        self.token_tracker: Any = None
        self._fs_assembler = fs_assembler

        # Agentic loop settings
        self.max_loop_iterations = max_loop_iterations
        self._docker_runtime = docker_runtime
        self.tool_loop_iterations: int = 0
        self.tool_calls_dispatched: int = 0

        # Video info (set once via seed_video_info)
        self._video_info: Optional[Dict[str, Any]] = None

        # Analysis tracer (None unless explicitly passed or env enables one)
        self._tracer = tracer
        # Buffers populated by _call_orchestrator for the tracer to pick up:
        self._last_subloop_calls: List[Dict[str, Any]] = []
        self._last_prompt_tokens: int = 0
        self._last_output_tokens: int = 0

    def seed_video_info(self, video_info_json: str) -> None:
        """Seed video metadata into working memory."""
        try:
            info = json.loads(video_info_json)
            self._video_info = info
            dur = info.get("duration_seconds", "?")
            vid = info.get("video", {})
            res = f"{vid.get('width', '?')}x{vid.get('height', '?')}" if vid else "?"
            logger.info(f"[Orchestrator] Seeded video info: {dur}s, {res}")
            self.working_memory = (
                f"# Video Analysis Memory\n\n"
                f"## Video Info\n"
                f"- Duration: {dur}s, Resolution: {res}\n"
                f"- Question: {self.question}\n"
                f"- Options: {self.options}\n\n"
                f"## Current Understanding\n"
                f"Analysis has not yet begun.\n\n"
                f"## Key Evidence\n"
                f"(none yet)\n\n"
                f"## Temporal Coverage\n"
                f"- No intervals analyzed yet\n\n"
                f"## Activity Log\n"
                f"(no activity yet)\n\n"
                f"## Open Questions\n"
                f"- What is the overall content/structure of this video?\n"
                f"- Where in the timeline does the question-relevant content occur?\n"
            )
        except (json.JSONDecodeError, TypeError):
            pass

    def seed_narrative_timeline(self, neutral_vd_json: str) -> None:
        """Seed a narrative timeline from neutral visual descriptions.

        Parses a video-level clip-by-clip description file and builds a compact
        timeline that's injected as a ## Narrative Timeline section in working
        memory. This gives the agent the full story arc before it starts
        investigating specific moments.
        """
        try:
            data = json.loads(neutral_vd_json)
        except (json.JSONDecodeError, TypeError):
            return

        chunks = data.get("chunks", [])
        if not chunks:
            return

        # Build compact timeline: one line per clip with start-end and summary
        lines = []
        for c in chunks:
            start = int(c.get("start_seconds", 0))
            end = int(c.get("end_seconds", 0))
            summary = c.get("summary", "").strip()
            # Strip JSON/markdown preamble noise from neutral VDs
            if summary.startswith(("Certainly", "Sure", "Here", "Of course", "**")):
                # Find first real sentence after preamble
                for marker in ("\n", ". ", ": "):
                    idx = summary.find(marker, 20)
                    if idx != -1 and idx < 150:
                        summary = summary[idx+1:].strip()
                        break
            # Remove markdown artifacts
            summary = re.sub(r"\*\*|##|###", "", summary)
            summary = re.sub(r"\s+", " ", summary).strip()
            # Truncate to keep timeline compact (target ~150 chars per line)
            if len(summary) > 180:
                summary = summary[:177] + "..."
            if summary:
                lines.append(f"- [{start}s-{end}s] {summary}")

        if not lines:
            return

        timeline_block = "## Narrative Timeline\n" + "\n".join(lines) + "\n"
        logger.info(f"[Orchestrator] Seeded narrative timeline: {len(lines)} clips")

        # Insert BEFORE Current Understanding so the agent sees the full story first
        if "## Current Understanding" in self.working_memory:
            self.working_memory = self.working_memory.replace(
                "## Current Understanding",
                timeline_block + "\n## Current Understanding",
                1,
            )
        else:
            self.working_memory += "\n" + timeline_block

    def seed_transcript(self, transcript_json: str) -> None:
        """Seed transcript availability into working memory Key Evidence section."""
        try:
            data = json.loads(transcript_json)
            segments = data.get("segments_count", 0)
            duration = data.get("duration", 0)
            logger.info(f"[Orchestrator] Seeded transcript: {segments} segments, {duration:.0f}s")
            note = (
                f"- Transcript available ({segments} segments, {duration:.0f}s). "
                f"Call transcribe_audio for retrieval.\n"
            )
            if "## Key Evidence" in self.working_memory:
                self.working_memory = self.working_memory.replace(
                    "## Key Evidence\n", f"## Key Evidence\n{note}", 1
                )
            else:
                self.working_memory += f"\n## Key Evidence\n{note}"
        except (json.JSONDecodeError, TypeError):
            pass

    def seed_visual_plan(self, descriptions_json: str, transcript_summary: str = "") -> None:
        """Use orchestrator LLM to create warm-start working memory from visual descriptions.

        Replaces the cold-start template from seed_video_info() with a rich initial
        working memory that includes investigation targets derived from visual descriptions.
        Falls back to mechanical append if LLM call fails.
        """
        _completion = _get_completion()

        try:
            desc = json.loads(descriptions_json)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[Orchestrator] Failed to parse visual descriptions: {e}")
            return

        # Condense descriptions to relevant_segments + summaries
        condensed = []
        for chunk in desc.get("chunks", []):
            start_s = chunk.get("start_seconds", 0)
            end_s = chunk.get("end_seconds", 0)
            for seg in chunk.get("relevant_segments", []):
                condensed.append({
                    "window": f"{start_s}s-{end_s}s",
                    "time": seg.get("time", ""),
                    "description": seg.get("description", ""),
                })
            if chunk.get("summary"):
                condensed.append({
                    "window": f"{start_s}s-{end_s}s",
                    "time": "chunk_summary",
                    "description": chunk["summary"],
                })

        if not condensed:
            logger.info("[Orchestrator] No visual segments found in descriptions, skipping visual plan")
            return

        # Truncate to keep prompt manageable (~8K chars for visual segments)
        visual_text = json.dumps(condensed, ensure_ascii=False)
        if len(visual_text) > 8000:
            # Keep first and last segments, trim middle
            half = len(condensed) // 2
            truncated = condensed[:half] + [{"window": "...", "time": "...", "description": "[middle segments omitted]"}] + condensed[-half:]
            visual_text = json.dumps(truncated, ensure_ascii=False)

        # Get video metadata
        dur = "?"
        res = "?"
        if self._video_info:
            dur = self._video_info.get("duration_seconds", "?")
            vid = self._video_info.get("video", {})
            res = f"{vid.get('width', '?')}x{vid.get('height', '?')}" if vid else "?"

        template = self.prompts.get("user_template_visual_bootstrap", "")
        if not template:
            logger.warning("[Orchestrator] No user_template_visual_bootstrap prompt, skipping visual plan")
            return

        # Detect counting questions — need dense full-video coverage
        is_counting = any(w in (self.question or "").lower() for w in [
            "how many", "count", "number of", "how often", "how much time",
        ])
        if is_counting:
            counting_instructions = (
                "COUNTING QUESTION DETECTED — This question requires counting events/items across the video.\n"
                "Your Open Questions MUST cover the ENTIRE video systematically:\n"
                f"- Divide the full {dur}s video into intervals of ~30s each\n"
                "- Create one Open Question per interval: '- [Xs-Ys] Count occurrences of [target] in this segment'\n"
                "- The agent must analyze EVERY interval — missing any segment could miss a count\n"
                "- In ## Current Understanding, note: 'This is a counting question. Must systematically scan the entire video.'\n"
                "- Tell the agent to maintain a running tally and use execute_bash(python3) to track counts"
            )
            logger.info(f"[Orchestrator] Counting question detected — injecting dense coverage plan")
        else:
            counting_instructions = ""

        # Use safe substitution to avoid KeyError from literal braces in content
        user_msg = template
        for key, value in {
            "question": self.question,
            "options": self.options,
            "duration": dur,
            "resolution": res,
            "transcript_summary": transcript_summary[:3000] if transcript_summary else "(no transcript available)",
            "visual_segments": visual_text,
            "counting_instructions": counting_instructions,
        }.items():
            user_msg = user_msg.replace(f"{{{key}}}", str(value))

        system_prompt = self._render_system_prompt()

        try:
            self.calls_made += 1
            resp = _completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=1.0,
                timeout=self._raw_timeout,
            )
            usage = resp.get("usage", {})
            if self.token_tracker and usage:
                self.token_tracker.record(
                    "summarizer", self.model,
                    usage.get("promptTokenCount", 0) or 0,
                    (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0),
                    call_site="orchestrator.visual_plan",
                )
            raw = (resp.get("content", "") or "").strip()
            if raw:
                # Strip markdown code fences if present
                raw = re.sub(r"^```(?:markdown)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
                self.working_memory = raw
                logger.info(
                    f"[Orchestrator] Visual plan seeded: {len(self.working_memory)} chars, "
                    f"{len(condensed)} visual segments"
                )
                return
        except Exception as e:
            logger.warning(f"[Orchestrator] Visual plan LLM call failed: {e}")

        # Fallback: mechanically append top visual segments to Open Questions
        logger.info("[Orchestrator] Visual plan fallback: mechanical append to Open Questions")
        top_segments = [s for s in condensed if s.get("time") != "chunk_summary"][:7]
        questions = []
        for seg in top_segments:
            t = seg.get("time", "?")
            desc_text = seg.get("description", "")[:100]
            questions.append(f"- [{t}] Verify: {desc_text}")
        if questions and "## Open Questions" in self.working_memory:
            self.working_memory = self.working_memory.replace(
                "## Open Questions\n- What is the overall content/structure of this video?\n"
                "- Where in the timeline does the question-relevant content occur?\n",
                "## Open Questions\n" + "\n".join(questions) + "\n",
            )

    def process(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: str,
        iteration: int,
        thought: str = "",
    ) -> str:
        """Process a tool result: update manifest, rewrite working memory.

        Returns the updated working memory string.
        """
        if tool_name == "submit_answer":
            return self.working_memory

        self.total_raw_chars += len(tool_output)

        # 1. Create manifest entry
        entry = self._create_manifest_entry(
            tool_name, tool_input, tool_output, iteration, thought
        )
        self.manifest.add(entry)
        logger.info(
            f"[Orchestrator] iter={iteration} tool={tool_name} "
            f"imp={entry.importance:.1f} ts={entry.timestamp_range} "
            f"output_len={len(tool_output)} manifest_size={len(self.manifest.entries)}"
        )

        # 2. Compress old manifest entries if needed
        pre_compress = len(self.manifest.detailed_entries)
        self.manifest.maybe_compress(
            model=self.model, api_base=self.api_base,
            api_key=self.api_key, timeout=self.timeout,
            token_tracker=self.token_tracker,
        )
        post_compress = len(self.manifest.detailed_entries)
        if pre_compress != post_compress:
            logger.info(
                f"[Orchestrator] Manifest compressed: {pre_compress} → {post_compress} detailed entries"
            )

        # 3. Call orchestrator LLM
        wm_before = len(self.working_memory)
        wm_before_text = self.working_memory  # snapshot for tracer
        updated = self._call_orchestrator(tool_name, tool_input, tool_output,
                                          iteration, thought, entry)
        if updated is not None:
            self.working_memory = updated
            logger.info(
                f"[Orchestrator] LLM updated working memory: {wm_before} → {len(self.working_memory)} chars"
            )
        else:
            # Fallback: mechanical append
            self._fallback_append(entry)
            logger.warning(
                f"[Orchestrator] LLM failed, mechanical fallback: {wm_before} → {len(self.working_memory)} chars"
            )

        # 4. Deduplicate sections (keep first occurrence of each ## header)
        self.working_memory = self._deduplicate_sections(self.working_memory)

        # 4b. Convert raw base64 image blobs to markdown references.
        # The orchestrator may copy {{IMAGE:data:...}} from context into edits.
        # These waste tokens. Convert to markdown using last viewed frame path if available.
        _blob_recoveries = []
        def _recover_blob(match):
            if self._last_viewed_frames:
                path = self._last_viewed_frames[0]
                _blob_recoveries.append(path)
                return f'![frame]({path})'
            _blob_recoveries.append(None)
            return ''  # No frame context — just strip
        self.working_memory = re.sub(
            r'\{\{IMAGE:data:image/[^}]+\}\}',
            _recover_blob,
            self.working_memory
        )
        if _blob_recoveries:
            recovered = [p for p in _blob_recoveries if p]
            stripped = len(_blob_recoveries) - len(recovered)
            if recovered:
                logger.info(f"[FrameEmbed] Recovered {len(recovered)} base64 blob(s) → markdown: {recovered}")
            if stripped:
                logger.info(f"[FrameEmbed] Stripped {stripped} base64 blob(s) (no frame context)")

        # 4c. Validate markdown image paths — strip hallucinated ones that aren't in cache
        def _validate_md_image(match):
            path = match.group(1)
            fname = os.path.basename(path)
            # Check exact path and fuzzy (filename only) match in cache
            if path in self._frame_cache:
                return match.group(0)  # Keep it
            for cached_path in self._frame_cache:
                if os.path.basename(cached_path) == fname:
                    return match.group(0)  # Keep it — fuzzy match exists
            # Also keep if it can be read from Docker right now
            if self._docker_runtime:
                ec, _ = self._docker_runtime.execute(f"test -f {path} && echo ok", timeout=3)
                if ec == 0:
                    return match.group(0)  # File exists in Docker
            # Hallucinated path — strip the markdown image but keep any surrounding text
            logger.debug(f"[FrameEmbed] Stripping hallucinated path: {path}")
            return ''
        self.working_memory = re.sub(
            r'!\[[^\]]*\]\((/(?:outputs|tmp)/[^)]*\.(?:jpg|png))\)',
            _validate_md_image,
            self.working_memory
        )

        # 4d. Auto-inspect new frames and embed the most relevant one
        if tool_name == "analyze_frames" and self._docker_runtime:
            frame_paths = tool_input.get("image_paths") or tool_input.get("frame_paths", [])
            if frame_paths:
                # Check which frames haven't been inspected yet
                new_paths = [p for p in frame_paths if p not in self._inspected_frames]
                if new_paths:
                    self._auto_inspect_and_embed(new_paths)

        # 4e. For analyze_clip, sample keyframes from the clip's time range
        # so the orchestrator can embed visual evidence into Key Evidence
        if tool_name == "analyze_clip" and self._docker_runtime:
            st = tool_input.get("start_time")
            et = tool_input.get("end_time")
            if st is not None and et is not None:
                try:
                    keyframe_paths = self._sample_clip_keyframes(float(st), float(et))
                    new_paths = [p for p in keyframe_paths if p not in self._inspected_frames]
                    if new_paths:
                        self._auto_inspect_and_embed(new_paths)
                except Exception as e:
                    logger.debug(f"[ClipKeyframes] sampling failed: {e}")

        # 5. Emergency trim if over budget
        if len(self.working_memory) > self.max_working_memory_chars:
            pre_trim = len(self.working_memory)
            self.working_memory = self._emergency_trim(
                self.working_memory, self.max_working_memory_chars
            )
            logger.warning(
                f"[Orchestrator] Emergency trim: {pre_trim} → {len(self.working_memory)} chars"
            )

        # 6. Snapshot working memory for delta detection
        if self._fs_assembler:
            self._fs_assembler.snapshot_memory(iteration, self.working_memory)

        # 7. Analysis tracer: record section-level diff + FS sub-loop activity
        if self._tracer is not None:
            try:
                self._tracer.record_step(
                    iteration=iteration,
                    tool_name=tool_name,
                    memory_before=wm_before_text,
                    memory_after=self.working_memory,
                    subloop_calls=self._last_subloop_calls,
                    prompt_tokens=self._last_prompt_tokens,
                    output_tokens=self._last_output_tokens,
                )
            except Exception as e:  # tracer must never break the eval
                logger.warning(f"[Orchestrator] tracer.record_step failed: {e}")

        return self.working_memory

    def get_working_memory(self) -> str:
        """Return current working memory content for the agent."""
        return self.working_memory

    def _create_manifest_entry(
        self, tool_name: str, tool_input: Dict, tool_output: str,
        iteration: int, thought: str,
    ) -> ManifestEntry:
        """Create a manifest entry from a tool call."""
        args_summary = self._summarize_args(tool_name, tool_input)
        result_summary = self._summarize_result(tool_name, tool_output)
        importance = self._score_importance(tool_name, tool_input, tool_output)
        ts_range = self._extract_timestamp_range(tool_name, tool_input)

        return ManifestEntry(
            iteration=iteration,
            tool_name=tool_name,
            tool_args_summary=args_summary,
            result_summary=result_summary,
            importance=importance,
            timestamp_range=ts_range,
            thought_snippet=thought[:200] if thought else "",
        )

    def _summarize_args(self, tool_name: str, tool_input: Dict) -> str:
        """Produce concise args summary."""
        if tool_name in ("analyze_frames", "describe_frames"):
            paths = tool_input.get("image_paths", []) or tool_input.get("frame_paths", [])
            q = tool_input.get("question", "")
            return f"{len(paths)} frames, q='{q[:80]}'"
        elif tool_name == "analyze_clip":
            start = float(tool_input.get("start_time", 0))
            end = float(tool_input.get("end_time", 0))
            q = tool_input.get("question", "")
            return f"clip {start:.0f}s-{end:.0f}s, q='{q[:80]}'"
        elif tool_name == "execute_bash":
            cmd = tool_input.get("command", "")
            return cmd[:120]
        elif tool_name == "create_file":
            return tool_input.get("path", "")[:100]
        elif tool_name == "transcribe_audio":
            return "full video"
        else:
            return json.dumps(tool_input)[:120]

    def _summarize_result(self, tool_name: str, output: str) -> str:
        """Produce a 1-3 sentence result summary (rule-based, fast)."""
        if not output:
            return "(empty output)"
        # For short outputs, use as-is
        if len(output) < 200:
            return output.replace("\n", " ").strip()
        # For longer outputs, take first meaningful lines
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        # Skip common noise lines
        meaningful = [
            ln for ln in lines
            if not ln.startswith(("#", "---", "===", "WARNING", "INFO"))
            and len(ln) > 10
        ]
        if not meaningful:
            meaningful = lines
        summary = " ".join(meaningful[:3])
        if len(summary) > 300:
            summary = summary[:297] + "..."
        return summary

    def _score_importance(self, tool_name: str, tool_input: Dict,
                          output: str) -> float:
        """Score importance of this step for manifest compression."""
        if tool_name == "analyze_frames":
            paths = tool_input.get("image_paths", []) or tool_input.get("frame_paths", [])
            return 0.8 if len(paths) >= 3 else 0.7
        elif tool_name == "analyze_clip":
            return 0.8
        elif tool_name == "transcribe_audio":
            return 0.6
        elif tool_name == "execute_bash":
            cmd = tool_input.get("command", "")
            if self._FFMPEG_RE.search(cmd):
                return 0.3
            return 0.4
        elif tool_name == "create_file":
            return 0.2
        return 0.5

    def _extract_timestamp_range(
        self, tool_name: str, tool_input: Dict,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract temporal range from tool input."""
        if tool_name == "analyze_clip":
            st = tool_input.get("start_time")
            et = tool_input.get("end_time")
            return (
                float(st) if st is not None else None,
                float(et) if et is not None else None,
            )
        # For frame-based tools, try to parse timestamps from paths
        paths = (
            tool_input.get("image_paths", [])
            or tool_input.get("frame_paths", [])
        )
        if paths:
            timestamps = []
            for fp in paths:
                m = re.search(r"_t(\d+\.?\d*)", fp)
                if m:
                    timestamps.append(float(m.group(1)))
            if timestamps:
                return (min(timestamps), max(timestamps))
        return (None, None)

    # -- Agentic orchestrator tools --

    def _render_system_prompt(self) -> str:
        """Render the orchestrator system prompt with placeholders filled."""
        system_prompt = self.prompts.get("system_prompt", "")
        system_prompt = system_prompt.replace("{max_chars}", str(self.max_working_memory_chars))
        return system_prompt

    def _define_orchestrator_tools(self) -> List[Dict]:
        """Return tool schemas for the orchestrator's agentic loop.

        Tool surface: ``read_file`` (and the implicit ``inspect_frames``
        handled in ``_dispatch_orchestrator_tool``) for filesystem retrieval.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a file from the Docker container filesystem. "
                        "Use for transcripts, frame lists, analysis files, etc. "
                        "Paths are absolute within the container (e.g., /outputs/transcript.txt)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute path to the file in the container",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to read (default 4000)",
                                "default": 4000,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

    def _sample_clip_keyframes(self, start_time: float, end_time: float,
                                n_frames: int = 30) -> List[str]:
        """Extract N keyframes from a video clip's time range.

        Runs ffmpeg in the Docker container to extract frames at evenly-spaced
        intervals within [start_time, end_time]. Returns the container paths.
        The extracted frames use the standard frame_t{seconds}.jpg naming so
        _auto_inspect_and_embed can parse their timestamps.
        """
        if not self._docker_runtime or end_time <= start_time:
            return []
        duration = end_time - start_time
        # Scale frame count by clip length — don't over-sample very short clips
        if duration < 5:
            n_frames = min(n_frames, max(2, int(duration)))
        elif duration < 15:
            n_frames = min(n_frames, 10)
        elif duration < 60:
            n_frames = min(n_frames, 20)
        # Longer clips get the full n_frames (default 30)

        # Evenly space timestamps within the clip (avoid exact boundaries)
        step = duration / (n_frames + 1)
        timestamps = [start_time + step * (i + 1) for i in range(n_frames)]

        # Find the video path — try common locations
        video_path = None
        exit_code, ls = self._docker_runtime.execute(
            "ls /videos/*.mp4 2>/dev/null | head -1", timeout=5
        )
        if exit_code == 0 and ls.strip():
            video_path = ls.strip()
        if not video_path:
            return []

        # Ensure output directory exists
        self._docker_runtime.execute("mkdir -p /outputs", timeout=5)

        # Extract each keyframe (container paths only — read happens later)
        # Use frame_t{seconds}.jpg naming so _auto_inspect_and_embed can parse
        # timestamps and existing frame-cache logic finds them.
        frame_paths = []
        for t in timestamps:
            fname = f"frame_t{int(round(t))}.jpg"
            fpath = f"/outputs/{fname}"
            cmd = (
                f"test -f {fpath} || "
                f"ffmpeg -y -ss {t:.2f} -i {video_path} "
                f"-frames:v 1 -q:v 3 {fpath} 2>/dev/null"
            )
            ec, _ = self._docker_runtime.execute(cmd, timeout=30)
            if ec == 0:
                frame_paths.append(fpath)

        logger.info(
            f"[ClipKeyframes] Sampled {len(frame_paths)} keyframes from "
            f"clip {start_time:.0f}s-{end_time:.0f}s"
        )
        return frame_paths

    def _auto_inspect_and_embed(self, frame_paths: List[str]) -> None:
        """Automatically inspect new frames and embed the most relevant one.

        Called after analyze_frames — reads frames from Docker/cache, asks
        the model to select the best one, and injects it into Key Evidence.
        """
        # Read frame base64 from cache or Docker
        frame_parts = []  # (timestamp, path, base64)
        for path in frame_paths:
            self._inspected_frames.add(path)
            t_match = re.search(r'frame_t(\d+(?:\.\d+)?)', path)
            t = float(t_match.group(1)) if t_match else 0.0

            b64 = self._frame_cache.get(path)
            if not b64:
                # Fuzzy match by filename
                fname = os.path.basename(path)
                for cached_path, cached_b64 in self._frame_cache.items():
                    if os.path.basename(cached_path) == fname:
                        b64 = cached_b64
                        break
            if not b64 and self._docker_runtime:
                ec, b64_raw = self._docker_runtime.execute(f"base64 -w0 {path} 2>/dev/null", timeout=5)
                if ec == 0 and b64_raw and not b64_raw.startswith("[STDERR]"):
                    b64 = compress_frame_b64(b64_raw.strip())
                    self._frame_cache[path] = b64
            if b64:
                frame_parts.append((t, path, b64))

        if not frame_parts:
            logger.debug("[FrameEmbed] No readable frames for auto-inspect")
            return

        # Ask model to select the best frame
        selected_path, description = self._select_best_frame(frame_parts)
        if not selected_path:
            return

        # Inject into Key Evidence with description
        t_match = re.search(r'frame_t(\d+(?:\.\d+)?)', selected_path)
        t_str = f"{float(t_match.group(1)):.0f}s" if t_match else "analysis"
        md_tag = f"![{description[:80]}]({selected_path})"

        if selected_path not in self.working_memory:
            entry = f"- [{t_str}] {description}\n  {md_tag}"
            if "## Key Evidence" in self.working_memory:
                self.working_memory = self.working_memory.replace(
                    "## Key Evidence",
                    f"## Key Evidence\n{entry}",
                    1
                )
            else:
                self.working_memory += f"\n## Key Evidence\n{entry}\n"
            logger.info(f"[FrameEmbed] Auto-inspect selected {selected_path} from {len(frame_parts)} candidates: {description[:100]}")

    def _select_best_frame(self, frame_parts: List[tuple]) -> tuple:
        """Ask the model which frame is most relevant and why.

        Args:
            frame_parts: list of (timestamp, path, base64) tuples
        Returns:
            (path, description) tuple. Description explains why this frame matters.
        """
        from ..llm_api import openai_completion
        if not frame_parts or not self.question:
            return (frame_parts[0][1], "First available frame") if frame_parts else (None, "")

        # Cap to 30 frames for selection to keep the payload manageable
        if len(frame_parts) > 30:
            # Sample evenly
            step = len(frame_parts) / 30
            frame_parts = [frame_parts[int(i * step)] for i in range(30)]

        # Build multimodal prompt with all candidate frames
        prompt_parts = [{
            "type": "text",
            "text": (
                f"Question: {self.question}\n"
                f"Current understanding: {self.working_memory[:500]}\n\n"
                f"Below are {len(frame_parts)} video frames. "
                f"Select the ONE frame most relevant to answering the question.\n"
                f"Reply in this exact format:\n"
                f"PATH: <exact file path>\n"
                f"REASON: <one sentence explaining what this frame shows and why it's relevant>\n"
            ),
        }]
        for t, path, b64 in frame_parts:
            prompt_parts.append({"type": "text", "text": f"[{path} at {t:.0f}s]"})
            prompt_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        try:
            resp = openai_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt_parts}],
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=2048,
                timeout=self.timeout,
            )
            text = resp.get("content", "").strip()

            # Parse PATH and REASON
            import re as _re
            path_match = _re.search(r'PATH:\s*(.+)', text)
            reason_match = _re.search(r'REASON:\s*(.+)', text)
            description = reason_match.group(1).strip() if reason_match else "Visual evidence"

            # Match path from candidates
            selected = None
            if path_match:
                raw_path = path_match.group(1).strip()
                for _, path, _ in frame_parts:
                    if path in raw_path or os.path.basename(path) in raw_path:
                        selected = path
                        break
            if not selected:
                # Fallback: check entire response for any path match
                for _, path, _ in frame_parts:
                    if path in text or os.path.basename(path) in text:
                        selected = path
                        break
            if not selected:
                selected = frame_parts[0][1]
                description = "First available frame (model selection unclear)"
                logger.warning(f"[FrameEmbed] Model response didn't match any path: {text[:150]}")

            return (selected, description)
        except Exception as e:
            logger.warning(f"[FrameEmbed] Frame selection failed: {e}")
            return (frame_parts[0][1], "First available frame (selection error)")

    def _dispatch_orchestrator_tool(self, name: str, args: Dict) -> str:
        """Execute an orchestrator tool and return the result string."""
        # Trace the call before dispatch so it's captured even if dispatch raises.
        self._last_subloop_calls.append({"tool": name, "args": self._summarize_args(name, args)})
        try:
            if name == "read_file":
                if not self._docker_runtime:
                    return "Error: no Docker runtime available"
                path = args.get("path", "")
                max_chars = min(args.get("max_chars", 4000), 8000)
                exit_code, output = self._docker_runtime.execute(
                    f"head -c {max_chars} '{path}'", timeout=5
                )
                if exit_code != 0:
                    return f"Error reading {path}: {output[:500]}"
                return output[:max_chars]

            elif name == "inspect_frames":
                if not self._docker_runtime:
                    return "Error: no Docker runtime available"
                start = args.get("start_seconds", 0)
                end = args.get("end_seconds", 0)
                max_frames = min(args.get("max_frames", 6), 6)

                # Find frame files in the container matching this interval
                exit_code, output = self._docker_runtime.execute(
                    "find /outputs -name 'frame_t*.jpg' -o -name 'frame_t*.png' | sort", timeout=5
                )
                if exit_code != 0 or not output.strip():
                    return f"No frames found for interval [{start:.0f}-{end:.0f}s]"

                # Filter by timestamp
                matching = []
                for path in output.strip().split('\n'):
                    t_match = re.search(r'frame_t(\d+(?:\.\d+)?)', path)
                    if t_match:
                        t = float(t_match.group(1))
                        if start <= t <= end:
                            matching.append((t, path.strip()))

                if not matching:
                    return f"No frames found in [{start:.0f}-{end:.0f}s]"

                # Sample down to max_frames
                if len(matching) > max_frames:
                    step = len(matching) / max_frames
                    matching = [matching[int(i * step)] for i in range(max_frames)]

                # Track served frame paths for base64→markdown recovery
                self._last_viewed_frames = [path for _, path in matching]
                logger.info(f"[FrameEmbed] inspect_frames [{start:.0f}-{end:.0f}s]: serving {len(matching)} frames: {[p for _,p in matching]}")

                # Build multimodal content with all frames
                frame_parts = []  # (t, path, b64) for selection call
                content_parts = [{"type": "text", "text": (
                    f"Frames from [{start:.0f}s-{end:.0f}s] ({len(matching)} frames)."
                )}]

                for t, path in matching:
                    exit_code, b64_data = self._docker_runtime.execute(
                        f"base64 -w0 {path} 2>/dev/null", timeout=5
                    )
                    if exit_code == 0 and b64_data and not b64_data.startswith("[STDERR]"):
                        b64_clean = compress_frame_b64(b64_data.strip())
                        self._frame_cache[path] = b64_clean
                        frame_parts.append((t, path, b64_clean))
                        content_parts.append({"type": "text", "text": f"[{path} at {t:.0f}s]"})
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_clean}"}
                        })

                # Ask the model to select the most relevant frame
                if frame_parts:
                    selected_path, description = self._select_best_frame(frame_parts)
                    if selected_path:
                        t_match = re.search(r'frame_t(\d+(?:\.\d+)?)', selected_path)
                        t_str = f"{float(t_match.group(1)):.0f}s" if t_match else f"{start:.0f}-{end:.0f}s"
                        md_tag = f"![{description[:80]}]({selected_path})"
                        if selected_path not in self.working_memory:
                            entry = f"- [{t_str}] {description}\n  {md_tag}"
                            if "## Key Evidence" in self.working_memory:
                                self.working_memory = self.working_memory.replace(
                                    "## Key Evidence",
                                    f"## Key Evidence\n{entry}",
                                    1
                                )
                            else:
                                self.working_memory += f"\n## Key Evidence\n{entry}\n"
                            logger.info(f"[FrameEmbed] Model selected {selected_path} from {len(frame_parts)} candidates: {description[:100]}")

                return content_parts  # Multimodal return for orchestrator's loop


            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Tool error ({name}): {e}"

    # -- Orchestrator LLM call --

    def _call_orchestrator(
        self,
        tool_name: str,
        tool_input: Dict,
        tool_output: str,
        iteration: int,
        thought: str,
        entry: ManifestEntry,
    ) -> Optional[str]:
        """Call the orchestrator LLM to produce updated working memory."""
        _completion = _get_completion()

        # Reset tracer buffers for this step.
        self._last_subloop_calls = []
        self._last_prompt_tokens = 0
        self._last_output_tokens = 0

        is_bootstrap = not self.working_memory or "Analysis has not yet begun" in self.working_memory
        current_len = len(self.working_memory)

        mode = "bootstrap" if is_bootstrap else "edit"
        logger.info(
            f"[Orchestrator] LLM call mode={mode} wm_len={current_len} "
            f"fs_assembler={'yes' if self._fs_assembler else 'no'}"
        )

        # Build prompt — use _safe_format to avoid KeyError from literal braces
        # in working memory, tool output, or manifest content
        def _safe_format(template: str, **kwargs) -> str:
            """Replace {key} placeholders without interpreting other braces."""
            result = template
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result

        system_prompt = self._render_system_prompt()

        # Prepend genre context so orchestrator curates memory with the right lens
        effective_question = self.question
        if self.genre_context:
            effective_question = f"[{self.genre_context}]\n{self.question}"

        if is_bootstrap:
            template = self.prompts.get("user_template_bootstrap", "")
            dur = "?"
            res = "?"
            if self._video_info:
                dur = self._video_info.get("duration_seconds", "?")
                vid = self._video_info.get("video", {})
                res = f"{vid.get('width', '?')}x{vid.get('height', '?')}" if vid else "?"
            user_msg = _safe_format(
                template,
                question=effective_question,
                options=self.options,
                duration=dur,
                resolution=res,
                iteration=iteration,
                tool_name=tool_name,
                tool_args=self._summarize_args(tool_name, tool_input),
                tool_output=tool_output[:16000],
            )
            logger.info(
                f"[Orchestrator] bootstrap prompt: tool_output={len(tool_output)} chars (truncated to 8K)"
            )
        else:
            template = self.prompts.get("user_template_edit", "")
            fs_context = ""
            if self._fs_assembler:
                fs_context = self._fs_assembler.assemble(
                    mode="edit", tool_name=tool_name, tool_input=tool_input,
                    current_memory=self.working_memory, iteration=iteration,
                )
                if fs_context:
                    fs_context = f"FILESYSTEM CONTEXT:\n{fs_context}"
            manifest_text = self.manifest.render_for_orchestrator(max_chars=16000)
            user_msg = _safe_format(
                template,
                question=effective_question,
                options=self.options,
                working_memory=self.working_memory,
                iteration=iteration,
                tool_name=tool_name,
                tool_args=self._summarize_args(tool_name, tool_input),
                thought=thought[:4000] if thought else "(none)",
                tool_output=tool_output[:16000],
                manifest=manifest_text,
                fs_context=fs_context,
            )
            logger.info(
                f"[Orchestrator] edit prompt: wm={len(self.working_memory)} "
                f"tool_output={len(tool_output)} manifest={len(manifest_text)} "
                f"fs_context={len(fs_context)} thought={len(thought) if thought else 0} "
                f"total_msg={len(user_msg)} chars"
            )

        try:
            self.calls_made += 1
            # Strip any {{IMAGE:...}} base64 markers — these should never appear in text
            user_msg = re.sub(r'\{\{IMAGE:data:image/[^}]+\}\}', '', user_msg)

            # Only resolve question images for the orchestrator (not frame images).
            # Frame images are expensive and the orchestrator can use
            # inspect_frames when it needs to see them.
            # The markdown text ![caption](/path) stays visible so the orchestrator
            # knows what's embedded for the main agent.
            if self._question_images:
                # Resolve only ![...](/question_image.png) to actual image
                qi_pattern = re.compile(r'(!\[[^\]]*\]\(/question_image\.png\))')
                qi_matches = list(qi_pattern.finditer(user_msg))
                if qi_matches:
                    user_content = []
                    last_end = 0
                    for match in qi_matches:
                        # Keep text including the markdown syntax
                        if match.end() > last_end:
                            user_content.append({"type": "text", "text": user_msg[last_end:match.end()]})
                        last_end = match.end()
                        # Inject actual question image after
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self._question_images[0]}"},
                        })
                    if last_end < len(user_msg):
                        user_content.append({"type": "text", "text": user_msg[last_end:]})
                else:
                    # No markdown reference to question image — append at end
                    user_content = [{"type": "text", "text": user_msg}]
                    for img_b64 in self._question_images:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        })
            else:
                user_content = user_msg

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # Decide whether to use agentic loop or single-pass.
            use_agentic = (
                self.max_loop_iterations > 0
                and self._docker_runtime is not None
            )
            tools = self._define_orchestrator_tools() if use_agentic else None

            raw = None
            loop_count = self.max_loop_iterations if use_agentic else 1

            for loop_i in range(loop_count):
                # Strip image blocks from older tool results to prevent context explosion
                # (inspect_frames returns multimodal content that accumulates)
                if loop_i > 0:
                    for msg in messages[:-2]:  # Keep last 2 messages (most recent tool result)
                        content = msg.get("content")
                        if isinstance(content, list):
                            msg["content"] = [
                                b for b in content
                                if not (isinstance(b, dict) and b.get("type") == "image_url")
                            ]

                resp = _completion(
                    model=self.model,
                    messages=messages,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    temperature=1.0,
                    timeout=self._raw_timeout,
                    tools=tools,
                    reasoning_effort=self.reasoning_effort,
                )
                usage = resp.get("usage", {})
                if self.token_tracker and usage:
                    self.token_tracker.record(
                        "summarizer", self.model,
                        usage.get("promptTokenCount", 0) or 0,
                        (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0),
                        call_site=f"orchestrator.{'loop' if use_agentic else 'process'}",
                    )

                content = resp.get("content", "")
                tool_calls = resp.get("tool_calls", [])
                prompt_tokens = usage.get("promptTokenCount", 0) or 0
                output_tokens = (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0)
                thoughts_tokens = usage.get("thoughtsTokenCount", 0) or 0

                self.cumulative_prompt_tokens += prompt_tokens
                self._last_prompt_tokens += prompt_tokens
                self._last_output_tokens += output_tokens
                logger.info(
                    f"[Orchestrator] loop={loop_i} prompt={prompt_tokens:,} "
                    f"output={output_tokens:,} thinking={thoughts_tokens:,} "
                    f"tool_calls={len(tool_calls)} content={len(content)} chars "
                    f"cumulative={self.cumulative_prompt_tokens:,}"
                )

                # Warn if token usage is high
                if prompt_tokens > 500_000:
                    logger.warning(
                        f"[Orchestrator] High token usage: {prompt_tokens:,} prompt tokens in single call"
                    )

                if not tool_calls:
                    # No tools -> text is the final answer
                    raw = content.strip()
                    if use_agentic and loop_i > 0:
                        logger.info(
                            f"[Orchestrator] Agentic loop finished after {loop_i + 1} iterations "
                            f"({self.tool_calls_dispatched} tool calls)"
                        )
                    break

                # Append assistant message with tool_calls
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                # Execute each tool call
                for tc in tool_calls:
                    tc_name = tc.get("function", {}).get("name", "")
                    try:
                        tc_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {}
                    result = self._dispatch_orchestrator_tool(tc_name, tc_args)
                    # Handle multimodal results (e.g. inspect_frames returns list)
                    if isinstance(result, list):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": tc_name,
                            "content": result,
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": tc_name,
                            "content": result[:4000],
                        })
                    self.tool_calls_dispatched += 1
                    if isinstance(result, list):
                        img_count = sum(1 for b in result if isinstance(b, dict) and b.get("type") == "image_url")
                        logger.info(
                            f"[Orchestrator] Tool {tc_name}({tc_args}) → "
                            f"multimodal: {len(result)} blocks, {img_count} images"
                        )
                    else:
                        logger.info(
                            f"[Orchestrator] Tool {tc_name}({tc_args}) → "
                            f"{len(result)} chars"
                        )

                self.tool_loop_iterations += 1
            else:
                # Budget exhausted — extract text from last response
                if raw is None:
                    raw = (msg.content or "").strip()
                    logger.warning(
                        f"[Orchestrator] Agentic loop exhausted budget ({loop_count} iterations)"
                    )

            if not raw:
                logger.warning(f"[Orchestrator] LLM returned empty response")
                return None

            logger.debug(f"[Orchestrator] LLM response ({len(raw)} chars): {raw[:200]}...")

            # Parse response — bootstrap returns raw markdown, edit parses commands
            if mode == "bootstrap":
                if raw.startswith("```"):
                    raw = re.sub(r"^```\w*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                return raw
            else:
                result = self._apply_edits(raw)
                if result is None:
                    logger.warning(f"[Orchestrator] Edit parse failed, raw starts with: {raw[:100]}")
                return result

        except Exception as e:
            self.calls_failed += 1
            logger.warning(f"[Orchestrator] LLM call failed: {e}")
            return None

    def _apply_edits(self, raw: str) -> Optional[str]:
        """Parse UPDATE/APPEND/DELETE instructions and apply to working memory.

        Thin wrapper that uses ``self.working_memory`` as the base. Real
        parsing/applying lives in ``_apply_edits_to``.
        """
        result, _ = self._apply_edits_to(self.working_memory, raw)
        return result

    def _apply_edits_to(self, base: str, raw: str) -> Tuple[Optional[str], int]:
        """Apply UPDATE/APPEND/DELETE commands in ``raw`` to ``base``.

        Returns ``(new_text, edit_count)``. ``new_text`` is None for malformed
        output that we couldn't accept. ``edit_count`` is the number of
        UPDATE/APPEND/DELETE blocks we recognized; 0 means a full-rewrite path
        was taken (``new_text`` will be the full-rewrite result if accepted).
        """
        has_commands = any(
            raw.strip().startswith(cmd) or f"\n{cmd}" in raw
            for cmd in ("UPDATE ", "APPEND ", "DELETE ")
        )
        if not has_commands:
            # Model output a full rewrite instead of edits
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            if raw.startswith("#") or "## " in raw[:200]:
                # Only accept if it preserves existing sections (not a partial rewrite)
                existing_sections = re.findall(r"^## .+", base, re.MULTILINE)
                preserved = sum(1 for s in existing_sections if s in raw)
                if preserved >= len(existing_sections) * 0.6:
                    # Preserve auto-embedded frame images from old memory
                    old_images = re.findall(
                        r'  !\[[^\]]*\]\(/(?:outputs|tmp)/[^)]*\.(?:jpg|png)\)',
                        base
                    )
                    for img_line in old_images:
                        img_path = re.search(r'\((/[^)]+)\)', img_line)
                        if img_path and img_path.group(1) not in raw:
                            # Find Key Evidence section to append to
                            if "## Key Evidence" in raw:
                                raw = raw.replace("## Key Evidence", f"## Key Evidence\n{img_line}", 1)
                            else:
                                raw += f"\n{img_line}\n"
                    logger.info(f"[Orchestrator] Edit mode got full rewrite ({len(raw)} chars, {preserved}/{len(existing_sections)} sections preserved)")
                    return raw, 0
                else:
                    logger.warning(
                        f"[Orchestrator] Rejected partial rewrite: only {preserved}/{len(existing_sections)} sections preserved"
                    )
                    return None, 0
            return None, 0  # Malformed output

        result = base

        # Parse commands: UPDATE/APPEND/DELETE ## Section\n...\nEND
        # Content ends at END marker, next command, or end of string
        pattern = re.compile(
            r"(UPDATE|APPEND|DELETE)\s+(##\s+[^\n]+)\n(.*?)(?=\nEND\b|\n(?:UPDATE|APPEND|DELETE)\s+##|\Z)",
            re.DOTALL,
        )

        edit_count = 0
        for match in pattern.finditer(raw):
            command = match.group(1)
            section_header = match.group(2).strip()
            content = match.group(3).strip()
            edit_count += 1
            logger.debug(f"[Orchestrator] Applying {command} on '{section_header}' ({len(content)} chars)")

            if section_header not in result:
                if command == "APPEND" or command == "UPDATE":
                    # Section doesn't exist yet — create it
                    result = result.rstrip() + f"\n\n{section_header}\n{content}\n"
                continue

            # Find section boundaries
            sec_start = result.index(section_header)
            sec_content_start = sec_start + len(section_header)
            # Find next section or end
            next_match = re.search(r"\n## ", result[sec_content_start + 1:])
            if next_match:
                sec_end = sec_content_start + 1 + next_match.start()
            else:
                sec_end = len(result)

            if command == "UPDATE":
                # Preserve auto-embedded frame images that the orchestrator may have dropped
                old_section = result[sec_content_start:sec_end]
                preserved_images = re.findall(
                    r'  !\[[^\]]*\]\(/(?:outputs|tmp)/[^)]*\.(?:jpg|png)\)',
                    old_section
                )
                if preserved_images:
                    # Check which image lines are missing from new content
                    for img_line in preserved_images:
                        img_path = re.search(r'\((/[^)]+)\)', img_line)
                        if img_path and img_path.group(1) not in content:
                            content = content.rstrip() + "\n" + img_line
                result = (
                    result[:sec_content_start] + "\n" + content + "\n"
                    + result[sec_end:]
                )
            elif command == "APPEND":
                # Insert before the end of the section
                insert_point = sec_end
                result = (
                    result[:insert_point].rstrip() + "\n" + content + "\n"
                    + result[insert_point:]
                )
            elif command == "DELETE":
                # Remove specific lines from the section
                section_text = result[sec_content_start:sec_end]
                for line_to_delete in content.splitlines():
                    line_to_delete = line_to_delete.strip()
                    if line_to_delete and line_to_delete in section_text:
                        section_text = section_text.replace(line_to_delete, "", 1)
                # Clean up empty lines
                section_text = re.sub(r"\n{3,}", "\n\n", section_text)
                result = result[:sec_content_start] + section_text + result[sec_end:]

        if edit_count:
            logger.info(f"[Orchestrator] Applied {edit_count} edit(s), wm: {len(base)} → {len(result)} chars")
        return result, edit_count

    @staticmethod
    def _deduplicate_sections(content: str) -> str:
        """Remove duplicate ## sections, keeping the first occurrence of each."""
        lines = content.split("\n")
        sections: list[tuple[str, int, int]] = []  # (header, start_line, end_line)
        current_header = None
        current_start = 0

        for i, line in enumerate(lines):
            if line.startswith("## "):
                if current_header is not None:
                    sections.append((current_header, current_start, i))
                current_header = line.strip()
                current_start = i
        if current_header is not None:
            sections.append((current_header, current_start, len(lines)))

        if not sections:
            return content

        # Check for duplicates
        seen: dict[str, int] = {}
        duplicates = []
        for idx, (header, start, end) in enumerate(sections):
            if header in seen:
                duplicates.append(idx)
            else:
                seen[header] = idx

        if not duplicates:
            return content

        # Remove duplicate sections (keep first occurrence)
        lines_to_remove: set[int] = set()
        for dup_idx in duplicates:
            _, start, end = sections[dup_idx]
            for line_num in range(start, end):
                lines_to_remove.add(line_num)

        result_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
        result = "\n".join(result_lines)
        # Clean up excess blank lines
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        logger.info(
            f"[Orchestrator] Deduplicated {len(duplicates)} section(s): "
            f"{[sections[d][0] for d in duplicates]}"
        )
        return result

    def _fallback_append(self, entry: ManifestEntry) -> None:
        """Mechanical fallback when orchestrator fails."""
        ts = ""
        if entry.timestamp_range[0] is not None and entry.timestamp_range[1] is not None:
            ts = f" [{entry.timestamp_range[0]:.0f}-{entry.timestamp_range[1]:.0f}s]"
        line = (
            f"\n### Step {entry.iteration}: {entry.tool_name}{ts}\n"
            f"- {entry.result_summary[:500]}\n"
        )

        if "## Key Evidence" in self.working_memory:
            # Find end of Key Evidence section
            idx = self.working_memory.index("## Key Evidence")
            next_section = self.working_memory.find("\n## ", idx + 1)
            if next_section == -1:
                self.working_memory += line
            else:
                self.working_memory = (
                    self.working_memory[:next_section].rstrip()
                    + line + "\n"
                    + self.working_memory[next_section:]
                )
        else:
            self.working_memory += line

    @staticmethod
    def _emergency_trim(content: str, max_chars: int) -> str:
        """Progressive trimming when over budget."""
        if len(content) <= max_chars:
            return content

        # Step 1: Trim Key Evidence to last 10 entries
        if "## Key Evidence" in content:
            start = content.index("## Key Evidence")
            next_sec = re.search(r"\n## ", content[start + 1:])
            if next_sec:
                sec_end = start + 1 + next_sec.start()
                sec_body = content[start:sec_end]
                lines = sec_body.splitlines()
                header = lines[0]
                evidence_lines = [ln for ln in lines[1:] if ln.strip()]
                if len(evidence_lines) > 10:
                    evidence_lines = evidence_lines[-10:]
                    trimmed_sec = header + "\n" + "\n".join(evidence_lines) + "\n"
                    content = content[:start] + trimmed_sec + content[sec_end:]

        if len(content) <= max_chars:
            return content

        # Step 2: Truncate Current Understanding to 3 lines
        if "## Current Understanding" in content:
            start = content.index("## Current Understanding")
            next_sec = re.search(r"\n## ", content[start + 1:])
            if next_sec:
                sec_end = start + 1 + next_sec.start()
                header = "## Current Understanding"
                sec_body = content[start + len(header):sec_end].strip()
                sentences = re.split(r"(?<=[.!?])\s+", sec_body)
                if len(sentences) > 3:
                    trimmed = " ".join(sentences[:3])
                    content = content[:start] + header + "\n" + trimmed + "\n\n" + content[sec_end:]

        if len(content) <= max_chars:
            return content

        # Step 3: Hard truncate
        content = content[:max_chars - 50] + "\n\n[memory truncated]"
        return content

    def render_stats(self) -> str:
        """Render orchestrator stats."""
        rendered = len(self.working_memory)
        ratio = self.total_raw_chars / max(rendered, 1)
        agentic_info = ""
        if self.max_loop_iterations > 0:
            agentic_info = (
                f", agentic_loops={self.tool_loop_iterations}, "
                f"orch_tools={self.tool_calls_dispatched}"
            )
        return (
            f"Orchestrator: manifest={len(self.manifest.entries)} entries, "
            f"absorbed={self.total_raw_chars}ch, rendered={rendered}ch, "
            f"compression={ratio:.1f}x, "
            f"calls={self.calls_made}/{self.calls_failed}failed"
            f"{agentic_info}"
        )

    # ------------------------------------------------------------------
    # Visual planner helpers (memory warm-up)
    # ------------------------------------------------------------------

    def call_visual_planner(
        self,
        question_id: str,
        question: str,
        options: Optional[List[str]],
        transcript: str = "",
        descriptions_dir: str = "",
        project_root: Optional[Path] = None,
        planner_model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[List[dict]]:
        """Make a planning API call to generate a todo list from visual descriptions.

        Reads the pre-generated description JSON for this question, condenses it
        to relevant segments and chunk summaries (plus an optional transcript excerpt),
        and asks the same model as the main agent to produce a prioritized list of
        timestamps/ideas to investigate.

        Returns a list of todo dicts {id, timestamp_hint, task}, or None on failure.
        """
        desc_dir = Path(descriptions_dir) if descriptions_dir else Path(".")
        if not desc_dir.is_absolute() and project_root:
            desc_dir = project_root / desc_dir
        desc_path = desc_dir / f"{question_id}.json"
        if not desc_path.exists():
            logger.info(f"Visual planner: no description file for {question_id}, skipping")
            return None

        try:
            with open(desc_path) as f:
                desc = json.load(f)
        except Exception as e:
            logger.warning(f"Visual planner: failed to read {desc_path}: {e}")
            return None

        # Condense description to relevant_segments + summaries (no structural wrapper)
        condensed = []
        for chunk in desc.get("chunks", []):
            start_s = chunk.get("start_seconds", 0)
            end_s = chunk.get("end_seconds", 0)
            for seg in chunk.get("relevant_segments", []):
                condensed.append({
                    "window": f"{start_s}s-{end_s}s",
                    "time": seg.get("time", ""),
                    "description": seg.get("description", ""),
                })
            if chunk.get("summary"):
                condensed.append({
                    "window": f"{start_s}s-{end_s}s",
                    "time": "chunk_summary",
                    "description": chunk["summary"],
                })

        opts_text = "\n".join(options) if options else "No options"

        # Truncate transcript to keep prompt size manageable (~5000 chars ≈ first ~5 min)
        transcript_section = ""
        if transcript:
            truncated = transcript[:5000]
            if len(transcript) > 5000:
                truncated += "\n... [transcript truncated]"
            transcript_section = f"\nVideo transcript (partial):\n{truncated}\n"

        prompt = (
            f"Question: {question}\n"
            f"Options:\n{opts_text}\n"
            f"{transcript_section}\n"
            f"Visual evidence from pre-scan of video segments:\n"
            f"{json.dumps(condensed, ensure_ascii=False)}\n\n"
            f"Generate a concise TODO list (3-7 items) of specific timestamps and visual "
            f"moments the agent should investigate to answer this question. Use both the "
            f"transcript and visual evidence to identify the most promising leads. "
            f"Be specific about what to look for at each timestamp.\n\n"
            f'Return ONLY a JSON array:\n'
            f'[{{"id": 1, "timestamp_hint": "0:50-1:30", "task": "Verify X by examining Y"}}, ...]\n'
            f"No explanation, no markdown — only the JSON array."
        )

        messages = [
            {"role": "system", "content": "You are a video analysis planner. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        model = planner_model or self.model
        completion = _get_completion()
        for attempt in range(3):
            try:
                resp = openai_completion(
                    model=model,
                    messages=messages,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    timeout=120,
                )
                usage = resp.get("usage", {})
                if self.token_tracker:
                    self.token_tracker.record(
                        agent_type="visual_planner",
                        model=model,
                        input_tokens=usage.get("promptTokenCount", 0) or 0,
                        output_tokens=(usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0),
                        call_site="plan",
                    )
                text = resp.get("content", "") or ""
                text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
                m = re.search(r"\[.*\]", text, re.DOTALL)
                if m:
                    todos = json.loads(m.group())
                    if isinstance(todos, list) and todos:
                        logger.info(f"Visual planner: generated {len(todos)} todos for {question_id}")
                        return todos
                break  # Got a response but couldn't parse — don't retry
            except Exception as e:
                logger.warning(f"Visual planner call failed for {question_id} (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    import time as _time
                    _time.sleep(5 * (attempt + 1))
        return None

    def verify_todo(
        self,
        todo_item: dict,
        finding: str,
        question: str = "",
        options: Optional[List[str]] = None,
        planner_model: Optional[str] = None,
    ) -> tuple:
        """Ask the planner to verify whether the agent's finding is sufficient to close a todo.

        Returns (approved: bool, feedback: str).
        On any error, defaults to approved=True to avoid blocking the agent.
        """
        opts_text = "\n".join(options) if options else "No options"

        prompt = (
            f"Question the agent is answering: {question}\n"
            f"Options:\n{opts_text}\n\n"
            f"Todo item: \"{todo_item.get('task', '')}\" (at {todo_item.get('timestamp_hint', 'unknown time')})\n"
            f"Agent's finding: \"{finding}\"\n\n"
            f"Is this finding specific and sufficient to close this investigation item?\n"
            f"A good finding describes exactly what was seen, at what timestamp, and how it "
            f"relates to the question.\n\n"
            f'Reply with JSON only: {{"approved": true/false, "feedback": "one sentence"}}'
        )

        messages = [
            {"role": "system", "content": "You are a video analysis verifier. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        model = planner_model or self.model
        completion = _get_completion()
        for attempt in range(3):
            try:
                resp = openai_completion(
                    model=model,
                    messages=messages,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    max_tokens=256,
                    temperature=1.0,
                    timeout=60,
                )
                usage = resp.get("usage", {})
                if self.token_tracker:
                    self.token_tracker.record(
                        agent_type="visual_planner",
                        model=model,
                        input_tokens=usage.get("promptTokenCount", 0) or 0,
                        output_tokens=(usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0),
                        call_site="verify",
                    )
                text = resp.get("content", "") or ""
                text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    result = json.loads(m.group())
                    approved = bool(result.get("approved", True))
                    feedback = str(result.get("feedback", ""))
                    return approved, feedback
                break  # Got a response but couldn't parse — default to approved
            except Exception as e:
                logger.warning(f"Visual planner verification failed (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    import time as _time
                    _time.sleep(3 * (attempt + 1))
        # Default to approved on error so agent isn't blocked
        return True, ""
