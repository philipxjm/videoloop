"""
Transcript caching layer.

Handles in-memory caching, on-disk lookup, and compact formatting.
Container-based Whisper fallback is delegated back to the caller via
a callback.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_utils import get_logger

logger = get_logger(__name__)

# Derive default cache paths from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "dataset" / "videomme_long" / "transcripts_whisperx"
# Additional transcript directories to search
_EXTRA_CACHE_DIRS = [
    _PROJECT_ROOT / "dataset" / "videommmu" / "transcripts_whisperx",
]


# Global in-memory cache persisting across instances
# Format: {video_path: [segment_dicts]}
_global_cache: Dict[str, List[Dict[str, Any]]] = {}


def _normalize_segments(raw_segments: list) -> List[Dict[str, Any]]:
    """Normalise raw Whisper/JSON segments into [{start, end, text, speaker?}, ...]."""
    result = []
    for s in raw_segments:
        text = s.get("text", "").strip()
        if not text:
            continue
        entry: Dict[str, Any] = {
            "start": round(s.get("start", 0), 1),
            "end": round(s.get("end", 0), 1),
            "text": text,
        }
        if "speaker" in s:
            entry["speaker"] = s["speaker"]
        result.append(entry)
    return result


def format_transcript(segments: List[Dict[str, Any]]) -> str:
    """Format segments as compact timestamped lines with optional speaker labels.

    With speaker:    '[6s-9s] SPEAKER_00: Ah, it's blinding me.'
    Without speaker: '[12s-42s] spoken text here'
    """
    lines = []
    for s in segments:
        ts = f"[{s['start']:.0f}s-{s['end']:.0f}s]"
        speaker = s.get("speaker")
        if speaker:
            lines.append(f"{ts} {speaker}: {s['text']}")
        else:
            lines.append(f"{ts} {s['text']}")
    return "\n".join(lines)


class TranscriptCache:
    """Read-through cache for video transcripts."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        video_path: str,
        whisper_fallback: Optional[Callable[[str], Tuple[int, str]]] = None,
    ) -> Dict[str, Any]:
        """Look up a transcript.  Returns a result dict with *segments* and *transcript*.

        ``whisper_fallback`` is an optional ``(video_path) -> (exit_code, json_output)``
        callable used when no cached transcript is found.
        """
        video_name = os.path.basename(video_path)
        video_id = video_name.rsplit(".", 1)[0] if "." in video_name else video_name

        # 1. In-memory cache
        if video_path in _global_cache:
            segments = _global_cache[video_path]
            logger.info(f"Using in-memory cached transcript for {video_path} ({len(segments)} segments)")
            return self._result(segments)

        # 2. Segments-only cache file
        segments = self._load_segments_file(video_id)
        if segments is not None:
            _global_cache[video_path] = segments
            return self._result(segments)

        # 3. Whisper fallback
        if whisper_fallback is not None:
            segments = self._run_whisper(video_path, whisper_fallback)
            if segments is not None:
                _global_cache[video_path] = segments
                return self._result(segments)

        return {"segments_count": 0, "duration": 0, "transcript": "(no transcript available)"}

    @staticmethod
    def clear() -> None:
        """Clear the global in-memory cache."""
        _global_cache.clear()
        logger.info("Transcript cache cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_segments_file(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        # Search primary cache dir, then extra dirs
        candidates = [self.cache_dir / f"{video_id}.json"]
        for extra in _EXTRA_CACHE_DIRS:
            candidates.append(extra / f"{video_id}.json")
        path = None
        for c in candidates:
            if c.exists():
                path = c
                break
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # WhisperX format: {"segments": [...], ...} or legacy flat array [...]
            raw = data.get("segments", data) if isinstance(data, dict) else data
            segments = _normalize_segments(raw)
            logger.info(f"Loaded transcript from {path} ({len(segments)} segments)")
            return segments
        except Exception as e:
            logger.warning(f"Failed to load transcript from {path}: {e}")
            return None

    @staticmethod
    def _run_whisper(
        video_path: str,
        fallback: Callable[[str], Tuple[int, str]],
    ) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"Transcribing {video_path} with Whisper (not cached)")
        exit_code, output = fallback(video_path)
        if exit_code != 0:
            return None
        if "\n[STDERR]" in output:
            output = output.split("\n[STDERR]")[0]
        try:
            result = json.loads(output)
            raw = result.get("segments", []) if isinstance(result, dict) else result
            segments = _normalize_segments(raw)
            logger.info(f"Transcribed {video_path}: {len(segments)} segments")
            return segments
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _result(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not segments:
            return {"segments_count": 0, "duration": 0, "transcript": "(no speech detected)"}
        duration = segments[-1].get("end", 0)
        return {
            "segments_count": len(segments),
            "duration": duration,
            "transcript": format_transcript(segments),
            "cached": True,
            "note": "Transcript is cached. Call transcribe_audio again with the same video_path for instant retrieval.",
        }
