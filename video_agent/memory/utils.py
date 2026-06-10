"""
Standalone utilities and data classes for the memory subsystem.

Contains:
- truncate_tool_output: First+last line truncation for long tool outputs.
- TemporalInterval: Time interval dataclass used by memory and file_memory.
"""

from dataclasses import dataclass
from typing import Optional

TRUNCATION_LINES = 40  # default: keep first 40 + last 40 lines


@dataclass
class TemporalInterval:
    """A time interval in the video, in seconds."""
    start: float
    end: float
    label: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "TemporalInterval") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def __repr__(self) -> str:
        label = f" ({self.label})" if self.label else ""
        return f"[{self.start:.1f}s-{self.end:.1f}s{label}]"


def truncate_tool_output(output: str, num_lines: int = TRUNCATION_LINES) -> str:
    """Truncate tool output keeping first+last ``num_lines`` lines.

    Matches the strategy in LLM-in-Sandbox (arxiv 2601.16206):
    if output exceeds 2*num_lines, keep head and tail with a marker in between.
    """
    if not output:
        return ""
    lines = output.splitlines()
    if len(lines) <= 2 * num_lines:
        return output
    top = "\n".join(lines[:num_lines])
    bottom = "\n".join(lines[-num_lines:])
    divider = "-" * 50
    return (
        f"{top}\n"
        f"{divider}\n"
        f"<Observation truncated in middle for saving context>\n"
        f"{divider}\n"
        f"{bottom}"
    )
