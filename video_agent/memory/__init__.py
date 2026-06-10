"""
Memory subsystem for Video Understanding Agent.

Contains:
- truncate_tool_output: First+last line truncation for long tool outputs.
- TemporalInterval: Time interval dataclass.
- ManifestEntry, StepManifest: Structured step log with progressive compression.
- FilesystemContextAssembler: Pre-assembles filesystem context for orchestrator.
- MemoryOrchestrator: LLM-powered memory manager that curates working memory.
"""

from .orchestrator import MemoryOrchestrator, FilesystemContextAssembler, ManifestEntry, StepManifest
from .utils import truncate_tool_output, TRUNCATION_LINES, TemporalInterval

__all__ = [
    "MemoryOrchestrator",
    "FilesystemContextAssembler",
    "ManifestEntry",
    "StepManifest",
    "TemporalInterval",
    "truncate_tool_output",
    "TRUNCATION_LINES",
]
