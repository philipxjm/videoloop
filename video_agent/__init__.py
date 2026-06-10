"""
Video Understanding Agent

A Docker-based agent system for solving long video understanding tasks
using a hierarchical agent architecture.
"""

__version__ = "0.1.0"

from .agent import VideoUnderstandingAgent, AgentResult, TrajectoryStep
from .docker_runtime import DockerRuntime
from .tools import TOOLS, format_tools_for_claude

__all__ = [
    "VideoUnderstandingAgent",
    "AgentResult",
    "TrajectoryStep",
    "DockerRuntime",
    "TOOLS",
    "format_tools_for_claude",
]

