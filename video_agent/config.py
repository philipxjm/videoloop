"""Configuration loading for the video understanding agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelLimits:
    """Token limits and defaults for a specific model."""
    max_output_tokens: int = 8192
    temperature: float = 0.0


@dataclass
class MainAgentConfig:
    model: str = "gemini-3.1-flash-lite-preview"
    api_base: Optional[str] = None
    temperature: Optional[float] = None  # None = derive from models lookup
    max_output_tokens: int = 65536
    max_retries: int = 3
    timeout: int = 120
    thinking: bool = False  # Enable extended thinking
    thinking_budget: Optional[int] = None  # Max thinking tokens (None = model default)
    thinking_level: Optional[str] = None  # "low"/"high" for models that use thinking levels
    max_images: int = 32  # Max images per visual-analysis call
    multimodal_agent: bool = True  # Native multimodal: frames go directly to the main agent


@dataclass
class SandboxConfig:
    image: str = "video-understanding-sandbox"
    tag: str = "latest"
    gpu: bool = True
    memory_limit: str = "32g"
    shm_size: str = "8g"

    @property
    def full_image(self) -> str:
        return f"{self.image}:{self.tag}"


@dataclass
class MemoryConfig:
    """Hierarchical memory configuration."""
    enabled: bool = False
    max_sensory_entries: int = 40
    max_result_entries: int = 60
    max_working_entries: int = 20
    max_context_chars: int = 16000
    recent_messages_window: int = 8  # raw (assistant+tool) pairs to keep
    min_vlm_calls_before_submit: int = 3  # Must analyze frames with VLM at least N times
    min_iterations_before_submit: int = 6  # Cannot submit before this iteration
    require_transcript_before_submit: bool = True  # Must have called transcribe_audio at least once
    summarizer_model: str = "gemini-3.1-flash-lite-preview"  # Model for the memory orchestrator
    summarizer_max_tokens: int = 800  # Max output tokens for summarizer
    summarizer_timeout: int = 30  # Seconds before summarizer fails fast
    file_backed: bool = False  # Use file-backed memory with temporal cone
    orchestrated: bool = False  # Use orchestrated memory (LLM-managed working memory)
    temporal_radius: float = 120.0  # L1 temporal entries loaded within ±N seconds of focal point
    frame_radius: float = 60.0  # L2 frame entries loaded within ±N seconds of focal point
    max_narrative_entries: int = 40  # Max entries in narrative thread
    max_pinned_entries: int = 20  # Max entries promoted to L0_PINNED
    manifest_max_detailed: int = 30  # Detailed manifest entries before compression
    manifest_compression_batch: int = 10  # Entries per compression batch
    orchestrator_max_iterations: int = 0  # 0 = single-pass (legacy), >0 = agentic loop with tools
    fs_context_budget: int = 16000  # Max chars for filesystem context in orchestrator prompt
    max_key_frames: int = 6  # Max frames to pin as visual memory for the main agent
    summarizer_reasoning_effort: Optional[str] = None  # "low"/"medium"/"high" for the orchestrator model


@dataclass
class AgentPricing:
    input_per_million: float = 0.0
    output_per_million: float = 0.0

@dataclass
class PricingConfig:
    input_per_million: float = 0.0   # Default / main_agent rate
    output_per_million: float = 0.0
    currency: str = "USD"
    vlm: AgentPricing = field(default_factory=AgentPricing)
    summarizer: AgentPricing = field(default_factory=AgentPricing)

@dataclass
class VisualPlannerConfig:
    """Configuration for the visual planner todo list."""
    enabled: bool = False
    descriptions_dir: str = "dataset/videomme_long/visual_descriptions"
    narrative_dir: str = ""  # Dir of per-video neutral descriptions to seed as a narrative timeline ("" = off)
    model: Optional[str] = None  # None = use main_agent.model; set to a non-thinking model to avoid ~64K thinking token blowup
    max_tokens: int = 2048  # Output budget for planner/verifier calls


@dataclass
class AgentConfig:
    main_agent: MainAgentConfig = field(default_factory=MainAgentConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    visual_planner: VisualPlannerConfig = field(default_factory=VisualPlannerConfig)
    include_tools: List[str] = field(default_factory=list)  # Whitelist of tool names; empty = all tools
    models: Dict[str, ModelLimits] = field(default_factory=lambda: {
        "default": ModelLimits()
    })

    def get_model_limits(self, model: str) -> ModelLimits:
        """Look up token limits for a model.

        Strips provider prefixes (openai/, anthropic/, etc.) and
        falls back to the "default" entry if the model isn't listed.
        """
        from .llm_api import strip_provider_prefix
        bare = strip_provider_prefix(model)

        if bare in self.models:
            return self.models[bare]
        return self.models.get("default", ModelLimits())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        # Filter out deprecated fields
        main_agent_data = data.get("main_agent", {})
        main_agent_data.pop("max_tokens", None)  # Now in models section
        main_agent = MainAgentConfig(**main_agent_data)

        sandbox = SandboxConfig(**data.get("sandbox", {}))
        memory = MemoryConfig(**data.get("memory", {}))
        pricing_data = data.get("pricing", {})
        vlm_pricing = AgentPricing(**pricing_data.pop("vlm", {}))
        summarizer_pricing = AgentPricing(**pricing_data.pop("summarizer", {}))
        pricing = PricingConfig(**pricing_data, vlm=vlm_pricing, summarizer=summarizer_pricing)
        visual_planner = VisualPlannerConfig(**data.get("visual_planner", {}))
        include_tools = data.get("include_tools", [])

        # Parse models section
        models_data = data.get("models", {})
        models = {}
        for name, limits in models_data.items():
            models[name] = ModelLimits(**limits)
        if "default" not in models:
            models["default"] = ModelLimits()

        return cls(
            main_agent=main_agent,
            sandbox=sandbox,
            memory=memory,
            pricing=pricing,
            visual_planner=visual_planner,
            include_tools=include_tools,
            models=models,
        )


def find_config_file() -> Optional[Path]:
    """Search for the active config YAML.

    Honors the ``CONFIG_FILE`` env var (filename or absolute path) — same
    pattern as ``PROMPTS_FILE``. Falls back to ``config.yaml`` in the
    standard locations.
    """
    import os
    override = os.environ.get("CONFIG_FILE")
    if override:
        cand = Path(override)
        if cand.is_absolute() and cand.exists():
            return cand
        # bare filename: look in the standard configs/ dir
        bare = Path(__file__).resolve().parent.parent / "configs" / override
        if bare.exists():
            return bare
    filename = "config.yaml"
    candidates = [
        Path.cwd() / filename,
        Path.cwd() / "configs" / filename,
        Path(__file__).resolve().parent.parent / filename,
        Path(__file__).resolve().parent.parent / "configs" / filename,
        Path.home() / ".config" / "video-agent" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(path: Optional[Path] = None) -> AgentConfig:
    """Load configuration from YAML file or return defaults."""
    if path is None:
        path = find_config_file()

    if path is None or not path.exists():
        return AgentConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    return AgentConfig.from_dict(data)


# Singleton cached config (lazy-loaded)
_cached_config: Optional[AgentConfig] = None


def get_config(reload: bool = False) -> AgentConfig:
    """Get the global config, loading from file if needed."""
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = load_config()
    return _cached_config
