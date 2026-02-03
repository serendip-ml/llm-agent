"""Typed configuration for agent server.

Parses raw YAML config into typed Pydantic models.
"""

from __future__ import annotations

from typing import Any, Literal

from appinfra.app.fastapi.config import ApiConfig
from pydantic import BaseModel, Field

from llm_agent.core.traits.identity import Identity


class LearnBackendConfig(BaseModel):
    """Learn trait backend configuration."""

    profile_id: str
    db: dict[str, Any]
    """Database configuration dict (url, extensions, etc.)."""
    embedder_url: str | None = None
    embedder_model: str = "default"


class TaskConfigYAML(BaseModel):
    """Task configuration from YAML."""

    description: str
    output_schema: dict[str, Any] | None = None
    max_iterations: int = 10
    """Maximum LLM round-trips before stopping. 0 means no limit."""
    timeout_secs: float = 0
    """Maximum seconds for task execution. 0 means no limit."""


class ScheduleConfigYAML(BaseModel):
    """Schedule configuration from YAML."""

    interval: int
    """Seconds between automatic executions."""


class AgentConfigYAML(BaseModel):
    """Configuration for a single agent from YAML.

    Example:
        codebase-explorer:
          class: prompt
          identity: |
            You are a codebase exploration agent.
            You value curiosity over speed, insight over coverage.
          method: |
            - Start with high-level structure
            - Read important files first
            - Follow interesting patterns deeper
          task:
            description: Explore the codebase
          tools:
            shell: {allowed_commands: [grep, find]}
          schedule:
            interval: 600
    """

    class_: Literal["prompt", "programmatic"] = Field(alias="class", default="prompt")
    """Agent class: 'prompt' for YAML-only, 'programmatic' for custom Python."""

    identity: Identity | str | None = None
    """Agent's identity - who it is (string or Identity object)."""

    method: str | None = None
    """Agent's operational method - how it works."""

    task: TaskConfigYAML
    """Primary task configuration."""

    tools: dict[str, dict[str, Any]] = {}
    """Tool configurations keyed by tool type."""

    schedule: ScheduleConfigYAML | None = None
    """Optional automatic execution schedule."""

    conversation: dict[str, Any] = {}
    """Conversation settings (max_tokens, compact_threshold, etc.)."""

    model_config = {"populate_by_name": True}


class AgentServerConfig(BaseModel):
    """Complete server configuration.

    Loaded from etc/llm-agent.yaml:

        server:
          host: 0.0.0.0
          port: 8080
          uvicorn: !include './uvicorn.yaml'

        llm: !include './llm.yaml'

        learn:
          profile_id: "1"
          db: !include './infra.yaml#dbs.main'

        agents:
          codebase-explorer:
            class: prompt
            ...
    """

    model_config = {"arbitrary_types_allowed": True}

    server: ApiConfig
    """HTTP server settings (appinfra ApiConfig)."""

    llm: dict[str, Any]
    """LLM backend configuration (see llm-infer LLMClient.from_config)."""

    learn: LearnBackendConfig | None = None
    """Optional learn backend for memory/feedback."""

    agents: dict[str, AgentConfigYAML] = {}
    """Agent definitions keyed by name."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentServerConfig:
        """Create config from raw YAML dict.

        Args:
            raw: Raw configuration dictionary from YAML.

        Returns:
            Validated AgentServerConfig.

        Raises:
            ValidationError: If config is invalid.
        """
        from appinfra.app.fastapi.config import UvicornConfig

        # Parse server config into ApiConfig
        server_raw = raw.get("server", {})
        uvicorn_raw = server_raw.get("uvicorn", {})

        server = ApiConfig(
            host=server_raw.get("host", "0.0.0.0"),
            port=server_raw.get("port", 8080),
            response_timeout=server_raw.get("response_timeout", 60.0),
            uvicorn=UvicornConfig(**uvicorn_raw) if uvicorn_raw else UvicornConfig(),
        )

        return cls(
            server=server,
            llm=raw.get("llm", {}),
            learn=raw.get("learn"),
            agents=raw.get("agents", {}),
        )
