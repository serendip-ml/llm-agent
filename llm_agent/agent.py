"""Core agent implementation."""

from __future__ import annotations

from llm_agent.config import AgentConfig
from llm_agent.llm import CompletionResult, LLMBackend, Message


class Agent:
    """Learning agent that improves through feedback.

    Phase 1 implementation: basic completion only.
    Future phases add: memory (facts), feedback collection, RAG, adapters.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMBackend,
    ) -> None:
        """Initialize agent.

        Args:
            config: Agent configuration.
            llm: LLM backend for completions.
        """
        self._config = config
        self._llm = llm

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    def complete(
        self,
        query: str,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a response.

        Args:
            query: User input.
            system_prompt: System prompt (uses default if None).

        Returns:
            Completion result with response and metadata.
        """
        prompt = system_prompt or self._config.default_prompt

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=query),
        ]

        return self._llm.complete(
            messages=messages,
            model=self._config.model,
        )
