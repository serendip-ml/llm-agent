"""Directive trait for agents with a purpose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from llm_agent.agent import Agent


class Directive(BaseModel):
    """What the agent is trying to be good at.

    A directive defines the agent's purpose - injected into every prompt.
    The prompt is the source of truth for agent behavior.

    Future: structured constraints and success signals will be added
    for enforcement and measurement. Design TBD based on concrete use cases.

    Example YAML config:
        directive:
          prompt: |
            You are a code reviewer. Your goal is to catch all issues in one pass.

            Be critical and evidence-based. Verify before flagging.
            No false positives. No nitpicks that waste developer time.
    """

    prompt: str
    """The directive prompt - injected into system prompt."""

    # Placeholder for future structured behavior (constraints, signals, etc.)
    # Design TBD based on concrete use cases like code review agent.
    extensions: dict[str, Any] = {}
    """Reserved for future structured constraints/signals."""


class DirectiveTrait:
    """Trait for agents that have a directive.

    Holds the agent's directive and injects it into prompts.
    Feedback collection and improvement tracking are handled by llm-learn.

    Usage:
        directive = Directive(
            goal="Review code thoroughly",
            persona="Critical, evidence-based",
            constraints=["No false positives"],
        )
        trait = DirectiveTrait(directive)
        agent.add_trait(trait)
    """

    def __init__(self, directive: Directive) -> None:
        """Initialize directive trait.

        Args:
            directive: The agent's directive.
        """
        self._directive = directive
        self._agent: Agent | None = None

    @property
    def directive(self) -> Directive:
        """The agent's directive."""
        return self._directive

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent.

        Args:
            agent: The agent this trait is attached to.
        """
        self._agent = agent

    def on_start(self) -> None:
        """Called when agent starts."""
        pass

    def on_stop(self) -> None:
        """Called when agent stops."""
        pass

    def build_prompt(self, base_prompt: str) -> str:
        """Build system prompt with directive injected.

        Args:
            base_prompt: The base system prompt.

        Returns:
            System prompt with directive prepended.
        """
        return f"{self._directive.prompt}\n\n{base_prompt}"
