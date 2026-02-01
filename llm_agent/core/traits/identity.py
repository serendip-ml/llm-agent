"""Identity and method traits for agents.

Identity: "Who I am" - the agent's core identity and purpose.
Method: "How I operate" - the agent's operational approach, which can evolve.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm_agent.core.traits.base import BaseTrait


class Identity(BaseModel):
    """Agent's identity - who it is and what it's trying to be good at.

    An identity defines the agent's core purpose and persona, injected into
    every prompt. The prompt is the source of truth for agent behavior.

    Future: structured constraints and success signals will be added
    for enforcement and measurement. Design TBD based on concrete use cases.

    Example YAML config:
        identity: |
            You are a code reviewer. Your goal is to catch all issues in one pass.

            Be critical and evidence-based. Verify before flagging.
            No false positives. No nitpicks that waste developer time.

    Or with the full object syntax:
        identity:
          prompt: |
            You are a code reviewer...
    """

    prompt: str
    """The identity prompt - injected into system prompt."""

    # Placeholder for future structured behavior (constraints, signals, etc.)
    # Design TBD based on concrete use cases like code review agent.
    extensions: dict[str, Any] = Field(default_factory=dict)
    """Reserved for future structured constraints/signals."""


class IdentityTrait(BaseTrait):
    """Trait for agents that have an identity.

    Holds the agent's identity and injects it into prompts.
    Identity is immutable - it defines who the agent is.

    Usage:
        identity = Identity(
            prompt="You are a code reviewer. Be critical and evidence-based."
        )
        trait = IdentityTrait(identity)
        agent.add_trait(trait)

    Or with a simple string:
        trait = IdentityTrait("You are a code reviewer.")
        agent.add_trait(trait)
    """

    def __init__(self, identity: Identity | str) -> None:
        """Initialize identity trait.

        Args:
            identity: The agent's identity (Identity object or prompt string).
        """
        super().__init__()
        if isinstance(identity, str):
            identity = Identity(prompt=identity)
        self._identity = identity

    @property
    def identity(self) -> Identity:
        """The agent's identity."""
        return self._identity

    def build_prompt(self, base_prompt: str) -> str:
        """Build system prompt with identity injected.

        Args:
            base_prompt: The base system prompt.

        Returns:
            System prompt with identity prepended.
        """
        return f"{self._identity.prompt}\n\n{base_prompt}"


class MethodTrait(BaseTrait):
    """Trait for agents that have an operational method.

    Holds the agent's method - how it operates. Unlike identity, method
    can evolve over time based on learning and feedback.

    Usage:
        trait = MethodTrait(
            "- Read the full diff before commenting\\n"
            "- Prioritize security over style\\n"
            "- Be concise but thorough"
        )
        agent.add_trait(trait)

        # Later, update based on learning
        trait.update("- Read the full diff before commenting\\n"
                     "- Prioritize security over style\\n"
                     "- Be concise but thorough\\n"
                     "- Always check for SQL injection")
    """

    def __init__(self, method: str) -> None:
        """Initialize method trait.

        Args:
            method: The agent's operational method.
        """
        super().__init__()
        self._method = method

    @property
    def method(self) -> str:
        """The agent's current method."""
        return self._method

    def build_prompt(self, base_prompt: str) -> str:
        """Build system prompt with method injected.

        Args:
            base_prompt: The base system prompt.

        Returns:
            System prompt with method appended.
        """
        return f"{base_prompt}\n\n## Method\n{self._method}"

    def update(self, new_method: str) -> None:
        """Update method based on learning.

        Args:
            new_method: The new operational method.
        """
        self._method = new_method
