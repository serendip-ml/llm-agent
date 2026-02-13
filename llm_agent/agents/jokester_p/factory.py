"""Factory for creating jokester agent instances."""

from __future__ import annotations

from ...core.agent import Factory as BaseFactory
from ...core.traits import TraitName as TN
from .agent import JokesterAgent


class Factory(BaseFactory):
    """Factory for jokester agent.

    Declares DIRECTIVE, LLM, LEARN, STORAGE, and RATING as required traits.
    These can be overridden in YAML:
        traits:
          required: [directive, llm, learn, storage, rating]

    StorageTrait provides agent-specific tables for tracking model usage and training metadata.
    RatingTrait enables inline rating of generated jokes if rating.auto: true.
    No tools needed - agent uses traits directly in code.
    """

    agent_class = JokesterAgent
    # IMPORTANT: STORAGE and RATING must come after LEARN - they depend on LearnTrait's database
    required_traits = [TN.DIRECTIVE, TN.LLM, TN.LEARN, TN.STORAGE, TN.RATING]
    default_tools = {}  # This agent doesn't need tools (uses traits directly in code)
