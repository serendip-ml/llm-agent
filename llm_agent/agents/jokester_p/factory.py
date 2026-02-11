"""Factory for creating JokesterAgent instances."""

from __future__ import annotations

from ...core.agent import Factory as BaseFactory
from ...core.traits import TraitName as TN
from .agent import JokesterAgent


class Factory(BaseFactory):
    """Factory for JokesterAgent.

    Declares DIRECTIVE, LLM, and LEARN as required traits. These can be overridden in YAML:
        traits:
          required: [directive, llm, learn]

    No tools needed - agent uses LLMTrait/LearnTrait directly in code.
    """

    agent_class = JokesterAgent
    required_traits = [TN.DIRECTIVE, TN.LLM, TN.LEARN]
    default_tools = {}  # This agent doesn't need tools (uses traits directly in code)
