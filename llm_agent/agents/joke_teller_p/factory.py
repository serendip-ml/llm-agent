"""Factory for creating JokeTellerAgent instances."""

from __future__ import annotations

from ...core.agent import Factory as BaseFactory
from ...core.traits import TraitName as TN
from .agent import JokeTellerAgent


class Factory(BaseFactory):
    """Factory for JokeTellerAgent.

    Declares LLM and LEARN as required traits. These can be overridden in YAML:
        traits:
          required: [llm, learn]

    No tools needed - agent uses LLMTrait/LearnTrait directly in code.
    """

    agent_class = JokeTellerAgent
    required_traits = [TN.DIRECTIVE, TN.LLM, TN.LEARN]
    default_tools = {}  # This agent doesn't need tools (uses traits directly in code)
