"""Factory for creating JokeTellerAgent instances."""

from __future__ import annotations

from ...core.agent.prog import ProgAgentFactory
from ...core.traits import TraitName as TN
from .agent import JokeTellerAgent


class Factory(ProgAgentFactory):
    """Factory for JokeTellerAgent.

    Declares LLM and LEARN as required traits. These can be overridden in YAML:
        traits:
          required: [llm, learn]

    Or removed entirely by setting required: [] in YAML to handle validation in code.
    """

    agent_class = JokeTellerAgent
    required_traits = [TN.LLM, TN.LEARN]
