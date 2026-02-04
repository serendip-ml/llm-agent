"""Default agent - config-driven, no learning."""

from llm_agent.agents.default.agent import Agent
from llm_agent.agents.default.factory import Factory


__all__ = [
    "Agent",
    "Factory",
]
