"""Agent framework with learning capabilities."""

from llm_learn.collection import ScoredFact

from llm_agent.agent import Agent
from llm_agent.config import AgentConfig
from llm_agent.llm import CompletionResult, HTTPBackend, LLMBackend, LLMError, Message
from llm_agent.traits import Directive, DirectiveTrait, Trait


__version__ = "0.0.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "CompletionResult",
    "Directive",
    "DirectiveTrait",
    "HTTPBackend",
    "LLMBackend",
    "LLMError",
    "Message",
    "ScoredFact",
    "Trait",
]
