"""Agent traits for composable capabilities."""

from llm_agent.core.traits.base import BaseTrait, Trait
from llm_agent.core.traits.http import HTTPConfig, HTTPTrait
from llm_agent.core.traits.identity import Identity, IdentityTrait, MethodTrait
from llm_agent.core.traits.learn import LearnConfig, LearnTrait
from llm_agent.core.traits.llm import LLMConfig, LLMTrait
from llm_agent.core.traits.tools import ToolsTrait


__all__ = [
    # Base
    "BaseTrait",
    "Trait",
    # Identity/Method
    "Identity",
    "IdentityTrait",
    "MethodTrait",
    # HTTP
    "HTTPConfig",
    "HTTPTrait",
    # Learn
    "LearnConfig",
    "LearnTrait",
    # LLM
    "LLMConfig",
    "LLMTrait",
    # Tools
    "ToolsTrait",
]
