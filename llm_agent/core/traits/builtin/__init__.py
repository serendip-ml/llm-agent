"""Built-in traits for composable agent capabilities."""

from llm_agent.core.traits.builtin.directive import Directive, DirectiveTrait, MethodTrait
from llm_agent.core.traits.builtin.http import HTTPConfig, HTTPTrait
from llm_agent.core.traits.builtin.learn import LearnConfig, LearnTrait
from llm_agent.core.traits.builtin.llm import LLMConfig, LLMTrait
from llm_agent.core.traits.builtin.saia import SAIAConfig, SAIATrait
from llm_agent.core.traits.builtin.tools import ToolsTrait


__all__ = [
    # Directive/Method
    "Directive",
    "DirectiveTrait",
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
    # SAIA
    "SAIAConfig",
    "SAIATrait",
    # Tools
    "ToolsTrait",
]
