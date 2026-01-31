"""Agent traits for composable capabilities."""

from llm_agent.core.traits.base import BaseTrait, Trait
from llm_agent.core.traits.directive import Directive, DirectiveTrait
from llm_agent.core.traits.http import HTTPConfig, HTTPTrait
from llm_agent.core.traits.learn import LearnConfig, LearnTrait
from llm_agent.core.traits.llm import LLMConfig, LLMTrait
from llm_agent.core.traits.tools import ToolsTrait


__all__ = [
    "BaseTrait",
    "Directive",
    "DirectiveTrait",
    "HTTPConfig",
    "HTTPTrait",
    "LearnConfig",
    "LearnTrait",
    "LLMConfig",
    "LLMTrait",
    "ToolsTrait",
    "Trait",
]
