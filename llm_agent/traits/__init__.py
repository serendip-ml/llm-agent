"""Agent traits for composable capabilities."""

from llm_agent.traits.base import BaseTrait, Trait
from llm_agent.traits.directive import Directive, DirectiveTrait
from llm_agent.traits.http import HTTPConfig, HTTPTrait
from llm_agent.traits.learn import LearnConfig, LearnTrait
from llm_agent.traits.llm import LLMConfig, LLMTrait


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
    "Trait",
]
