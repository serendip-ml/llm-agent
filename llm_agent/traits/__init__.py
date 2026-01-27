"""Agent traits for composable capabilities."""

from llm_agent.traits.base import BaseTrait, Trait
from llm_agent.traits.directive import Directive, DirectiveTrait
from llm_agent.traits.http import HTTPConfig, HTTPTrait


__all__ = [
    "BaseTrait",
    "Directive",
    "DirectiveTrait",
    "HTTPConfig",
    "HTTPTrait",
    "Trait",
]
