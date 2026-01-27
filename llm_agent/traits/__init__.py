"""Agent traits for composable capabilities."""

from llm_agent.traits.base import Trait
from llm_agent.traits.directive import Directive, DirectiveTrait
from llm_agent.traits.http import HTTPConfig, HTTPTrait


__all__ = [
    "Directive",
    "DirectiveTrait",
    "HTTPConfig",
    "HTTPTrait",
    "Trait",
]
