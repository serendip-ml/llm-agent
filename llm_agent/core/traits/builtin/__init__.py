"""Built-in traits for composable agent capabilities."""

from .directive import Directive, DirectiveTrait, MethodTrait
from .http import HTTPConfig, HTTPTrait
from .learn import LearnConfig, LearnTrait
from .llm import LLMConfig, LLMTrait
from .saia import SAIAConfig, SAIATrait
from .tools import ToolsTrait


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
