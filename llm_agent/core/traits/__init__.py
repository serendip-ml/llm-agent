"""Agent traits for composable capabilities."""

from enum import Enum

from llm_agent.core.traits.base import BaseTrait, Trait
from llm_agent.core.traits.factory import TraitFactory
from llm_agent.core.traits.http import HTTPConfig, HTTPTrait
from llm_agent.core.traits.identity import Directive, DirectiveTrait, MethodTrait
from llm_agent.core.traits.learn import LearnConfig, LearnTrait
from llm_agent.core.traits.llm import LLMConfig, LLMTrait
from llm_agent.core.traits.registry import TraitRegistry
from llm_agent.core.traits.saia import SAIAConfig, SAIATrait
from llm_agent.core.traits.tools import ToolsTrait


class TraitName(str, Enum):
    """Trait name identifiers.

    Using str Enum so values work in YAML configs and as strings.
    Agents can declare required/optional traits using these enum values.
    """

    LLM = "llm"
    LEARN = "learn"
    HTTP = "http"
    SAIA = "saia"
    TOOLS = "tools"
    IDENTITY = "identity"
    METHOD = "method"


__all__ = [
    # Base
    "BaseTrait",
    "Trait",
    # Names
    "TraitName",
    # Factory
    "TraitFactory",
    # Registry
    "TraitRegistry",
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
