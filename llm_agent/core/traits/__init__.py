"""Agent traits for composable capabilities."""

from enum import StrEnum

from llm_agent.core.traits.base import BaseTrait, Trait
from llm_agent.core.traits.directive import Directive, DirectiveTrait, MethodTrait
from llm_agent.core.traits.factory import Factory
from llm_agent.core.traits.http import HTTPConfig, HTTPTrait
from llm_agent.core.traits.learn import LearnConfig, LearnTrait
from llm_agent.core.traits.llm import LLMConfig, LLMTrait
from llm_agent.core.traits.registry import Registry
from llm_agent.core.traits.saia import SAIAConfig, SAIATrait
from llm_agent.core.traits.tools import ToolsTrait


class TraitName(StrEnum):
    """Trait name identifiers.

    Using str Enum so values work in YAML configs and as strings.
    Agents can declare required/optional traits using these enum values.
    """

    DIRECTIVE = "directive"
    LLM = "llm"
    LEARN = "learn"
    HTTP = "http"
    SAIA = "saia"
    TOOLS = "tools"
    METHOD = "method"


# All trait types available in the platform
ALL_TRAITS: list[TraitName] = [
    TraitName.DIRECTIVE,
    TraitName.LLM,
    TraitName.LEARN,
    TraitName.HTTP,
    TraitName.SAIA,
    TraitName.TOOLS,
    TraitName.METHOD,
]


__all__ = [
    # Base
    "BaseTrait",
    "Trait",
    # Names & Catalogs
    "TraitName",
    "ALL_TRAITS",
    # Factory & Registry
    "Factory",
    "Registry",
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
