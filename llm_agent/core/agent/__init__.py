"""Agent package - core agent abstractions and implementations."""

from .agent import Agent
from .config import Config
from .factory import Factory, _substitute_in_dict, _substitute_variables
from .identity import Identity
from .types import ExecutionResult


__all__ = [
    "Agent",
    "Config",
    "ExecutionResult",
    "Factory",
    "Identity",
    "_substitute_in_dict",
    "_substitute_variables",
]
