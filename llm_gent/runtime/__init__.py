"""Runtime infrastructure for operating agents.

This package provides the operational infrastructure for running agents:
- Core: Orchestrates agent subprocess lifecycle
- AgentRegistry: Manages agent configurations
- AgentRunner: Runs agents in subprocesses
- AgentHandle/AgentInfo: Agent state tracking
- AgentState: Lifecycle state machine
"""

from .core import Core
from .handle import AgentHandle, AgentInfo
from .registry import AgentRegistry
from .runner import AgentRunner
from .state import AgentState, InvalidTransitionError, can_transition, transition


__all__ = [
    "AgentHandle",
    "AgentInfo",
    "AgentRegistry",
    "AgentRunner",
    "AgentState",
    "Core",
    "InvalidTransitionError",
    "can_transition",
    "transition",
]
