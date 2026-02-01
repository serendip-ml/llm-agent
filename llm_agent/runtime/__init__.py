"""Runtime infrastructure for operating agents.

This package provides the operational infrastructure for running agents:
- Core: Orchestrates agent subprocess lifecycle
- AgentRegistry: Manages agent configurations
- AgentRunner: Runs agents in subprocesses
- AgentHandle/AgentInfo: Agent state tracking
- AgentState: Lifecycle state machine
"""

from llm_agent.runtime.core import Core
from llm_agent.runtime.handle import AgentHandle, AgentInfo
from llm_agent.runtime.registry import AgentRegistry
from llm_agent.runtime.runner import AgentRunner
from llm_agent.runtime.state import AgentState, InvalidTransitionError, can_transition, transition


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
