"""Agent state machine with explicit states and validated transitions.

The state machine ensures agents follow a well-defined lifecycle:
IDLE -> STARTING -> RUNNING -> STOPPING -> STOPPED

Error can occur from STARTING, RUNNING, or STOPPING.
Recovery is possible from ERROR or STOPPED back to STARTING.
"""

from __future__ import annotations

from enum import Enum, auto


class AgentState(Enum):
    """Agent lifecycle states.

    States:
        IDLE: Registered but process not spawned.
        STARTING: Process spawning.
        RUNNING: Process alive, agent active.
        STOPPING: Shutdown in progress.
        STOPPED: Process terminated cleanly.
        ERROR: Process crashed or error occurred.
    """

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


# Valid state transitions
TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.STARTING},
    AgentState.STARTING: {AgentState.RUNNING, AgentState.ERROR},
    AgentState.RUNNING: {AgentState.STOPPING, AgentState.ERROR},
    AgentState.STOPPING: {AgentState.STOPPED, AgentState.ERROR},
    AgentState.STOPPED: {AgentState.STARTING},
    AgentState.ERROR: {AgentState.STARTING},
}


class InvalidTransitionError(ValueError):
    """Raised when attempting an invalid state transition."""

    def __init__(self, current: AgentState, target: AgentState) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Invalid transition: {current.name} -> {target.name}")


def transition(current: AgentState, target: AgentState) -> AgentState:
    """Validate and perform state transition.

    Args:
        current: Current state.
        target: Desired target state.

    Returns:
        The target state if transition is valid.

    Raises:
        InvalidTransitionError: If the transition is not allowed.
    """
    valid_targets = TRANSITIONS.get(current, set())
    if target not in valid_targets:
        raise InvalidTransitionError(current, target)
    return target


def can_transition(current: AgentState, target: AgentState) -> bool:
    """Check if a transition is valid without raising.

    Args:
        current: Current state.
        target: Desired target state.

    Returns:
        True if transition is valid, False otherwise.
    """
    return target in TRANSITIONS.get(current, set())
