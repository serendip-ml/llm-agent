"""Tests for Agent State Machine."""

import pytest

from llm_gent.runtime import AgentState, InvalidTransitionError, can_transition, transition


pytestmark = pytest.mark.unit


class TestAgentState:
    """Tests for AgentState enum."""

    def test_all_states_exist(self):
        """All expected states exist."""
        assert AgentState.IDLE
        assert AgentState.STARTING
        assert AgentState.RUNNING
        assert AgentState.STOPPING
        assert AgentState.STOPPED
        assert AgentState.ERROR


class TestTransition:
    """Tests for transition function."""

    def test_idle_to_starting(self):
        """IDLE -> STARTING is valid."""
        result = transition(AgentState.IDLE, AgentState.STARTING)
        assert result == AgentState.STARTING

    def test_starting_to_running(self):
        """STARTING -> RUNNING is valid."""
        result = transition(AgentState.STARTING, AgentState.RUNNING)
        assert result == AgentState.RUNNING

    def test_starting_to_error(self):
        """STARTING -> ERROR is valid."""
        result = transition(AgentState.STARTING, AgentState.ERROR)
        assert result == AgentState.ERROR

    def test_running_to_stopping(self):
        """RUNNING -> STOPPING is valid."""
        result = transition(AgentState.RUNNING, AgentState.STOPPING)
        assert result == AgentState.STOPPING

    def test_running_to_error(self):
        """RUNNING -> ERROR is valid."""
        result = transition(AgentState.RUNNING, AgentState.ERROR)
        assert result == AgentState.ERROR

    def test_stopping_to_stopped(self):
        """STOPPING -> STOPPED is valid."""
        result = transition(AgentState.STOPPING, AgentState.STOPPED)
        assert result == AgentState.STOPPED

    def test_stopping_to_error(self):
        """STOPPING -> ERROR is valid."""
        result = transition(AgentState.STOPPING, AgentState.ERROR)
        assert result == AgentState.ERROR

    def test_stopped_to_starting(self):
        """STOPPED -> STARTING is valid (restart)."""
        result = transition(AgentState.STOPPED, AgentState.STARTING)
        assert result == AgentState.STARTING

    def test_error_to_starting(self):
        """ERROR -> STARTING is valid (recovery)."""
        result = transition(AgentState.ERROR, AgentState.STARTING)
        assert result == AgentState.STARTING

    def test_invalid_idle_to_running(self):
        """IDLE -> RUNNING is invalid (must go through STARTING)."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            transition(AgentState.IDLE, AgentState.RUNNING)

        assert exc_info.value.current == AgentState.IDLE
        assert exc_info.value.target == AgentState.RUNNING
        assert "IDLE -> RUNNING" in str(exc_info.value)

    def test_invalid_running_to_idle(self):
        """RUNNING -> IDLE is invalid."""
        with pytest.raises(InvalidTransitionError):
            transition(AgentState.RUNNING, AgentState.IDLE)

    def test_invalid_stopped_to_running(self):
        """STOPPED -> RUNNING is invalid (must go through STARTING)."""
        with pytest.raises(InvalidTransitionError):
            transition(AgentState.STOPPED, AgentState.RUNNING)


class TestCanTransition:
    """Tests for can_transition function."""

    def test_valid_transition_returns_true(self):
        """can_transition returns True for valid transitions."""
        assert can_transition(AgentState.IDLE, AgentState.STARTING) is True
        assert can_transition(AgentState.STARTING, AgentState.RUNNING) is True
        assert can_transition(AgentState.RUNNING, AgentState.STOPPING) is True

    def test_invalid_transition_returns_false(self):
        """can_transition returns False for invalid transitions."""
        assert can_transition(AgentState.IDLE, AgentState.RUNNING) is False
        assert can_transition(AgentState.RUNNING, AgentState.IDLE) is False
        assert can_transition(AgentState.STOPPED, AgentState.RUNNING) is False

    def test_self_transition_returns_false(self):
        """Self-transitions are invalid."""
        assert can_transition(AgentState.RUNNING, AgentState.RUNNING) is False


class TestInvalidTransitionError:
    """Tests for InvalidTransitionError."""

    def test_exception_attributes(self):
        """Exception stores current and target states."""
        error = InvalidTransitionError(AgentState.IDLE, AgentState.RUNNING)

        assert error.current == AgentState.IDLE
        assert error.target == AgentState.RUNNING

    def test_exception_message(self):
        """Exception has descriptive message."""
        error = InvalidTransitionError(AgentState.IDLE, AgentState.RUNNING)

        assert "IDLE" in str(error)
        assert "RUNNING" in str(error)
        assert "->" in str(error)
