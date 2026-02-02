"""Tests for GovernorPolicy."""

import pytest

from llm_agent.core.governor.policy import DefaultGovernorPolicy
from llm_agent.core.governor.types import (
    Decision,
    GovernorContext,
    InterpretedResponse,
    ResponseEvent,
)
from llm_agent.core.llm.types import CompletionResult
from llm_agent.core.task import Task
from llm_agent.core.tools.base import ToolCall


pytestmark = pytest.mark.unit


def make_context(
    iteration: int = 0,
    elapsed_secs: float = 0.0,
    tokens_used: int = 0,
    has_work: bool = False,
    max_iterations: int = 10,
    timeout_secs: float = 0,
    wrap_up_injected: bool = False,
) -> GovernorContext:
    """Helper to create GovernorContext."""
    task = Task(
        name="test",
        description="test task",
        max_iterations=max_iterations,
        timeout_secs=timeout_secs,
    )
    tool_calls = (
        (
            # Create mock tool call result if has_work
        )
        if not has_work
        else ()
    )

    return GovernorContext(
        task=task,
        iteration=iteration,
        elapsed_secs=elapsed_secs,
        tokens_used=tokens_used,
        tool_calls_so_far=tool_calls,
        wrap_up_injected=wrap_up_injected,
    )


def make_interpreted(
    event: ResponseEvent,
    tool_calls: tuple = (),
    terminal_call: ToolCall | None = None,
) -> InterpretedResponse:
    """Helper to create InterpretedResponse."""
    raw = CompletionResult(
        id="test",
        content="test",
        model="test",
        tokens_used=10,
        latency_ms=100,
    )
    return InterpretedResponse(
        event=event,
        raw=raw,
        tool_calls=tool_calls,
        terminal_call=terminal_call,
    )


class TestDefaultGovernorPolicy:
    """Tests for DefaultGovernorPolicy decisions."""

    @pytest.fixture
    def policy(self):
        """Default policy instance."""
        return DefaultGovernorPolicy()

    def test_text_only_confirm_completion(self, policy):
        """TEXT_ONLY triggers CONFIRM_COMPLETION."""
        response = make_interpreted(ResponseEvent.TEXT_ONLY)
        context = make_context()

        decision = policy.decide(response, context)

        assert decision.decision == Decision.CONFIRM_COMPLETION
        assert decision.confirm_prompt is not None
        assert "call the completion tool" in decision.confirm_prompt

    def test_work_tools_execute(self, policy):
        """WORK_TOOLS triggers EXECUTE_TOOLS."""
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        context = make_context()

        decision = policy.decide(response, context)

        assert decision.decision == Decision.EXECUTE_TOOLS

    def test_terminal_after_work_accept(self, policy):
        """TERMINAL_AFTER_WORK triggers ACCEPT_TERMINAL."""
        response = make_interpreted(ResponseEvent.TERMINAL_AFTER_WORK)
        context = make_context()

        decision = policy.decide(response, context)

        assert decision.decision == Decision.ACCEPT_TERMINAL

    def test_early_terminal_confirm(self, policy):
        """EARLY_TERMINAL triggers CONFIRM_EARLY."""
        response = make_interpreted(ResponseEvent.EARLY_TERMINAL)
        context = make_context()

        decision = policy.decide(response, context)

        assert decision.decision == Decision.CONFIRM_EARLY
        assert decision.confirm_prompt is not None
        assert "without having used any tools" in decision.confirm_prompt

    def test_mixed_batch_execute(self, policy):
        """MIXED_BATCH triggers EXECUTE_TOOLS."""
        response = make_interpreted(ResponseEvent.MIXED_BATCH)
        context = make_context()

        decision = policy.decide(response, context)

        assert decision.decision == Decision.EXECUTE_TOOLS


class TestGracefulTimeout:
    """Tests for graceful timeout warnings."""

    def test_time_approaching_injects_warning(self):
        """Warning injected when time is running out."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_secs=30.0)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        # 25 seconds remaining, threshold is 30
        context = make_context(timeout_secs=100, elapsed_secs=75)

        decision = policy.decide(response, context)

        assert decision.decision == Decision.INJECT_AND_CONTINUE
        assert "approaching the time limit" in decision.inject_message

    def test_time_not_approaching_executes_normally(self):
        """Normal execution when time is not running out."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_secs=30.0)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        # 50 seconds remaining, threshold is 30
        context = make_context(timeout_secs=100, elapsed_secs=50)

        decision = policy.decide(response, context)

        assert decision.decision == Decision.EXECUTE_TOOLS

    def test_no_timeout_no_warning(self):
        """No warning when timeout_secs is 0 (unlimited)."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_secs=30.0)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        context = make_context(timeout_secs=0, elapsed_secs=100)

        decision = policy.decide(response, context)

        assert decision.decision == Decision.EXECUTE_TOOLS

    def test_iterations_approaching_injects_warning(self):
        """Warning injected when iterations running out."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_iterations=2)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        # 1 iteration remaining, threshold is 2
        context = make_context(max_iterations=10, iteration=9)

        decision = policy.decide(response, context)

        assert decision.decision == Decision.INJECT_AND_CONTINUE
        assert "iteration limit" in decision.inject_message

    def test_iterations_not_approaching_executes_normally(self):
        """Normal execution when iterations not running out."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_iterations=2)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        # 5 iterations remaining, threshold is 2
        context = make_context(max_iterations=10, iteration=5)

        decision = policy.decide(response, context)

        assert decision.decision == Decision.EXECUTE_TOOLS

    def test_no_iteration_limit_no_warning(self):
        """No warning when max_iterations is 0 (unlimited)."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_iterations=2)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        context = make_context(max_iterations=0, iteration=100)

        decision = policy.decide(response, context)

        assert decision.decision == Decision.EXECUTE_TOOLS

    def test_wrap_up_only_injected_once(self):
        """Wrap-up message not re-injected if already done."""
        policy = DefaultGovernorPolicy(wrap_up_threshold_secs=30.0)
        response = make_interpreted(ResponseEvent.WORK_TOOLS)
        context = make_context(timeout_secs=100, elapsed_secs=75, wrap_up_injected=True)

        decision = policy.decide(response, context)

        # Should execute normally, not inject again
        assert decision.decision == Decision.EXECUTE_TOOLS


class TestGovernorContext:
    """Tests for GovernorContext properties."""

    def test_has_prior_work_false_when_empty(self):
        """has_prior_work is False when no tools called."""
        context = make_context()
        assert context.has_prior_work is False

    def test_time_remaining_calculation(self):
        """time_remaining calculated correctly."""
        context = make_context(timeout_secs=100, elapsed_secs=30)
        assert context.time_remaining == 70.0

    def test_time_remaining_none_when_no_timeout(self):
        """time_remaining is None when no timeout set."""
        context = make_context(timeout_secs=0)
        assert context.time_remaining is None

    def test_time_remaining_clamped_to_zero(self):
        """time_remaining doesn't go negative."""
        context = make_context(timeout_secs=100, elapsed_secs=120)
        assert context.time_remaining == 0.0

    def test_iterations_remaining_calculation(self):
        """iterations_remaining calculated correctly."""
        context = make_context(max_iterations=10, iteration=3)
        assert context.iterations_remaining == 7

    def test_iterations_remaining_none_when_no_limit(self):
        """iterations_remaining is None when no limit set."""
        context = make_context(max_iterations=0)
        assert context.iterations_remaining is None
