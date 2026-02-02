"""Execution policy for governor decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from llm_agent.core.governor.types import Decision, PolicyDecision, ResponseEvent


if TYPE_CHECKING:
    from llm_agent.core.governor.types import GovernorContext, InterpretedResponse


@runtime_checkable
class GovernorPolicy(Protocol):
    """Protocol for execution policy implementations.

    Policies decide what action to take based on an interpreted response
    and the current execution context. Different policies can implement
    different behaviors (e.g., strict vs lenient completion checking).
    """

    def decide(
        self,
        response: InterpretedResponse,
        context: GovernorContext,
    ) -> PolicyDecision:
        """Determine action for this response.

        Args:
            response: Parsed and classified LLM response.
            context: Current execution state snapshot.

        Returns:
            PolicyDecision with the action to take.
        """
        ...


class DefaultGovernorPolicy:
    """Default policy matching current executor behavior plus graceful timeout.

    Decision logic:
    - TEXT_ONLY: Ask LLM to confirm completion
    - WORK_TOOLS: Execute the tools
    - TERMINAL_AFTER_WORK: Accept terminal, end loop
    - EARLY_TERMINAL: Challenge with confirmation prompt
    - MIXED_BATCH: Execute work tools first, handle terminal after

    Additional features:
    - Graceful timeout: Inject wrap-up warning before hard limit
    - Iteration warning: Similar warning for iteration limits
    """

    _COMPLETION_CONFIRM_PROMPT = (
        "You didn't call any tools in your last response. "
        "If you're done, call the completion tool. "
        "If you meant to take an action, call the appropriate tool now."
    )

    _EARLY_COMPLETION_PROMPT = (
        "You're completing the task without having used any tools first. "
        "If you truly cannot proceed, confirm by calling the completion tool again. "
        "Otherwise, use the available tools to work on the task."
    )

    _WRAP_UP_MESSAGE = (
        "You are approaching the time limit. Please wrap up your work now "
        "and call complete_task with your findings so far."
    )

    _ITERATION_WRAP_UP_MESSAGE = (
        "You are approaching the iteration limit. Please wrap up your work now "
        "and call complete_task with your findings so far."
    )

    def __init__(
        self,
        wrap_up_threshold_secs: float = 30.0,
        wrap_up_threshold_iterations: int = 2,
    ) -> None:
        """Initialize policy.

        Args:
            wrap_up_threshold_secs: Seconds before timeout to warn.
            wrap_up_threshold_iterations: Iterations before limit to warn.
        """
        self._wrap_up_threshold_secs = wrap_up_threshold_secs
        self._wrap_up_threshold_iterations = wrap_up_threshold_iterations

    def decide(
        self,
        response: InterpretedResponse,
        context: GovernorContext,
    ) -> PolicyDecision:
        """Determine action for this response."""
        # Check for approaching limits first (inject wrap-up if needed)
        limit_decision = self._check_approaching_limits(context)
        if limit_decision is not None:
            return limit_decision

        # Handle based on event type
        return self._decide_for_event(response, context)

    def _decide_for_event(
        self,
        response: InterpretedResponse,
        context: GovernorContext,
    ) -> PolicyDecision:
        """Make decision based on response event type."""
        match response.event:
            case ResponseEvent.TEXT_ONLY:
                return PolicyDecision(
                    decision=Decision.CONFIRM_COMPLETION,
                    confirm_prompt=self._COMPLETION_CONFIRM_PROMPT,
                )

            case ResponseEvent.WORK_TOOLS:
                return PolicyDecision(decision=Decision.EXECUTE_TOOLS)

            case ResponseEvent.TERMINAL_AFTER_WORK:
                return PolicyDecision(decision=Decision.ACCEPT_TERMINAL)

            case ResponseEvent.EARLY_TERMINAL:
                return PolicyDecision(
                    decision=Decision.CONFIRM_EARLY,
                    confirm_prompt=self._EARLY_COMPLETION_PROMPT,
                )

            case ResponseEvent.MIXED_BATCH:
                # Execute all tools - if terminal succeeds, it ends the loop
                # This matches current behavior of processing all tools in batch
                return PolicyDecision(decision=Decision.EXECUTE_TOOLS)

    def _check_approaching_limits(
        self,
        context: GovernorContext,
    ) -> PolicyDecision | None:
        """Check if approaching execution limits and return wrap-up decision if so.

        Returns:
            PolicyDecision to inject wrap-up message, or None to continue normally.
        """
        # Don't inject wrap-up multiple times
        if context.wrap_up_injected:
            return None

        # Check time limit
        if (
            context.time_remaining is not None
            and context.time_remaining <= self._wrap_up_threshold_secs
        ):
            return PolicyDecision(
                decision=Decision.INJECT_AND_CONTINUE,
                inject_message=self._WRAP_UP_MESSAGE,
            )

        # Check iteration limit
        if (
            context.iterations_remaining is not None
            and context.iterations_remaining <= self._wrap_up_threshold_iterations
        ):
            return PolicyDecision(
                decision=Decision.INJECT_AND_CONTINUE,
                inject_message=self._ITERATION_WRAP_UP_MESSAGE,
            )

        return None
