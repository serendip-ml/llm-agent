"""Types for the governor layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from llm_agent.core.llm import CompletionResult
    from llm_agent.core.task import Task
    from llm_agent.core.tools.base import ToolCall, ToolCallResult


class ResponseEvent(Enum):
    """What semantically happened in an LLM response.

    Used by the interpreter to classify LLM responses for policy decisions.
    """

    TEXT_ONLY = auto()
    """No tool calls in response."""

    WORK_TOOLS = auto()
    """Non-terminal tools called (normal work)."""

    TERMINAL_AFTER_WORK = auto()
    """Terminal tool called after prior work - accept."""

    EARLY_TERMINAL = auto()
    """Terminal tool without prior work - needs confirmation."""

    MIXED_BATCH = auto()
    """Work + terminal in same response (unusual)."""


class Decision(Enum):
    """What action to take based on policy evaluation.

    The governor loop uses these decisions to determine next steps.
    """

    EXECUTE_TOOLS = auto()
    """Execute the requested tools and continue."""

    CONFIRM_COMPLETION = auto()
    """Ask LLM to confirm completion (no tools case)."""

    CONFIRM_EARLY = auto()
    """Challenge early terminal attempt."""

    ACCEPT_TERMINAL = auto()
    """Accept terminal tool, end loop."""

    INJECT_AND_CONTINUE = auto()
    """Inject a message (e.g., wrap-up warning) and continue."""

    ABORT = auto()
    """Raise error and stop execution."""


@dataclass(frozen=True)
class GovernorContext:
    """Immutable snapshot of execution state for policy decisions.

    Provides all the information a policy needs to make decisions without
    giving it mutable access to the execution state.
    """

    task: Task
    """The task being executed."""

    iteration: int
    """Current iteration number (0-indexed)."""

    elapsed_secs: float
    """Seconds elapsed since execution started."""

    tokens_used: int
    """Total tokens consumed so far."""

    tool_calls_so_far: tuple[ToolCallResult, ...]
    """All tool calls made in this execution (immutable)."""

    wrap_up_injected: bool = False
    """Whether wrap-up message has already been injected."""

    @property
    def has_prior_work(self) -> bool:
        """Check if any non-terminal tools have been called."""
        return any(not tr.result.terminal for tr in self.tool_calls_so_far)

    @property
    def time_remaining(self) -> float | None:
        """Seconds remaining before timeout, or None if no timeout."""
        if self.task.timeout_secs <= 0:
            return None
        return max(0.0, self.task.timeout_secs - self.elapsed_secs)

    @property
    def iterations_remaining(self) -> int | None:
        """Iterations remaining before limit, or None if no limit."""
        if self.task.max_iterations <= 0:
            return None
        return max(0, self.task.max_iterations - self.iteration)


@dataclass(frozen=True)
class InterpretedResponse:
    """Parsed and classified LLM response.

    The interpreter produces this from raw LLM responses, making it easier
    for the policy to make decisions without re-parsing.
    """

    event: ResponseEvent
    """Semantic classification of what happened."""

    raw: CompletionResult
    """Original LLM response."""

    tool_calls: tuple[ToolCall, ...]
    """Parsed tool calls (may be empty)."""

    terminal_call: ToolCall | None
    """The terminal tool call, if any."""

    parse_errors: dict[str, str] = field(default_factory=dict)
    """Parse errors keyed by call_id (for malformed arguments)."""


@dataclass(frozen=True)
class PolicyDecision:
    """Result of policy evaluation.

    Contains the decision and any additional data needed to execute it.
    """

    decision: Decision
    """The action to take."""

    inject_message: str | None = None
    """Message to inject (for INJECT_AND_CONTINUE)."""

    confirm_prompt: str | None = None
    """Prompt to use for confirmation (for CONFIRM_* decisions)."""

    abort_reason: str | None = None
    """Reason for abort (for ABORT decision)."""

    def __post_init__(self) -> None:
        """Validate decision-specific fields."""
        if self.decision == Decision.INJECT_AND_CONTINUE and not self.inject_message:
            raise ValueError("INJECT_AND_CONTINUE requires inject_message")
        if self.decision == Decision.ABORT and not self.abort_reason:
            raise ValueError("ABORT requires abort_reason")
