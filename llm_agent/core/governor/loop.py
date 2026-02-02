"""Main governor execution loop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from llm_agent.core.governor.executor import DecisionExecutor
from llm_agent.core.governor.interpreter import ResponseInterpreter
from llm_agent.core.governor.policy import DefaultGovernorPolicy, GovernorPolicy
from llm_agent.core.governor.types import Decision, GovernorContext
from llm_agent.core.llm import LLMCaller, Message
from llm_agent.core.tools.base import ToolCallResult


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.llm import LLMBackend
    from llm_agent.core.task import Task
    from llm_agent.core.tools.registry import ToolRegistry


class GovernorResult(BaseModel):
    """Result from governor execution loop."""

    content: str
    """Final response content from the LLM."""

    tool_calls: list[ToolCallResult]
    """All tool calls made during execution."""

    messages: list[Message]
    """Complete message history including all tool interactions."""

    iterations: int
    """Number of LLM round-trips."""

    total_tokens: int
    """Total tokens used across all iterations."""

    terminal_data: dict[str, Any] | None = None
    """Data from terminal tool call (e.g., task completion info)."""


class _LoopState:
    """Mutable state for the execution loop."""

    __slots__ = (
        "messages",
        "tool_calls",
        "total_tokens",
        "tools",
        "start_time",
        "iteration",
        "wrap_up_injected",
    )

    def __init__(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        self.messages = messages
        self.tool_calls: list[ToolCallResult] = []
        self.total_tokens = 0
        self.tools = tools
        self.start_time = time.monotonic()
        self.iteration = 0
        self.wrap_up_injected = False


class GovernorLoop:
    """Main execution loop - interprets, decides, executes.

    Coordinates the LLM -> interpret -> decide -> execute cycle:
    1. Call LLM with current messages
    2. Interpret response (classify event type)
    3. Consult policy for decision
    4. Execute decision (run tools, inject messages, etc.)
    5. Repeat until terminal or limits exceeded

    Example:
        interpreter = ResponseInterpreter(registry)
        policy = DefaultGovernorPolicy()
        loop = GovernorLoop(lg, llm, registry, interpreter, policy)
        result = loop.run(task, messages)
    """

    def __init__(
        self,
        lg: Logger,
        llm: LLMBackend,
        registry: ToolRegistry,
        interpreter: ResponseInterpreter | None = None,
        policy: GovernorPolicy | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize governor loop.

        Args:
            lg: Logger instance.
            llm: LLM backend for completions.
            registry: Registry of available tools.
            interpreter: Response interpreter (uses default if None).
            policy: Execution policy (uses DefaultGovernorPolicy if None).
            model: Model to use for completions (optional).
            temperature: Temperature for LLM completions.
        """
        self._lg = lg
        self._registry = registry
        self._interpreter = interpreter or ResponseInterpreter(registry)
        self._policy = policy or DefaultGovernorPolicy()
        self._llm_caller = LLMCaller(lg, llm, model, temperature)
        self._executor = DecisionExecutor(
            lg=lg,
            llm_caller=self._llm_caller,
            registry=registry,
            interpreter=self._interpreter,
        )

    def run(self, task: Task, messages: list[Message]) -> GovernorResult:
        """Run the execution loop.

        Args:
            task: Task being executed (provides iteration/timeout limits).
            messages: Initial messages to send to LLM.

        Returns:
            GovernorResult with final content and execution trace.

        Raises:
            RuntimeError: If max_iterations or timeout exceeded without completion.
        """
        state = _LoopState(
            messages=list(messages),
            tools=self._registry.to_openai_tools() or None,
        )

        while True:
            self._check_limits(task, state.iteration, state.start_time)
            context = self._build_context(task, state)

            self._lg.trace("governor loop iteration", extra={"iteration": state.iteration + 1})
            result = self._llm_caller.call(state.messages, state.tools)
            state.total_tokens += result.tokens_used

            interpreted = self._interpreter.interpret(result, context.has_prior_work)
            decision = self._policy.decide(interpreted, context)

            content, extra_tokens, terminal_data = self._executor.execute(
                decision, interpreted, state.messages, state.tool_calls, state.tools, context
            )
            state.total_tokens += extra_tokens

            if content is not None:
                return self._build_result(state, content, terminal_data)

            if decision.decision == Decision.INJECT_AND_CONTINUE:
                state.wrap_up_injected = True
            state.iteration += 1

    def _build_context(self, task: Task, state: _LoopState) -> GovernorContext:
        """Build context snapshot for policy decision."""
        return GovernorContext(
            task=task,
            iteration=state.iteration,
            elapsed_secs=time.monotonic() - state.start_time,
            tokens_used=state.total_tokens,
            tool_calls_so_far=tuple(state.tool_calls),
            wrap_up_injected=state.wrap_up_injected,
        )

    def _build_result(
        self,
        state: _LoopState,
        content: str,
        terminal_data: dict[str, Any] | None,
    ) -> GovernorResult:
        """Build final result from loop state and completion data."""
        return GovernorResult(
            content=content,
            tool_calls=state.tool_calls,
            messages=state.messages,
            iterations=state.iteration + 1,
            total_tokens=state.total_tokens,
            terminal_data=terminal_data,
        )

    def _check_limits(self, task: Task, iteration: int, start_time: float) -> None:
        """Check iteration and timeout limits, raise if exceeded."""
        if task.max_iterations > 0 and iteration >= task.max_iterations:
            raise RuntimeError(
                f"Tool execution exceeded {task.max_iterations} iterations without final response"
            )
        if task.timeout_secs > 0 and (time.monotonic() - start_time) >= task.timeout_secs:
            raise RuntimeError(
                f"Tool execution exceeded {task.timeout_secs}s timeout without final response"
            )
