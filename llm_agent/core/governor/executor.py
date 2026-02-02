"""Decision execution for the governor loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_agent.core.governor.messages import MessageBuilder
from llm_agent.core.governor.types import (
    Decision,
    GovernorContext,
    InterpretedResponse,
    PolicyDecision,
    ResponseEvent,
)
from llm_agent.core.llm import Message
from llm_agent.core.tools.base import ToolCallResult
from llm_agent.core.tools.executor import SimpleToolExecutor


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.governor.interpreter import ResponseInterpreter
    from llm_agent.core.llm import LLMCaller
    from llm_agent.core.tools.registry import ToolRegistry


class DecisionExecutor:
    """Executes policy decisions including LLM interactions.

    Handles the mechanics of executing each decision type:
    - Running tools and checking for terminal results
    - Confirming completion when LLM stops without tools
    - Confirming early terminal attempts
    - Injecting messages into conversation

    Example:
        executor = DecisionExecutor(lg, llm_caller, registry, interpreter)
        result = executor.execute(decision, interpreted, messages, tool_calls, tools, context)
        if result is not None:
            return build_final_result(result)
    """

    def __init__(
        self,
        lg: Logger,
        llm_caller: LLMCaller,
        registry: ToolRegistry,
        interpreter: ResponseInterpreter,
    ) -> None:
        """Initialize decision executor.

        Args:
            lg: Logger instance.
            llm_caller: LLM caller for completions.
            registry: Registry of available tools.
            interpreter: Response interpreter for classification.
        """
        self._lg = lg
        self._llm_caller = llm_caller
        self._interpreter = interpreter
        self._tool_executor = SimpleToolExecutor(lg, registry)
        self._messages = MessageBuilder()

    def execute(
        self,
        decision: PolicyDecision,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
        tools: list[dict[str, Any]] | None,
        context: GovernorContext,
    ) -> tuple[str | None, int, dict[str, Any] | None]:
        """Execute a policy decision.

        Args:
            decision: The policy decision to execute.
            interpreted: The interpreted LLM response.
            working_messages: Message list (modified in place).
            all_tool_calls: Tool call accumulator (modified in place).
            tools: OpenAI-format tool definitions.
            context: Current execution context.

        Returns:
            Tuple of (content, extra_tokens, terminal_data).
            - content=None means continue loop
            - content=str means finish with that content
            - extra_tokens always tracks tokens used by confirmation LLM calls
        """
        match decision.decision:
            case Decision.EXECUTE_TOOLS:
                return self._execute_tools(interpreted, working_messages, all_tool_calls)

            case Decision.CONFIRM_COMPLETION:
                return self._handle_confirm_completion(
                    interpreted, working_messages, all_tool_calls, tools, decision
                )

            case Decision.CONFIRM_EARLY:
                return self._handle_confirm_early(
                    interpreted, working_messages, all_tool_calls, tools, decision
                )

            case Decision.ACCEPT_TERMINAL:
                return self._accept_terminal(interpreted, working_messages, all_tool_calls)

            case Decision.INJECT_AND_CONTINUE:
                self._inject_message(decision.inject_message, working_messages)
                return None, 0, None

            case Decision.ABORT:
                raise RuntimeError(decision.abort_reason)

            case _:
                raise ValueError(f"Unhandled decision type: {decision.decision}")

    def _execute_tools(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
    ) -> tuple[str | None, int, dict[str, Any] | None]:
        """Execute tools and check for terminal."""
        tool_results = self._tool_executor.execute(
            list(interpreted.tool_calls), interpreted.parse_errors
        )
        all_tool_calls.extend(tool_results)
        self._messages.append_tool_round(
            working_messages, interpreted.raw, list(interpreted.tool_calls), tool_results
        )

        # Check for terminal tool result
        terminal_data = self._find_terminal_data(tool_results)
        if terminal_data is not None:
            self._lg.debug("terminal tool called")
            return interpreted.raw.content or "", 0, terminal_data

        return None, 0, None

    def _handle_confirm_completion(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
        tools: list[dict[str, Any]] | None,
        decision: PolicyDecision,
    ) -> tuple[str | None, int, dict[str, Any] | None]:
        """Handle CONFIRM_COMPLETION decision (no tools called)."""
        # Add assistant response and confirmation prompt
        working_messages.append(Message(role="assistant", content=interpreted.raw.content or ""))
        working_messages.append(Message(role="user", content=decision.confirm_prompt or ""))
        self._lg.trace("requesting completion confirmation")

        # Get LLM's response
        confirm_result = self._llm_caller.call(working_messages, tools)
        confirm_interpreted = self._interpreter.interpret(confirm_result, has_prior_work=False)

        if confirm_interpreted.event == ResponseEvent.TEXT_ONLY:
            # LLM confirmed done - remove confirmation prompt but keep assistant response
            working_messages.pop()  # Remove confirmation prompt
            self._lg.trace("completion confirmed")
            return interpreted.raw.content, confirm_result.tokens_used, None

        # LLM called tools - execute them
        return self._execute_confirmation_tools(
            confirm_interpreted, working_messages, all_tool_calls, confirm_result.tokens_used
        )

    def _handle_confirm_early(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
        tools: list[dict[str, Any]] | None,
        decision: PolicyDecision,
    ) -> tuple[str | None, int, dict[str, Any] | None]:
        """Handle CONFIRM_EARLY decision (terminal without prior work)."""
        self._lg.debug("early completion attempt, requesting confirmation")

        # Add the original response and confirmation prompt
        working_messages.append(
            self._messages.build_assistant(interpreted.raw, list(interpreted.tool_calls))
        )
        working_messages.append(Message(role="user", content=decision.confirm_prompt or ""))

        # Get LLM's response to confirmation
        confirm_result = self._llm_caller.call(working_messages, tools)
        confirm_interpreted = self._interpreter.interpret(confirm_result, has_prior_work=False)

        if confirm_interpreted.event == ResponseEvent.TEXT_ONLY:
            # No tools after prompt - continue loop
            self._lg.debug("no tools after early completion prompt, continuing")
            working_messages.append(Message(role="assistant", content=confirm_result.content or ""))
            return None, confirm_result.tokens_used, None

        # Check if LLM confirmed by calling terminal again
        if confirm_interpreted.terminal_call is not None:
            return self._accept_early_completion(
                confirm_interpreted, working_messages, all_tool_calls, confirm_result.tokens_used
            )

        # LLM called work tools - execute and continue
        self._continue_after_early_prompt(confirm_interpreted, working_messages, all_tool_calls)
        return None, confirm_result.tokens_used, None

    def _accept_terminal(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
    ) -> tuple[str, int, dict[str, Any] | None]:
        """Accept terminal tool after prior work."""
        tool_results = self._tool_executor.execute(
            list(interpreted.tool_calls), interpreted.parse_errors
        )
        all_tool_calls.extend(tool_results)
        terminal_data = self._find_terminal_data(tool_results)
        self._messages.append_tool_round(
            working_messages, interpreted.raw, list(interpreted.tool_calls), tool_results
        )

        self._lg.debug("terminal tool accepted")
        return interpreted.raw.content or "", 0, terminal_data or {}

    def _accept_early_completion(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
        extra_tokens: int,
    ) -> tuple[str, int, dict[str, Any] | None]:
        """Accept confirmed early completion."""
        self._lg.debug("early completion confirmed")
        tool_results = self._tool_executor.execute(
            list(interpreted.tool_calls), interpreted.parse_errors
        )
        all_tool_calls.extend(tool_results)
        terminal_data = self._find_terminal_data(tool_results)
        self._messages.append_tool_round(
            working_messages, interpreted.raw, list(interpreted.tool_calls), tool_results
        )

        return interpreted.raw.content or "", extra_tokens, terminal_data or {}

    def _continue_after_early_prompt(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
    ) -> None:
        """Execute work tools called after early completion prompt and continue."""
        self._lg.debug(
            "work tools called after early completion prompt",
            extra={"tools": [c.name for c in interpreted.tool_calls]},
        )
        tool_results = self._tool_executor.execute(
            list(interpreted.tool_calls), interpreted.parse_errors
        )
        all_tool_calls.extend(tool_results)
        self._messages.append_tool_round(
            working_messages, interpreted.raw, list(interpreted.tool_calls), tool_results
        )

    def _execute_confirmation_tools(
        self,
        interpreted: InterpretedResponse,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
        extra_tokens: int,
    ) -> tuple[str | None, int, dict[str, Any] | None]:
        """Execute tools called after confirmation prompt."""
        self._lg.debug(
            "tools called after confirmation",
            extra={"tool_count": len(interpreted.tool_calls)},
        )
        tool_results = self._tool_executor.execute(
            list(interpreted.tool_calls), interpreted.parse_errors
        )
        all_tool_calls.extend(tool_results)
        terminal_data = self._find_terminal_data(tool_results)
        self._messages.append_tool_round(
            working_messages, interpreted.raw, list(interpreted.tool_calls), tool_results
        )

        if terminal_data is not None:
            return interpreted.raw.content or "", extra_tokens, terminal_data

        return None, extra_tokens, None

    def _inject_message(self, message: str | None, working_messages: list[Message]) -> None:
        """Inject a user message into the conversation."""
        if message:
            working_messages.append(Message(role="user", content=message))
            self._lg.debug("injected wrap-up message")

    def _find_terminal_data(self, tool_results: list[ToolCallResult]) -> dict[str, Any] | None:
        """Find terminal data from tool results, if any tool was terminal.

        Returns terminal_data dict if present, empty dict if terminal but no data,
        or None if no terminal tool was called.
        """
        for tr in tool_results:
            if tr.result.terminal:
                return tr.result.terminal_data if tr.result.terminal_data is not None else {}
        return None
