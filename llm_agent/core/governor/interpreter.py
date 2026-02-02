"""Response interpreter for classifying LLM responses."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llm_agent.core.governor.types import InterpretedResponse, ResponseEvent
from llm_agent.core.tools.base import ToolCall


if TYPE_CHECKING:
    from llm_agent.core.llm import CompletionResult
    from llm_agent.core.tools.registry import ToolRegistry


class ResponseInterpreter:
    """Parses LLM responses into semantic events.

    Pure interpreter with no side effects. Takes a registry to identify
    terminal tools but doesn't execute anything.

    Example:
        interpreter = ResponseInterpreter(registry)
        interpreted = interpreter.interpret(result, has_prior_work=True)
        if interpreted.event == ResponseEvent.TERMINAL_AFTER_WORK:
            # Policy can accept this terminal
            ...
    """

    def __init__(self, registry: ToolRegistry) -> None:
        """Initialize interpreter.

        Args:
            registry: Tool registry to identify terminal tools.
        """
        self._registry = registry

    def interpret(
        self,
        result: CompletionResult,
        has_prior_work: bool,
    ) -> InterpretedResponse:
        """Parse and classify an LLM response.

        Args:
            result: Raw LLM completion result.
            has_prior_work: Whether non-terminal tools have been called before.

        Returns:
            InterpretedResponse with event classification and parsed data.
        """
        tool_calls, parse_errors = self._extract_tool_calls(result)

        if not tool_calls:
            return InterpretedResponse(
                event=ResponseEvent.TEXT_ONLY,
                raw=result,
                tool_calls=(),
                terminal_call=None,
                parse_errors=parse_errors,
            )

        terminal_call = self._find_terminal_call(tool_calls)
        has_work_calls = self._has_work_calls(tool_calls)

        event = self._classify_event(terminal_call, has_work_calls, has_prior_work)

        return InterpretedResponse(
            event=event,
            raw=result,
            tool_calls=tuple(tool_calls),
            terminal_call=terminal_call,
            parse_errors=parse_errors,
        )

    def _classify_event(
        self,
        terminal_call: ToolCall | None,
        has_work_calls: bool,
        has_prior_work: bool,
    ) -> ResponseEvent:
        """Classify the response event based on tool call patterns."""
        if terminal_call is None:
            # No terminal tool - just work tools
            return ResponseEvent.WORK_TOOLS

        if has_work_calls:
            # Both work and terminal in same batch
            return ResponseEvent.MIXED_BATCH

        # Terminal only - check if there's prior work
        if has_prior_work:
            return ResponseEvent.TERMINAL_AFTER_WORK

        return ResponseEvent.EARLY_TERMINAL

    def _extract_tool_calls(
        self, result: CompletionResult
    ) -> tuple[list[ToolCall], dict[str, str]]:
        """Extract tool calls from LLM response.

        Returns:
            Tuple of (tool_calls, parse_errors). Parse errors are keyed by
            call_id so they can be reported back to the LLM for specific calls.
        """
        if not result.tool_calls:
            return [], {}

        calls: list[ToolCall] = []
        errors: dict[str, str] = {}

        for tc in result.tool_calls:
            parsed, error = self._parse_tool_call(tc)
            if parsed.name:  # Only include if we got a valid name
                calls.append(parsed)
                if error:
                    errors[parsed.id] = error

        return calls, errors

    def _parse_tool_call(self, tc: Any) -> tuple[ToolCall, str | None]:
        """Parse a single tool call from API response.

        Returns:
            Tuple of (ToolCall, error_message). If parsing failed, error_message
            describes the issue (arguments will be empty dict in this case).
        """
        if isinstance(tc, dict):
            call_id = tc.get("id", "")
            function = tc.get("function", {})
            name = function.get("name", "")
            args_str = function.get("arguments", "{}")
        else:
            call_id = getattr(tc, "id", "")
            function = getattr(tc, "function", None)
            if not function:
                return ToolCall(id=call_id, name="", arguments={}), "Missing function in tool call"
            name = getattr(function, "name", "")
            args_str = getattr(function, "arguments", "{}")

        parse_error: str | None = None
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError as e:
            arguments = {}
            parse_error = f"Failed to parse arguments for tool '{name}': {e}"

        return ToolCall(id=call_id, name=name, arguments=arguments), parse_error

    def _find_terminal_call(self, tool_calls: list[ToolCall]) -> ToolCall | None:
        """Find a terminal tool call in the list, if any."""
        for call in tool_calls:
            tool = self._registry.get(call.name)
            if tool is not None and getattr(tool, "terminal", False):
                return call
        return None

    def _has_work_calls(self, tool_calls: list[ToolCall]) -> bool:
        """Check if there are any non-terminal tool calls."""
        for call in tool_calls:
            tool = self._registry.get(call.name)
            if tool is None or not getattr(tool, "terminal", False):
                return True
        return False
