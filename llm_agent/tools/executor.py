"""Tool execution loop for LLM ↔ tool interaction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from llm_agent.llm import Message
from llm_agent.tools.base import ToolCall, ToolCallResult, ToolResult
from llm_agent.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from llm_agent.llm import CompletionResult, LLMBackend


class ToolExecutionResult(BaseModel):
    """Result from tool execution loop."""

    content: str
    """Final response content from the LLM."""

    tool_calls: list[ToolCallResult]
    """All tool calls made during execution."""

    iterations: int
    """Number of LLM round-trips."""

    total_tokens: int
    """Total tokens used across all iterations."""


class ToolExecutor:
    """Executes tool calls in a loop with the LLM.

    Handles the back-and-forth between the LLM generating tool calls
    and executing those tools, feeding results back to the LLM until
    it produces a final response.

    Example:
        executor = ToolExecutor(llm, registry)
        result = executor.run(
            messages=[Message(role="user", content="List files in current dir")],
            max_iterations=5,
        )
        print(result.content)  # LLM's final response
        print(result.tool_calls)  # Tools that were called
    """

    def __init__(
        self,
        llm: LLMBackend,
        registry: ToolRegistry,
        model: str | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            llm: LLM backend for completions.
            registry: Registry of available tools.
            model: Model to use for completions (optional).
        """
        self._llm = llm
        self._registry = registry
        self._model = model

    def run(
        self,
        messages: list[Message],
        max_iterations: int = 10,
        temperature: float = 0.7,
    ) -> ToolExecutionResult:
        """Run the tool execution loop.

        Args:
            messages: Initial messages to send to LLM.
            max_iterations: Maximum number of LLM round-trips.
            temperature: Temperature for LLM completions.

        Returns:
            ToolExecutionResult with final content and execution trace.

        Raises:
            RuntimeError: If max_iterations exceeded without final response.
        """
        working_messages = list(messages)
        all_tool_calls: list[ToolCallResult] = []
        total_tokens = 0
        tools = self._registry.to_openai_tools() or None

        for iteration in range(max_iterations):
            result = self._call_llm(working_messages, temperature, tools)
            total_tokens += result.tokens_used
            tool_calls, parse_errors = self._extract_tool_calls(result)

            if not tool_calls:
                return self._build_final_result(result, all_tool_calls, iteration + 1, total_tokens)

            tool_results = self._execute_tool_calls(tool_calls, parse_errors)
            all_tool_calls.extend(tool_results)
            self._append_tool_messages(working_messages, result, tool_calls, tool_results)

        raise RuntimeError(
            f"Tool execution exceeded {max_iterations} iterations without final response"
        )

    def _call_llm(
        self, messages: list[Message], temperature: float, tools: list[dict[str, Any]] | None
    ) -> CompletionResult:
        """Call LLM with messages and tools."""
        return self._llm.complete(
            messages=messages,
            model=self._model,
            temperature=temperature,
            tools=tools,
        )

    def _build_final_result(
        self,
        result: CompletionResult,
        tool_calls: list[ToolCallResult],
        iterations: int,
        total_tokens: int,
    ) -> ToolExecutionResult:
        """Build final execution result when LLM is done."""
        return ToolExecutionResult(
            content=result.content,
            tool_calls=tool_calls,
            iterations=iterations,
            total_tokens=total_tokens,
        )

    def _append_tool_messages(
        self,
        messages: list[Message],
        result: CompletionResult,
        tool_calls: list[ToolCall],
        tool_results: list[ToolCallResult],
    ) -> None:
        """Append assistant and tool messages to conversation."""
        messages.append(self._build_assistant_message(result, tool_calls))
        for tr in tool_results:
            messages.append(self._build_tool_message(tr))

    def _build_assistant_message(
        self, result: CompletionResult, tool_calls: list[ToolCall]
    ) -> Message:
        """Build assistant message containing tool calls."""
        return Message(
            role="assistant",
            content=result.content or "",
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ],
        )

    def _build_tool_message(self, tool_result: ToolCallResult) -> Message:
        """Build tool result message."""
        return Message(
            role="tool",
            content=self._format_tool_result(tool_result.result),
            tool_call_id=tool_result.call_id,
        )

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
        calls = []
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

    def _execute_tool_calls(
        self, calls: list[ToolCall], parse_errors: dict[str, str]
    ) -> list[ToolCallResult]:
        """Execute a list of tool calls.

        Args:
            calls: Tool calls to execute.
            parse_errors: Dict mapping call_id to parse error message. If a call
                has a parse error, return the error instead of executing.
        """
        results = []
        for call in calls:
            if call.id in parse_errors:
                # Arguments couldn't be parsed - return error to LLM
                result = ToolResult(success=False, output="", error=parse_errors[call.id])
            else:
                result = self._execute_single_tool(call)
            results.append(ToolCallResult(call_id=call.id, name=call.name, result=result))
        return results

    def _execute_single_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self._registry.get(call.name)
        if tool is None:
            return ToolResult(success=False, output="", error=f"Unknown tool: {call.name}")

        try:
            return tool.execute(**call.arguments)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Tool execution error: {e}")

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format tool result for LLM consumption."""
        return result.output if result.success else f"Error: {result.error}"
