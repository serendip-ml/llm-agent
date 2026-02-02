"""Tool execution loop for LLM ↔ tool interaction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from llm_agent.core.llm import Message
from llm_agent.core.task import Task
from llm_agent.core.tools.base import ToolCall, ToolCallResult, ToolResult
from llm_agent.core.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.llm import CompletionResult, LLMBackend


class ToolExecutionResult(BaseModel):
    """Result from tool execution loop."""

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


class ToolExecutor:
    """Executes tool calls in a loop with the LLM.

    Handles the back-and-forth between the LLM generating tool calls
    and executing those tools, feeding results back to the LLM until
    it produces a final response.

    Example:
        task = Task(name="example", description="...", max_iterations=5, timeout_secs=60)
        executor = ToolExecutor(lg, llm, registry, task)
        result = executor.run(messages=[Message(role="user", content="List files")])
        print(result.content)  # LLM's final response
        print(result.tool_calls)  # Tools that were called
    """

    def __init__(
        self,
        lg: Logger,
        llm: LLMBackend,
        registry: ToolRegistry,
        task: Task,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize executor.

        Args:
            lg: Logger instance.
            llm: LLM backend for completions.
            registry: Registry of available tools.
            task: Task being executed (provides iteration/timeout limits).
            model: Model to use for completions (optional).
            temperature: Temperature for LLM completions.
        """
        self._lg = lg
        self._llm = llm
        self._registry = registry
        self._task = task
        self._model = model
        self._temperature = temperature

    def run(self, messages: list[Message]) -> ToolExecutionResult:
        """Run the tool execution loop.

        Args:
            messages: Initial messages to send to LLM.

        Returns:
            ToolExecutionResult with final content and execution trace.

        Raises:
            RuntimeError: If max_iterations or timeout exceeded without final response.
        """
        import time

        working_messages = list(messages)
        all_tool_calls: list[ToolCallResult] = []
        total_tokens = 0
        tools = self._registry.to_openai_tools() or None
        start_time = time.monotonic()
        iteration = 0

        while True:
            self._check_limits(iteration, start_time)
            iteration_result, tokens = self._run_iteration(
                working_messages, all_tool_calls, total_tokens, iteration, tools
            )
            total_tokens += tokens
            if iteration_result is not None:
                return iteration_result

            iteration += 1

    def _check_limits(self, iteration: int, start_time: float) -> None:
        """Check iteration and timeout limits, raise if exceeded."""
        import time

        task = self._task
        if task.max_iterations > 0 and iteration >= task.max_iterations:
            raise RuntimeError(
                f"Tool execution exceeded {task.max_iterations} iterations without final response"
            )
        if task.timeout_secs > 0 and (time.monotonic() - start_time) >= task.timeout_secs:
            raise RuntimeError(
                f"Tool execution exceeded {task.timeout_secs}s timeout without final response"
            )

    def _run_iteration(
        self,
        working_messages: list[Message],
        all_tool_calls: list[ToolCallResult],
        total_tokens: int,
        iteration: int,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[ToolExecutionResult | None, int]:
        """Run a single iteration. Returns (result, tokens_used). Result is None to continue."""
        self._lg.trace("tool loop iteration", extra={"iteration": iteration + 1})

        result = self._call_llm(working_messages, tools)
        tokens_used = result.tokens_used
        new_total = total_tokens + tokens_used
        tool_calls, parse_errors = self._extract_tool_calls(result)

        if not tool_calls:
            return self._handle_no_tool_calls(
                working_messages, result, all_tool_calls, iteration, tokens_used, new_total, tools
            )

        return self._handle_tool_calls(
            working_messages,
            result,
            all_tool_calls,
            tool_calls,
            parse_errors,
            iteration,
            tokens_used,
            new_total,
        )

    def _handle_no_tool_calls(
        self,
        working_messages: list[Message],
        result: CompletionResult,
        all_tool_calls: list[ToolCallResult],
        iteration: int,
        tokens_used: int,
        new_total: int,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[ToolExecutionResult | None, int]:
        """Handle iteration where LLM made no tool calls."""
        confirmation = self._confirm_completion(working_messages, result, all_tool_calls, tools)
        if confirmation is not None:
            extra_tokens, had_terminal = confirmation
            if had_terminal:
                terminal_data = self._find_terminal_data(all_tool_calls)
                return self._build_terminal_result(
                    result,
                    all_tool_calls,
                    working_messages,
                    iteration + 1,
                    new_total + extra_tokens,
                    terminal_data or {},
                ), tokens_used + extra_tokens
            return None, tokens_used + extra_tokens

        self._lg.debug("tool loop finished", extra={"iterations": iteration + 1})
        final = self._build_final_result(
            result, all_tool_calls, working_messages, iteration + 1, new_total
        )
        return final, tokens_used

    def _handle_tool_calls(
        self,
        working_messages: list[Message],
        result: CompletionResult,
        all_tool_calls: list[ToolCallResult],
        tool_calls: list[ToolCall],
        parse_errors: dict[str, str],
        iteration: int,
        tokens_used: int,
        new_total: int,
    ) -> tuple[ToolExecutionResult | None, int]:
        """Handle iteration where LLM made tool calls."""
        tool_results = self._execute_tool_calls(tool_calls, parse_errors)
        all_tool_calls.extend(tool_results)

        terminal_data = self._find_terminal_data(tool_results)
        if terminal_data is not None:
            self._lg.debug("terminal tool called", extra={"iterations": iteration + 1})
            terminal = self._build_terminal_result(
                result, all_tool_calls, working_messages, iteration + 1, new_total, terminal_data
            )
            return terminal, tokens_used

        self._append_tool_messages(working_messages, result, tool_calls, tool_results)
        return None, tokens_used

    def _build_terminal_result(
        self,
        result: CompletionResult,
        tool_calls: list[ToolCallResult],
        messages: list[Message],
        iterations: int,
        total_tokens: int,
        terminal_data: dict[str, Any],
    ) -> ToolExecutionResult:
        """Build result when terminal tool was called."""
        return ToolExecutionResult(
            content=result.content,
            tool_calls=tool_calls,
            messages=messages,
            iterations=iterations,
            total_tokens=total_tokens,
            terminal_data=terminal_data,
        )

    def _call_llm(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> CompletionResult:
        """Call LLM with messages and tools."""
        tool_count = len(tools) if tools else 0
        last_assistant_msg = self._find_last_assistant_message(messages)
        self._lg.debug(
            "calling LLM...",
            extra={
                "message_count": len(messages),
                "last_assistant": last_assistant_msg,
                "tool_count": tool_count,
            },
        )

        result = self._llm.complete(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            tools=tools,
        )

        self._log_llm_response(result)
        return result

    def _log_llm_response(self, result: CompletionResult) -> None:
        """Log LLM response details."""
        tool_call_count = len(result.tool_calls) if result.tool_calls else 0
        self._lg.trace(
            "LLM response",
            extra={
                "tokens": result.tokens_used,
                "tool_calls": tool_call_count,
                "content_len": len(result.content) if result.content else 0,
            },
        )

        if result.content:
            content_preview = self._truncate_str(result.content, 200)
            self._lg.debug("LLM content", extra={"content": content_preview})
        else:
            self._lg.trace("LLM response without content")

    def _build_final_result(
        self,
        result: CompletionResult,
        tool_calls: list[ToolCallResult],
        messages: list[Message],
        iterations: int,
        total_tokens: int,
    ) -> ToolExecutionResult:
        """Build final execution result when LLM is done."""
        return ToolExecutionResult(
            content=result.content,
            tool_calls=tool_calls,
            messages=messages,
            iterations=iterations,
            total_tokens=total_tokens,
        )

    _COMPLETION_CONFIRM_PROMPT = (
        "You didn't call any tools in your last response. "
        "If you're done, call the completion tool. "
        "If you meant to take an action, call the appropriate tool now."
    )

    def _confirm_completion(
        self,
        working_messages: list[Message],
        result: CompletionResult,
        all_tool_calls: list[ToolCallResult],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[int, bool] | None:
        """Ask LLM to confirm completion when no tool calls were made.

        Returns:
            None if LLM confirms done (no tools called after prompt).
            (tokens_used, had_terminal) if LLM called tools after prompt.
        """
        working_messages.append(Message(role="assistant", content=result.content or ""))
        working_messages.append(Message(role="user", content=self._COMPLETION_CONFIRM_PROMPT))
        self._lg.trace("requesting completion confirmation")

        confirm_result = self._call_llm(working_messages, tools)
        confirm_tool_calls, parse_errors = self._extract_tool_calls(confirm_result)

        if not confirm_tool_calls:
            self._lg.trace("completion confirmed")
            return None

        return self._execute_confirmation_tools(
            working_messages, confirm_result, confirm_tool_calls, parse_errors, all_tool_calls
        )

    def _execute_confirmation_tools(
        self,
        working_messages: list[Message],
        confirm_result: CompletionResult,
        confirm_tool_calls: list[ToolCall],
        parse_errors: dict[str, str],
        all_tool_calls: list[ToolCallResult],
    ) -> tuple[int, bool]:
        """Execute tools called after confirmation prompt."""
        self._lg.debug(
            "tools called after confirmation", extra={"tool_count": len(confirm_tool_calls)}
        )
        tool_results = self._execute_tool_calls(confirm_tool_calls, parse_errors)
        all_tool_calls.extend(tool_results)

        terminal_data = self._find_terminal_data(tool_results)
        self._append_tool_messages(
            working_messages, confirm_result, confirm_tool_calls, tool_results
        )
        return confirm_result.tokens_used, terminal_data is not None

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
        # Log what tools the LLM wants to call
        tool_names = [c.name for c in calls]
        self._lg.trace("LLM requested tools", extra={"tools": tool_names})

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
            self._lg.warning("unknown tool", extra={"tool": call.name})
            return ToolResult(success=False, output="", error=f"Unknown tool: {call.name}")

        # Log tool call with actual arguments (truncate long values)
        args_preview = self._truncate_args(call.arguments)
        self._lg.debug("executing tool", extra={"tool": call.name, "input": args_preview})

        try:
            result = tool.execute(**call.arguments)

            # Log result with actual output (truncated)
            if result.success:
                output_preview = self._truncate_str(result.output, 300)
                self._lg.trace(
                    "tool result",
                    extra={"tool": call.name, "success": True, "output": output_preview},
                )
            else:
                self._lg.trace(
                    "tool result",
                    extra={"tool": call.name, "success": False, "error": result.error},
                )
            return result
        except Exception as e:
            self._lg.warning("tool error", extra={"tool": call.name, "exception": e})
            return ToolResult(success=False, output="", error=f"Tool execution error: {e}")

    def _find_last_assistant_message(self, messages: list[Message], max_len: int = 200) -> str:
        """Find and return the last assistant message content (truncated)."""
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                return self._truncate_str(msg.content, max_len)
        return ""

    def _truncate_str(self, s: str | None, max_len: int = 200) -> str:
        """Truncate string for logging."""
        if not s:
            return ""
        return s[:max_len] + "..." if len(s) > max_len else s

    def _truncate_args(self, args: dict[str, Any], max_value_len: int = 100) -> dict[str, Any]:
        """Truncate argument values for logging."""
        result = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > max_value_len:
                result[k] = v[:max_value_len] + "..."
            else:
                result[k] = v
        return result

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format tool result for LLM consumption."""
        return result.output if result.success else f"Error: {result.error}"

    def _find_terminal_data(self, tool_results: list[ToolCallResult]) -> dict[str, Any] | None:
        """Find terminal data from tool results, if any tool was terminal.

        Returns terminal_data dict if present, empty dict if terminal but no data,
        or None if no terminal tool was called.
        """
        for tr in tool_results:
            if tr.result.terminal:
                return tr.result.terminal_data if tr.result.terminal_data is not None else {}
        return None
