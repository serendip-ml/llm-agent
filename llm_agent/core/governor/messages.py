"""Message building utilities for the governor loop."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from llm_agent.core.governor.helpers import truncate_str
from llm_agent.core.llm import Message
from llm_agent.core.tools.base import ToolCall, ToolCallResult, ToolResult


if TYPE_CHECKING:
    from llm_agent.core.llm import CompletionResult


class MessageBuilder:
    """Builds and manipulates conversation messages for the governor loop.

    Handles the construction of OpenAI-format messages for tool interactions:
    - Assistant messages with tool calls
    - Tool result messages
    - Appending complete tool rounds to conversation

    Example:
        builder = MessageBuilder()
        messages.append(builder.build_assistant(llm_result, tool_calls))
        for result in tool_results:
            messages.append(builder.build_tool_result(result))
    """

    def build_assistant(self, result: CompletionResult, tool_calls: list[ToolCall]) -> Message:
        """Build assistant message containing tool calls.

        Args:
            result: LLM completion result.
            tool_calls: Parsed tool calls from the response.

        Returns:
            Message with role="assistant" and tool_calls in OpenAI format.
        """
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

    def build_tool_result(self, tool_result: ToolCallResult) -> Message:
        """Build tool result message.

        Args:
            tool_result: Result from executing a tool.

        Returns:
            Message with role="tool" and formatted content.
        """
        return Message(
            role="tool",
            content=self._format_result(tool_result.result),
            tool_call_id=tool_result.call_id,
        )

    def append_tool_round(
        self,
        messages: list[Message],
        result: CompletionResult,
        tool_calls: list[ToolCall],
        tool_results: list[ToolCallResult],
    ) -> None:
        """Append a complete tool round (assistant + tool results) to conversation.

        Args:
            messages: Message list to append to (modified in place).
            result: LLM completion result.
            tool_calls: Tool calls from the LLM.
            tool_results: Results from executing the tools.
        """
        messages.append(self.build_assistant(result, tool_calls))
        for tr in tool_results:
            messages.append(self.build_tool_result(tr))

    def find_last_assistant(self, messages: list[Message], max_len: int = 200) -> str:
        """Find and return the last assistant message content (truncated).

        Args:
            messages: Conversation messages.
            max_len: Maximum length before truncation.

        Returns:
            Truncated content of last assistant message, or empty string.
        """
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                return truncate_str(msg.content, max_len)
        return ""

    def _format_result(self, result: ToolResult) -> str:
        """Format tool result for LLM consumption."""
        return result.output if result.success else f"Error: {result.error}"
