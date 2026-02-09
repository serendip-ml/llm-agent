"""LLM calling with logging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import Message


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.llm.backend import LLMBackend
    from llm_agent.core.llm.types import CompletionResult


class LLMCaller:
    """Wraps LLM calls with consistent logging.

    Provides a single place for LLM interaction with:
    - Pre-call logging (message count, last assistant content, tool count)
    - Post-call logging (tokens, tool calls, content preview)

    Example:
        caller = LLMCaller(lg, llm, model="gpt-4", temperature=0.7)
        result = caller.call(messages, tools)
    """

    def __init__(
        self,
        lg: Logger,
        llm: LLMBackend,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize LLM caller.

        Args:
            lg: Logger instance.
            llm: LLM backend for completions.
            model: Model to use for completions (optional).
            temperature: Temperature for LLM completions.
        """
        self._lg = lg
        self._llm = llm
        self._model = model
        self._temperature = temperature

    def call(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> CompletionResult:
        """Call LLM with messages and tools.

        Args:
            messages: Conversation messages to send.
            tools: OpenAI-format tool definitions (optional).

        Returns:
            CompletionResult from the LLM.
        """
        self._log_pre_call(messages, tools)

        result = self._llm.complete(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            tools=tools,
        )

        self._log_post_call(result)
        return result

    def _log_pre_call(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> None:
        """Log information before LLM call."""
        tool_count = len(tools) if tools else 0
        last_assistant = self._find_last_assistant(messages)
        self._lg.debug(
            "calling LLM...",
            extra={
                "message_count": len(messages),
                "last_assistant": last_assistant,
                "tool_count": tool_count,
            },
        )

    def _log_post_call(self, result: CompletionResult) -> None:
        """Log information after LLM call."""
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
            preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            self._lg.debug("LLM content", extra={"content": preview})
        else:
            self._lg.trace("LLM response without content")

    def _find_last_assistant(self, messages: list[Message], max_len: int = 200) -> str:
        """Find and return the last assistant message content (truncated)."""
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                content = msg.content
                return content[:max_len] + "..." if len(content) > max_len else content
        return ""
