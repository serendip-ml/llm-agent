"""Conversation compaction strategies.

When a conversation approaches its token limit, compactors reduce the
context while preserving important information. Different strategies
trade off between information loss and token savings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_agent.core.llm import Message


if TYPE_CHECKING:
    from llm_agent.core.conv.conversation import Conversation
    from llm_agent.core.llm import LLMBackend


class Compactor(ABC):
    """Base class for conversation compaction strategies."""

    @abstractmethod
    def compact(self, conversation: Conversation) -> None:
        """Compact the conversation in-place.

        Args:
            conversation: Conversation to compact.
        """
        ...


class SlidingWindowCompactor(Compactor):
    """Simple compactor that drops oldest messages.

    Fast and predictable, but loses information completely.
    Best for conversations where older context is less important.
    """

    def compact(self, conversation: Conversation) -> None:
        """Drop old messages, keeping only recent ones.

        Uses conversation.split_for_compaction() to determine
        which messages to preserve.
        """
        _, preserved = conversation.split_for_compaction()
        conversation.replace_messages(preserved)


class SummarizingCompactor(Compactor):
    """Compactor that summarizes older messages using an LLM.

    Preserves information by creating a summary of compacted messages,
    which is inserted as a system message or user context.

    More expensive (requires LLM call) but retains context better.
    """

    def __init__(
        self,
        llm: LLMBackend,
        model: str | None = None,
        summary_prompt: str | None = None,
    ) -> None:
        """Initialize summarizing compactor.

        Args:
            llm: LLM backend for generating summaries.
            model: Model to use (optional, uses backend default).
            summary_prompt: Custom prompt for summarization.
        """
        self._llm = llm
        self._model = model
        self._summary_prompt = summary_prompt or self._default_summary_prompt()

    def _default_summary_prompt(self) -> str:
        return (
            "Summarize the following conversation concisely, preserving key information, "
            "decisions made, and important context. Focus on facts and outcomes, not "
            "conversational filler."
        )

    def compact(self, conversation: Conversation) -> None:
        """Summarize old messages and replace with summary.

        Keeps system message and recent messages, summarizes the rest.
        """
        to_compact, preserved = conversation.split_for_compaction()

        if not to_compact:
            # Nothing to compact
            return

        # Generate summary of old messages
        summary = self._summarize(to_compact)

        # Build new message list
        new_messages: list[Message] = []

        # Find and add system message first (if present)
        system_msg = next((m for m in preserved if m.role == "system"), None)
        if system_msg is not None:
            new_messages.append(system_msg)
            preserved = [m for m in preserved if m.role != "system"]

        # Add summary as context
        summary_msg = Message(
            role="user",
            content=f"[Previous conversation summary]\n{summary}\n[End summary]",
        )
        new_messages.append(summary_msg)

        # Add preserved messages
        new_messages.extend(preserved)

        conversation.replace_messages(new_messages)

    def _summarize(self, messages: list[Message]) -> str:
        """Generate summary of messages using LLM.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary text.
        """
        # Format messages for summarization
        formatted = self._format_messages_for_summary(messages)

        summary_messages = [
            Message(role="system", content=self._summary_prompt),
            Message(role="user", content=formatted),
        ]

        result = self._llm.complete(
            messages=summary_messages,
            model=self._model,
            temperature=0.3,  # Low temperature for consistent summaries
        )

        return result.content

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        """Format messages as text for summarization."""
        lines = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content
            if msg.role == "tool":
                role = "TOOL RESULT"
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)
