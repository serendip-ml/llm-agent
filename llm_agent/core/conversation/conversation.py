"""Core conversation management for agents.

Conversation is a fundamental concept - it's the medium through which agents
learn and build understanding over time. Without persistent conversation,
an agent can't learn from experience.

Both Class 1 (custom Python) and Class 2 (YAML-configured) agents use
Conversation to maintain context across interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_agent.core.conversation.tokens import estimate_message_tokens
from llm_agent.core.llm import Message


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""

    max_tokens: int = 32000
    """Maximum tokens before compaction is required."""

    compact_threshold: float = 0.8
    """Trigger compaction when usage exceeds this fraction (0.0-1.0)."""

    preserve_system: bool = True
    """Always preserve the system message during compaction."""

    min_recent_messages: int = 4
    """Minimum recent messages to preserve during compaction."""


@dataclass
class Conversation:
    """Manages conversation history with context window awareness.

    Conversation tracks messages, estimates token usage, and signals when
    compaction is needed. It does not perform compaction itself - that's
    delegated to a Compactor.

    Example:
        conversation = Conversation(config=ConversationConfig(max_tokens=32000))
        conversation.add_system("You are a helpful assistant.")
        conversation.add_user("Hello!")
        conversation.add_assistant("Hi there!")

        if conversation.needs_compaction():
            # Use a compactor to summarize older messages
            compactor.compact(conversation)

        messages = conversation.messages()  # For LLM call
    """

    config: ConversationConfig = field(default_factory=ConversationConfig)
    """Conversation configuration."""

    _messages: list[Message] = field(default_factory=list, repr=False)
    """Message history."""

    _token_count: int = field(default=0, repr=False)
    """Estimated total tokens in conversation."""

    def add(self, message: Message) -> None:
        """Add a message to the conversation.

        Args:
            message: Message to add.
        """
        tokens = estimate_message_tokens(message.role, message.content)
        self._messages.append(message)
        self._token_count += tokens

    def add_system(self, content: str) -> None:
        """Add a system message."""
        self.add(Message(role="system", content=content))

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.add(Message(role="user", content=content))

    def add_assistant(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        """Add an assistant message.

        Args:
            content: Message content.
            tool_calls: Optional tool calls made by assistant.
        """
        self.add(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool(self, content: str, tool_call_id: str) -> None:
        """Add a tool result message.

        Args:
            content: Tool output.
            tool_call_id: ID of the tool call this responds to.
        """
        self.add(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def messages(self) -> list[Message]:
        """Get all messages for LLM call.

        Returns:
            Copy of message list.
        """
        return list(self._messages)

    @property
    def message_count(self) -> int:
        """Number of messages in conversation."""
        return len(self._messages)

    @property
    def token_count(self) -> int:
        """Estimated token count."""
        return self._token_count

    @property
    def token_limit(self) -> int:
        """Maximum token limit from config."""
        return self.config.max_tokens

    @property
    def usage_ratio(self) -> float:
        """Current usage as fraction of limit (0.0-1.0)."""
        if self.config.max_tokens <= 0:
            return 0.0
        return self._token_count / self.config.max_tokens

    def needs_compaction(self) -> bool:
        """Check if conversation needs compaction.

        Returns:
            True if token usage exceeds compact_threshold.
        """
        return self.usage_ratio >= self.config.compact_threshold

    def clear(self) -> None:
        """Clear all messages and reset token count."""
        self._messages.clear()
        self._token_count = 0

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace all messages (used by compactors).

        Args:
            messages: New message list.
        """
        self._messages = list(messages)
        self._token_count = sum(estimate_message_tokens(m.role, m.content) for m in self._messages)

    def get_system_message(self) -> Message | None:
        """Get the system message if present.

        Returns:
            System message or None.
        """
        for msg in self._messages:
            if msg.role == "system":
                return msg
        return None

    def split_for_compaction(self) -> tuple[list[Message], list[Message]]:
        """Split messages into compactable and preserved portions.

        Preserved messages include:
        - System message (if preserve_system is True)
        - Last min_recent_messages messages

        Returns:
            Tuple of (messages_to_compact, messages_to_preserve).
        """
        preserve_count = self.config.min_recent_messages
        system_msg = self.get_system_message() if self.config.preserve_system else None

        # Messages excluding system (if we're preserving it separately)
        non_system = [m for m in self._messages if m.role != "system"]

        if len(non_system) <= preserve_count:
            # Not enough messages to compact
            return [], list(self._messages)

        to_compact = non_system[:-preserve_count]
        to_preserve = non_system[-preserve_count:]

        # Add system message back to preserve list if applicable
        if system_msg is not None:
            to_preserve = [system_msg] + to_preserve

        return to_compact, to_preserve

    def __len__(self) -> int:
        """Number of messages."""
        return len(self._messages)
