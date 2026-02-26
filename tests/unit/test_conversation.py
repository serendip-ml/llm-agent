"""Tests for conversation management."""

from __future__ import annotations

from llm_gent.core.conv import (
    Conversation,
    ConversationConfig,
    SlidingWindowCompactor,
    SummarizingCompactor,
    estimate_message_tokens,
    estimate_tokens,
)
from llm_gent.core.llm import Message


class TestTokenEstimation:
    """Tests for token estimation utilities."""

    def test_estimate_tokens_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self) -> None:
        # "hello" = 5 chars, ~1-2 tokens
        result = estimate_tokens("hello")
        assert result >= 1

    def test_estimate_tokens_longer(self) -> None:
        # 100 chars should be ~25 tokens
        text = "a" * 100
        result = estimate_tokens(text)
        assert 20 <= result <= 30

    def test_estimate_message_tokens_includes_overhead(self) -> None:
        content = "hello"
        content_only = estimate_tokens(content)
        with_overhead = estimate_message_tokens("user", content)
        # Should have overhead added
        assert with_overhead > content_only


class TestConversation:
    """Tests for Conversation class."""

    def test_add_messages(self) -> None:
        conv = Conversation()
        conv.add_system("You are helpful.")
        conv.add_user("Hello")
        conv.add_assistant("Hi there!")

        assert conv.message_count == 3
        messages = conv.messages()
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_add_tool_message(self) -> None:
        conv = Conversation()
        conv.add_tool("result data", tool_call_id="call_123")

        messages = conv.messages()
        assert len(messages) == 1
        assert messages[0].role == "tool"
        assert messages[0].tool_call_id == "call_123"

    def test_token_count_increases(self) -> None:
        conv = Conversation()
        assert conv.token_count == 0

        conv.add_user("Hello world")
        count_after_one = conv.token_count
        assert count_after_one > 0

        conv.add_assistant("Hi there, how can I help?")
        assert conv.token_count > count_after_one

    def test_usage_ratio(self) -> None:
        config = ConversationConfig(max_tokens=100)
        conv = Conversation(config=config)

        assert conv.usage_ratio == 0.0

        # Add some content (rough token estimate)
        conv.add_user("x" * 50)  # ~12-15 tokens
        assert 0.0 < conv.usage_ratio < 1.0

    def test_needs_compaction(self) -> None:
        config = ConversationConfig(max_tokens=100, compact_threshold=0.5)
        conv = Conversation(config=config)

        assert not conv.needs_compaction()

        # Add enough to exceed threshold
        conv.add_user("x" * 200)  # Should exceed 50% of 100 tokens
        assert conv.needs_compaction()

    def test_clear(self) -> None:
        conv = Conversation()
        conv.add_user("Hello")
        conv.add_assistant("Hi")

        conv.clear()
        assert conv.message_count == 0
        assert conv.token_count == 0

    def test_get_system_message(self) -> None:
        conv = Conversation()
        conv.add_system("Be helpful")
        conv.add_user("Hello")

        system = conv.get_system_message()
        assert system is not None
        assert system.content == "Be helpful"

    def test_get_system_message_none(self) -> None:
        conv = Conversation()
        conv.add_user("Hello")

        assert conv.get_system_message() is None

    def test_split_for_compaction(self) -> None:
        config = ConversationConfig(min_recent_messages=2, preserve_system=True)
        conv = Conversation(config=config)

        conv.add_system("System prompt")
        conv.add_user("Old message 1")
        conv.add_assistant("Old response 1")
        conv.add_user("Recent message")
        conv.add_assistant("Recent response")

        to_compact, preserved = conv.split_for_compaction()

        # Should compact old messages
        assert len(to_compact) == 2
        assert to_compact[0].content == "Old message 1"
        assert to_compact[1].content == "Old response 1"

        # Should preserve system + recent
        assert len(preserved) == 3
        assert preserved[0].role == "system"
        assert preserved[1].content == "Recent message"
        assert preserved[2].content == "Recent response"

    def test_split_not_enough_to_compact(self) -> None:
        config = ConversationConfig(min_recent_messages=4)
        conv = Conversation(config=config)

        conv.add_user("Message 1")
        conv.add_assistant("Response 1")

        to_compact, preserved = conv.split_for_compaction()

        # Not enough messages to compact
        assert len(to_compact) == 0
        assert len(preserved) == 2

    def test_replace_messages(self) -> None:
        conv = Conversation()
        conv.add_user("Original")
        original_tokens = conv.token_count

        new_messages = [
            Message(role="system", content="New system"),
            Message(role="user", content="New user message"),
        ]
        conv.replace_messages(new_messages)

        assert conv.message_count == 2
        assert conv.messages()[0].content == "New system"
        # Token count should be recalculated
        assert conv.token_count != original_tokens

    def test_messages_returns_copy(self) -> None:
        conv = Conversation()
        conv.add_user("Hello")

        messages = conv.messages()
        messages.append(Message(role="user", content="Modified"))

        # Original should be unchanged
        assert conv.message_count == 1


class TestSlidingWindowCompactor:
    """Tests for SlidingWindowCompactor."""

    def test_compact_drops_old_messages(self) -> None:
        config = ConversationConfig(min_recent_messages=2, preserve_system=True)
        conv = Conversation(config=config)

        conv.add_system("System")
        conv.add_user("Old 1")
        conv.add_assistant("Old 2")
        conv.add_user("Recent 1")
        conv.add_assistant("Recent 2")

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        messages = conv.messages()
        # Should have: system + 2 recent
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[1].content == "Recent 1"
        assert messages[2].content == "Recent 2"

    def test_compact_nothing_to_compact(self) -> None:
        config = ConversationConfig(min_recent_messages=4)
        conv = Conversation(config=config)

        conv.add_user("Only message")

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        # Should be unchanged
        assert conv.message_count == 1


class TestSummarizingCompactor:
    """Tests for SummarizingCompactor."""

    def test_compact_creates_summary(self) -> None:
        # Mock LLM backend
        class MockLLM:
            def complete(self, messages, model=None, temperature=0.3):
                class Result:
                    content = "Summary of conversation"

                return Result()

        config = ConversationConfig(min_recent_messages=1, preserve_system=True)
        conv = Conversation(config=config)

        conv.add_system("System prompt")
        conv.add_user("Old message 1")
        conv.add_assistant("Old response 1")
        conv.add_user("Recent message")

        compactor = SummarizingCompactor(llm=MockLLM())
        compactor.compact(conv)

        messages = conv.messages()

        # Should have: system, summary, recent
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "System prompt"
        assert messages[1].role == "user"
        assert "[Previous conversation summary]" in messages[1].content
        assert "Summary of conversation" in messages[1].content
        assert messages[2].content == "Recent message"

    def test_compact_nothing_to_compact(self) -> None:
        class MockLLM:
            def complete(self, messages, model=None, temperature=0.3):
                raise AssertionError("Should not be called")

        config = ConversationConfig(min_recent_messages=4)
        conv = Conversation(config=config)

        conv.add_user("Only message")

        compactor = SummarizingCompactor(llm=MockLLM())
        compactor.compact(conv)

        # Should be unchanged, LLM not called
        assert conv.message_count == 1
