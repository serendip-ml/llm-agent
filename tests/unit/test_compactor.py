"""Tests for conversation compactors."""

from unittest.mock import MagicMock

import pytest

from llm_agent.core.conv.compactor import SlidingWindowCompactor, SummarizingCompactor
from llm_agent.core.llm import Message


pytestmark = pytest.mark.unit


class TestSlidingWindowCompactor:
    """Tests for SlidingWindowCompactor."""

    @pytest.fixture
    def mock_conversation(self):
        """Create a mock conversation."""
        return MagicMock()

    def test_compact_drops_old_messages(self, mock_conversation):
        """compact() drops old messages and keeps recent ones."""
        # Setup split_for_compaction to return old and new messages
        old_messages = [
            Message(role="user", content="old message 1"),
            Message(role="assistant", content="old response 1"),
        ]
        preserved_messages = [
            Message(role="user", content="recent message"),
            Message(role="assistant", content="recent response"),
        ]
        mock_conversation.split_for_compaction.return_value = (old_messages, preserved_messages)

        compactor = SlidingWindowCompactor()
        compactor.compact(mock_conversation)

        # Verify split was called
        mock_conversation.split_for_compaction.assert_called_once()

        # Verify replace_messages was called with preserved messages only
        mock_conversation.replace_messages.assert_called_once_with(preserved_messages)

    def test_compact_handles_empty_old_messages(self, mock_conversation):
        """compact() works when there are no old messages to drop."""
        preserved_messages = [
            Message(role="user", content="only message"),
        ]
        mock_conversation.split_for_compaction.return_value = ([], preserved_messages)

        compactor = SlidingWindowCompactor()
        compactor.compact(mock_conversation)

        # Should still replace with preserved messages
        mock_conversation.replace_messages.assert_called_once_with(preserved_messages)


class TestSummarizingCompactor:
    """Tests for SummarizingCompactor."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM backend."""
        llm = MagicMock()
        # Setup default complete response
        mock_result = MagicMock()
        mock_result.content = "Summary of conversation"
        llm.complete.return_value = mock_result
        return llm

    @pytest.fixture
    def mock_conversation(self):
        """Create a mock conversation."""
        return MagicMock()

    def test_init_default_prompt(self, mock_llm):
        """Compactor initializes with default summary prompt."""
        compactor = SummarizingCompactor(mock_llm)

        assert compactor._llm == mock_llm
        assert compactor._model is None
        assert "Summarize the following conversation" in compactor._summary_prompt

    def test_init_custom_prompt(self, mock_llm):
        """Compactor accepts custom summary prompt."""
        custom_prompt = "Custom summarization instruction"
        compactor = SummarizingCompactor(mock_llm, summary_prompt=custom_prompt)

        assert compactor._summary_prompt == custom_prompt

    def test_init_custom_model(self, mock_llm):
        """Compactor accepts custom model."""
        compactor = SummarizingCompactor(mock_llm, model="gpt-4")

        assert compactor._model == "gpt-4"

    def test_compact_no_messages_to_compact(self, mock_llm, mock_conversation):
        """compact() returns early when there are no messages to compact."""
        preserved_messages = [
            Message(role="user", content="recent message"),
        ]
        mock_conversation.split_for_compaction.return_value = ([], preserved_messages)

        compactor = SummarizingCompactor(mock_llm)
        compactor.compact(mock_conversation)

        # Should not call LLM or replace messages when nothing to compact
        mock_llm.complete.assert_not_called()
        mock_conversation.replace_messages.assert_not_called()

    def test_compact_summarizes_old_messages(self, mock_llm, mock_conversation):
        """compact() summarizes old messages and replaces conversation."""
        old_messages = [
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
        ]
        preserved_messages = [
            Message(role="user", content="recent question"),
            Message(role="assistant", content="recent answer"),
        ]
        mock_conversation.split_for_compaction.return_value = (old_messages, preserved_messages)

        mock_result = MagicMock()
        mock_result.content = "Summary: discussed old topic"
        mock_llm.complete.return_value = mock_result

        compactor = SummarizingCompactor(mock_llm)
        compactor.compact(mock_conversation)

        # Verify LLM was called to generate summary
        mock_llm.complete.assert_called_once()
        call_args = mock_llm.complete.call_args
        assert call_args.kwargs["model"] is None
        assert call_args.kwargs["temperature"] == 0.3

        # Verify messages structure in LLM call
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert "Summarize" in messages[0].content
        assert messages[1].role == "user"
        assert "old question" in messages[1].content
        assert "old answer" in messages[1].content

        # Verify conversation was replaced
        mock_conversation.replace_messages.assert_called_once()
        new_messages = mock_conversation.replace_messages.call_args[0][0]

        # Should have summary message + preserved messages
        assert len(new_messages) == 3  # summary + 2 preserved
        assert "[Previous conversation summary]" in new_messages[0].content
        assert "Summary: discussed old topic" in new_messages[0].content
        assert new_messages[1] == preserved_messages[0]
        assert new_messages[2] == preserved_messages[1]

    def test_compact_preserves_system_message(self, mock_llm, mock_conversation):
        """compact() preserves and repositions system message."""
        old_messages = [
            Message(role="user", content="old question"),
        ]
        preserved_messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="recent question"),
        ]
        mock_conversation.split_for_compaction.return_value = (old_messages, preserved_messages)

        mock_result = MagicMock()
        mock_result.content = "Summary text"
        mock_llm.complete.return_value = mock_result

        compactor = SummarizingCompactor(mock_llm)
        compactor.compact(mock_conversation)

        # Verify conversation structure
        new_messages = mock_conversation.replace_messages.call_args[0][0]

        # System message should be first, then summary, then other preserved
        assert new_messages[0].role == "system"
        assert new_messages[0].content == "You are helpful"
        assert new_messages[1].role == "user"
        assert "[Previous conversation summary]" in new_messages[1].content
        assert new_messages[2].role == "user"
        assert new_messages[2].content == "recent question"

    def test_compact_uses_custom_model(self, mock_llm, mock_conversation):
        """compact() uses custom model if specified."""
        old_messages = [Message(role="user", content="old")]
        preserved_messages = [Message(role="user", content="new")]
        mock_conversation.split_for_compaction.return_value = (old_messages, preserved_messages)

        mock_result = MagicMock()
        mock_result.content = "Summary"
        mock_llm.complete.return_value = mock_result

        compactor = SummarizingCompactor(mock_llm, model="gpt-4-turbo")
        compactor.compact(mock_conversation)

        # Verify custom model was used
        call_args = mock_llm.complete.call_args
        assert call_args.kwargs["model"] == "gpt-4-turbo"

    def test_format_messages_for_summary(self, mock_llm):
        """_format_messages_for_summary formats messages correctly."""
        compactor = SummarizingCompactor(mock_llm)

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="tool", content="Result: 42"),
        ]

        formatted = compactor._format_messages_for_summary(messages)

        assert "USER: Hello" in formatted
        assert "ASSISTANT: Hi there" in formatted
        assert "TOOL RESULT: Result: 42" in formatted
        # Check double newline separation
        assert "\n\n" in formatted

    def test_format_messages_handles_empty_list(self, mock_llm):
        """_format_messages_for_summary handles empty message list."""
        compactor = SummarizingCompactor(mock_llm)

        formatted = compactor._format_messages_for_summary([])

        assert formatted == ""

    def test_compact_multiple_system_messages(self, mock_llm, mock_conversation):
        """compact() handles case with system message in old and preserved."""
        old_messages = [
            Message(role="system", content="Old system"),
            Message(role="user", content="old"),
        ]
        preserved_messages = [
            Message(role="system", content="New system"),
            Message(role="user", content="new"),
        ]
        mock_conversation.split_for_compaction.return_value = (old_messages, preserved_messages)

        mock_result = MagicMock()
        mock_result.content = "Summary"
        mock_llm.complete.return_value = mock_result

        compactor = SummarizingCompactor(mock_llm)
        compactor.compact(mock_conversation)

        # Verify only preserved system message is kept (first in new messages)
        new_messages = mock_conversation.replace_messages.call_args[0][0]
        system_messages = [m for m in new_messages if m.role == "system"]

        # Should only have the system message from preserved
        assert len(system_messages) == 1
        assert system_messages[0].content == "New system"
