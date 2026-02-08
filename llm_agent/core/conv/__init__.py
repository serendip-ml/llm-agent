"""Conversation management for agents.

Conversation is a fundamental concept - it's the medium through which agents
learn and build understanding over time. An agent that forgets everything
between interactions can't truly learn from experience.

This module provides:
- Conversation: Core class for managing message history with token awareness
- Compactors: Strategies for reducing context when approaching token limits
- Token utilities: Fast estimation without tokenizer dependencies

Example:
    from llm_agent.core.conv import Conversation, ConversationConfig

    config = ConversationConfig(max_tokens=32000, compact_threshold=0.8)
    conversation = Conversation(config=config)

    conversation.add_system("You are a helpful assistant.")
    conversation.add_user("Hello!")
    conversation.add_assistant("Hi there!")

    if conversation.needs_compaction():
        compactor = SlidingWindowCompactor()
        compactor.compact(conversation)

    # Get messages for LLM call
    messages = conversation.messages()
"""

from llm_agent.core.conv.compactor import (
    Compactor,
    SlidingWindowCompactor,
    SummarizingCompactor,
)
from llm_agent.core.conv.conversation import Conversation, ConversationConfig
from llm_agent.core.conv.runner import ConversationRunner
from llm_agent.core.conv.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    estimate_message_tokens,
    estimate_tokens,
)


__all__ = [
    # Core
    "Conversation",
    "ConversationConfig",
    "ConversationRunner",
    # Compactors
    "Compactor",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    # Token utilities
    "estimate_tokens",
    "estimate_message_tokens",
    "DEFAULT_CHARS_PER_TOKEN",
]
