"""Core agent framework components.

This package contains the core abstractions for building agents:
- agent: Base Agent class
- config: Agent configuration
- conversation: Conversation management with context window awareness
- task: Task definitions and results
- prompt_agent: YAML-configured prompt-only agents
- traits: Composable agent behaviors (LLM, HTTP, Learn, etc.)
- tools: Tool definitions and execution
- llm: LLM backend abstractions
"""

from llm_agent.core.agent import Agent
from llm_agent.core.config import AgentConfig
from llm_agent.core.conversation import (
    Compactor,
    Conversation,
    ConversationConfig,
    SlidingWindowCompactor,
    SummarizingCompactor,
)
from llm_agent.core.prompt_agent import PromptAgent, PromptAgentConfig
from llm_agent.core.task import Task, TaskResult


__all__ = [
    "Agent",
    "AgentConfig",
    # Conversation
    "Compactor",
    "Conversation",
    "ConversationConfig",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    # Prompt agents
    "PromptAgent",
    "PromptAgentConfig",
    "Task",
    "TaskResult",
]
