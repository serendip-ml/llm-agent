"""Core agent framework components.

This package contains the core abstractions for building agents:
- runnable: Runnable interface for executable agents
- agent: Base Agent class with trait composition
- config: Agent configuration
- conversation: Conversation management utilities
- factory: Factory classes for creating traits and tools
- task: Task definitions and results
- traits: Composable agent behaviors (SAIA, LLM, HTTP, Learn, etc.)
- tools: Tool definitions and registry
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
from llm_agent.core.factory import (
    ToolFactory,
    TraitFactory,
)
from llm_agent.core.runnable import ExecutionResult, Runnable
from llm_agent.core.task import Task, TaskResult


__all__ = [
    # Runnable
    "ExecutionResult",
    "Runnable",
    # Agents
    "Agent",
    "AgentConfig",
    # Factories
    "ToolFactory",
    "TraitFactory",
    # Conversation
    "Compactor",
    "Conversation",
    "ConversationConfig",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    # Tasks
    "Task",
    "TaskResult",
]
