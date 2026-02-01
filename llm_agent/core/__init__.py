"""Core agent framework components.

This package contains the core abstractions for building agents:
- agent: Base Agent class
- conversational: ConversationalAgent with LLM and tool capabilities
- config: Agent configuration
- conversation: Conversation management with context window awareness
- factory: Factory classes for creating agents, traits, and tools
- task: Task definitions and results
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
from llm_agent.core.conversational import ConversationalAgent
from llm_agent.core.factory import (
    AgentFactory,
    ToolFactory,
    TraitFactory,
    create_agent_from_config,
)
from llm_agent.core.task import Task, TaskResult


__all__ = [
    # Agents
    "Agent",
    "AgentConfig",
    "ConversationalAgent",
    # Factories
    "AgentFactory",
    "ToolFactory",
    "TraitFactory",
    "create_agent_from_config",
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
