"""Agent framework with learning capabilities."""

from llm_learn.collection import ScoredFact

from llm_agent.core.agent import Agent
from llm_agent.core.config import AgentConfig
from llm_agent.core.conversational import ConversationalAgent
from llm_agent.core.factory import (
    AgentFactory,
    ToolFactory,
    TraitFactory,
    create_agent_from_config,
)
from llm_agent.core.llm import (
    CompletionResult,
    HTTPBackend,
    LLMBackend,
    LLMError,
    Message,
    StructuredOutputError,
)
from llm_agent.core.task import Task, TaskCompletion, TaskResult, TaskStatus
from llm_agent.core.tools import (
    BaseTool,
    CompleteTaskTool,
    FileReadTool,
    FileWriteTool,
    HTTPFetchTool,
    RecallTool,
    RememberTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolCallResult,
    ToolExecutionResult,
    ToolRegistry,
    ToolResult,
)
from llm_agent.core.traits import (
    BaseTrait,
    HTTPConfig,
    HTTPTrait,
    Identity,
    IdentityTrait,
    MethodTrait,
    ToolsTrait,
    Trait,
)


__version__ = "0.0.0"

__all__ = [
    # Agents
    "Agent",
    "AgentConfig",
    "ConversationalAgent",
    "create_agent_from_config",
    # Factories
    "AgentFactory",
    "ToolFactory",
    "TraitFactory",
    # Tools
    "BaseTool",
    "CompleteTaskTool",
    "FileReadTool",
    "FileWriteTool",
    "HTTPFetchTool",
    "RecallTool",
    "RememberTool",
    "ShellTool",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolExecutionResult",
    "ToolRegistry",
    "ToolResult",
    # Traits
    "BaseTrait",
    "Identity",
    "IdentityTrait",
    "MethodTrait",
    "Trait",
    # HTTP
    "HTTPConfig",
    "HTTPTrait",
    # LLM
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMError",
    "Message",
    "StructuredOutputError",
    # Memory
    "ScoredFact",
    # Tasks
    "Task",
    "TaskCompletion",
    "TaskResult",
    "TaskStatus",
    # Tools trait
    "ToolsTrait",
]
