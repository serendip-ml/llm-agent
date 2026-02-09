"""Agent framework with learning capabilities."""

from llm_learn.core.types import ScoredEntity
from llm_learn.memory.atomic import Fact

from llm_agent.core.agent import Agent, Config, Identity
from llm_agent.core.errors import AgentError, ConfigError
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
    Registry,
    RememberTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolCallResult,
    ToolExecutionResult,
    ToolResult,
)
from llm_agent.core.tools.factory import ToolFactory
from llm_agent.core.traits import (
    BaseTrait,
    Directive,
    DirectiveTrait,
    HTTPConfig,
    HTTPTrait,
    LearnConfig,
    LearnTrait,
    LLMConfig,
    LLMTrait,
    MethodTrait,
    SAIAConfig,
    SAIATrait,
    ToolsTrait,
    Trait,
)
from llm_agent.core.traits.factory import Factory as TraitFactory


__version__ = "0.0.0"

__all__ = [
    # Agents
    "Agent",
    "Config",
    "Identity",
    # Errors
    "AgentError",
    "ConfigError",
    # Factories
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
    "Registry",
    "ToolResult",
    # Traits
    "BaseTrait",
    "Directive",
    "DirectiveTrait",
    "LearnConfig",
    "LearnTrait",
    "LLMConfig",
    "LLMTrait",
    "MethodTrait",
    "SAIAConfig",
    "SAIATrait",
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
    "Fact",
    "ScoredEntity",
    # Tasks
    "Task",
    "TaskCompletion",
    "TaskResult",
    "TaskStatus",
    # Tools trait
    "ToolsTrait",
]
