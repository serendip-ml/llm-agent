"""Agent framework with learning capabilities."""

from llm_learn.collection import ScoredFact

from llm_agent.core.agent import Agent
from llm_agent.core.config import AgentConfig
from llm_agent.core.llm import (
    CompletionResult,
    HTTPBackend,
    LLMBackend,
    LLMError,
    Message,
    StructuredOutputError,
)
from llm_agent.core.prompt_agent import PromptAgent, PromptAgentConfig
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
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)
from llm_agent.core.traits import (
    BaseTrait,
    Directive,
    DirectiveTrait,
    HTTPConfig,
    HTTPTrait,
    ToolsTrait,
    Trait,
)


__version__ = "0.0.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "BaseTool",
    "BaseTrait",
    "CompletionResult",
    "CompleteTaskTool",
    "Directive",
    "DirectiveTrait",
    "FileReadTool",
    "FileWriteTool",
    "HTTPBackend",
    "HTTPFetchTool",
    "HTTPConfig",
    "HTTPTrait",
    "LLMBackend",
    "LLMError",
    "Message",
    "PromptAgent",
    "PromptAgentConfig",
    "RecallTool",
    "RememberTool",
    "ScoredFact",
    "ShellTool",
    "StructuredOutputError",
    "Task",
    "TaskCompletion",
    "TaskResult",
    "TaskStatus",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolExecutionResult",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolsTrait",
    "Trait",
]
