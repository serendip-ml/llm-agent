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
from llm_agent.core.prompt_agent import PromptOnlyAgent, PromptOnlyAgentConfig
from llm_agent.core.task import Task, TaskResult
from llm_agent.core.tools import (
    BaseTool,
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
    "Directive",
    "DirectiveTrait",
    "FileReadTool",
    "FileWriteTool",
    "HTTPBackend",
    "HTTPFetchTool",
    "HTTPConfig",
    "HTTPTrait",
    "RecallTool",
    "RememberTool",
    "LLMBackend",
    "LLMError",
    "Message",
    "PromptOnlyAgent",
    "PromptOnlyAgentConfig",
    "ScoredFact",
    "ShellTool",
    "StructuredOutputError",
    "Task",
    "TaskResult",
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
