"""Agent framework with learning capabilities."""

from llm_learn.collection import ScoredFact

from llm_agent.agent import Agent
from llm_agent.config import AgentConfig
from llm_agent.llm import CompletionResult, HTTPBackend, LLMBackend, LLMError, Message
from llm_agent.tools import (
    BaseTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolCallResult,
    ToolExecutionResult,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)
from llm_agent.traits import BaseTrait, Directive, DirectiveTrait, HTTPConfig, HTTPTrait, Trait


__version__ = "0.0.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "BaseTool",
    "BaseTrait",
    "CompletionResult",
    "Directive",
    "DirectiveTrait",
    "HTTPBackend",
    "HTTPConfig",
    "HTTPTrait",
    "LLMBackend",
    "LLMError",
    "Message",
    "ScoredFact",
    "ShellTool",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolExecutionResult",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "Trait",
]
