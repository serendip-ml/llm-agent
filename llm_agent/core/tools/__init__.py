"""Tool use infrastructure for agents."""

from enum import Enum

from llm_agent.core.tools.base import BaseTool, Tool, ToolCall, ToolCallResult, ToolResult
from llm_agent.core.tools.builtin import (
    CompleteTaskTool,
    FileReadTool,
    FileWriteTool,
    HTTPFetchTool,
    RecallTool,
    RememberTool,
    ShellTool,
)
from llm_agent.core.tools.executor import SimpleToolExecutor, ToolExecutionResult
from llm_agent.core.tools.factory import ToolFactory
from llm_agent.core.tools.registry import Registry


class ToolName(str, Enum):
    """Tool name identifiers.

    Using str Enum so values work in YAML configs and as strings.
    Matches ToolFactory constants for consistency.
    """

    SHELL = "shell"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    HTTP_FETCH = "http_fetch"
    COMPLETE_TASK = "complete_task"
    REMEMBER = "remember"
    RECALL = "recall"


# All tool types available in the platform
ALL_TOOLS: list[ToolName] = [
    ToolName.SHELL,
    ToolName.READ_FILE,
    ToolName.WRITE_FILE,
    ToolName.HTTP_FETCH,
    ToolName.COMPLETE_TASK,
    # Note: REMEMBER/RECALL require LearnTrait, added dynamically when needed
]


__all__ = [
    # Base types
    "BaseTool",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolResult",
    # Names & Catalogs
    "ToolName",
    "ALL_TOOLS",
    # Factory & Registry
    "ToolFactory",
    "Registry",
    # Executor
    "SimpleToolExecutor",
    "ToolExecutionResult",
    # Built-in tools
    "CompleteTaskTool",
    "FileReadTool",
    "FileWriteTool",
    "HTTPFetchTool",
    "RecallTool",
    "RememberTool",
    "ShellTool",
]
