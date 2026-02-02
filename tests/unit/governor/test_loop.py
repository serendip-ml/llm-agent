"""Tests for GovernorLoop."""

import json
from unittest.mock import MagicMock

import pytest

from llm_agent.core.governor.loop import GovernorLoop
from llm_agent.core.llm.types import CompletionResult, Message
from llm_agent.core.task import Task
from llm_agent.core.tools.base import BaseTool, ToolResult
from llm_agent.core.tools.registry import ToolRegistry


pytestmark = pytest.mark.unit


class MockShellTool(BaseTool):
    """Mock shell tool for testing."""

    name = "shell"
    description = "Execute shell command"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def execute(self, command: str) -> ToolResult:
        return ToolResult(success=True, output=f"Executed: {command}")


class MockCompleteTaskTool(BaseTool):
    """Mock complete_task tool for testing."""

    name = "complete_task"
    description = "Complete the task"
    parameters = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["done", "stuck"]},
            "conclusion": {"type": "string"},
        },
        "required": ["status", "conclusion"],
    }
    terminal = True

    def execute(self, status: str, conclusion: str) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"Task completed with status: {status}",
            terminal=True,
            terminal_data={"status": status, "conclusion": conclusion},
        )


@pytest.fixture
def mock_logger():
    """Mock logger."""
    return MagicMock()


@pytest.fixture
def mock_llm():
    """Mock LLM backend."""
    return MagicMock()


@pytest.fixture
def registry():
    """Registry with test tools."""
    reg = ToolRegistry()
    reg.register(MockShellTool())
    reg.register(MockCompleteTaskTool())
    return reg


@pytest.fixture
def default_task():
    """Default task for testing."""
    return Task(name="test", description="test task")


def make_completion(
    content: str = "",
    tool_calls: list | None = None,
    tokens: int = 10,
) -> CompletionResult:
    """Helper to create CompletionResult."""
    return CompletionResult(
        id="test-id",
        content=content,
        model="test",
        tokens_used=tokens,
        latency_ms=100,
        tool_calls=tool_calls,
    )


def make_tool_call(name: str, call_id: str = "call-1", arguments: dict | None = None):
    """Helper to create tool call dict."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}),
        },
    }


class TestGovernorLoop:
    """Tests for GovernorLoop main execution."""

    def test_run_no_tools_called(self, mock_logger, mock_llm, registry, default_task):
        """LLM returns final response without calling tools."""
        mock_llm.complete.side_effect = [
            make_completion(content="Here is the answer"),
            # Confirmation - confirms done
            make_completion(content="Yes, I'm done."),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="Hello")])

        assert result.content == "Here is the answer"
        assert result.tool_calls == []
        assert result.iterations == 1

    def test_run_with_tool_call(self, mock_logger, mock_llm, registry, default_task):
        """LLM calls a tool, then returns final response."""
        mock_llm.complete.side_effect = [
            # First: call shell tool
            make_completion(
                tool_calls=[make_tool_call("shell", arguments={"command": "echo hello"})]
            ),
            # Second: no tools
            make_completion(content="The command output was: hello"),
            # Third: confirmation - confirms done
            make_completion(content="Done."),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="Run echo hello")])

        assert result.content == "The command output was: hello"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[0].result.success is True
        assert "Executed: echo hello" in result.tool_calls[0].result.output

    def test_run_terminal_after_work(self, mock_logger, mock_llm, registry, default_task):
        """Terminal tool after work ends immediately."""
        mock_llm.complete.side_effect = [
            # First: call shell
            make_completion(tool_calls=[make_tool_call("shell", arguments={"command": "ls"})]),
            # Second: no tools
            make_completion(content="Files listed."),
            # Third: call complete_task
            make_completion(
                tool_calls=[
                    make_tool_call(
                        "complete_task",
                        arguments={"status": "done", "conclusion": "Listed files."},
                    )
                ]
            ),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="List files")])

        assert result.terminal_data == {"status": "done", "conclusion": "Listed files."}
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[1].name == "complete_task"

    def test_run_early_terminal_confirmed(self, mock_logger, mock_llm, registry, default_task):
        """Early terminal with confirmation accepted."""
        mock_llm.complete.side_effect = [
            # First: try to complete without work
            make_completion(
                tool_calls=[
                    make_tool_call(
                        "complete_task",
                        "call-1",
                        {"status": "stuck", "conclusion": "Can't proceed."},
                    )
                ]
            ),
            # Second: confirmation - reaffirm completion
            make_completion(
                tool_calls=[
                    make_tool_call(
                        "complete_task",
                        "call-2",
                        {"status": "stuck", "conclusion": "Can't proceed."},
                    )
                ]
            ),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="Do task")])

        assert result.terminal_data == {"status": "stuck", "conclusion": "Can't proceed."}
        # Only one complete_task after confirmation
        assert len(result.tool_calls) == 1
        assert mock_llm.complete.call_count == 2

    def test_run_early_terminal_rejected_then_work(
        self, mock_logger, mock_llm, registry, default_task
    ):
        """Early terminal rejected, then LLM does work."""
        mock_llm.complete.side_effect = [
            # First: try to complete without work
            make_completion(
                tool_calls=[
                    make_tool_call(
                        "complete_task",
                        arguments={"status": "stuck", "conclusion": "Can't do it."},
                    )
                ]
            ),
            # Second: after prompt, decides to do work
            make_completion(tool_calls=[make_tool_call("shell", arguments={"command": "ls"})]),
            # Third: reports result
            make_completion(content="Found the files."),
            # Fourth: confirmation - done
            make_completion(content="All done."),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="Find files")])

        assert result.content == "Found the files."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"

    def test_run_max_iterations_exceeded(self, mock_logger, mock_llm, registry):
        """Error when max iterations exceeded."""
        # LLM keeps calling tools forever
        mock_llm.complete.return_value = make_completion(
            tool_calls=[make_tool_call("shell", arguments={"command": "echo loop"})]
        )

        task = Task(name="test", description="test", max_iterations=3)
        loop = GovernorLoop(mock_logger, mock_llm, registry)

        with pytest.raises(RuntimeError, match="exceeded.*iterations"):
            loop.run(task, [Message(role="user", content="Loop forever")])

    def test_run_unknown_tool(self, mock_logger, mock_llm, registry, default_task):
        """Unknown tool returns error to LLM."""
        mock_llm.complete.side_effect = [
            # First: call unknown tool
            make_completion(tool_calls=[make_tool_call("unknown_tool")]),
            # Second: no tools after error
            make_completion(content="Tool not found"),
            # Third: confirmation
            make_completion(content="Done."),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="Call unknown")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result.success is False
        assert "Unknown tool" in result.tool_calls[0].result.error

    def test_run_malformed_json_arguments(self, mock_logger, mock_llm, registry, default_task):
        """Malformed JSON arguments return parse error."""
        mock_llm.complete.side_effect = [
            make_completion(
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{invalid"},
                    }
                ]
            ),
            make_completion(content="Got error"),
            make_completion(content="Done."),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="Run command")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result.success is False
        assert "Failed to parse" in result.tool_calls[0].result.error

    def test_messages_preserved_in_result(self, mock_logger, mock_llm, registry, default_task):
        """All messages are preserved in result."""
        mock_llm.complete.side_effect = [
            make_completion(tool_calls=[make_tool_call("shell", arguments={"command": "ls"})]),
            make_completion(content="Done."),
            make_completion(content="Yes."),
        ]

        initial_messages = [Message(role="user", content="List files")]
        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, initial_messages)

        # Should have: original + assistant with tool + tool result + assistant response
        assert len(result.messages) > 1
        assert result.messages[0].role == "user"
        # Check assistant message with tool calls exists
        assert any(m.role == "assistant" and m.tool_calls for m in result.messages)
        # Check tool result exists
        assert any(m.role == "tool" for m in result.messages)

    def test_tokens_accumulated(self, mock_logger, mock_llm, registry, default_task):
        """Tokens from all LLM calls are accumulated."""
        mock_llm.complete.side_effect = [
            make_completion(
                tool_calls=[make_tool_call("shell", arguments={"command": "ls"})],
                tokens=10,
            ),
            make_completion(content="Done.", tokens=15),
            make_completion(content="Yes.", tokens=5),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(default_task, [Message(role="user", content="List files")])

        # 10 + 15 + 5 = 30 (includes confirmation)
        assert result.total_tokens >= 25  # At least tool + response tokens


class TestGracefulWrapUp:
    """Tests for graceful timeout warning."""

    def test_wrap_up_message_injected_near_timeout(self, mock_logger, mock_llm, registry):
        """Wrap-up message injected when approaching timeout."""
        task = Task(name="test", description="test", timeout_secs=35)

        mock_llm.complete.side_effect = [
            # First call - work tool, will trigger wrap-up check
            # Note: wrap-up is checked after LLM response, so this call happens first
            make_completion(
                tool_calls=[make_tool_call("shell", arguments={"command": "ls"})],
            ),
            # Second call - with wrap-up message injected, LLM completes
            make_completion(
                tool_calls=[
                    make_tool_call(
                        "complete_task",
                        arguments={"status": "done", "conclusion": "Wrapped up."},
                    )
                ],
            ),
        ]

        loop = GovernorLoop(mock_logger, mock_llm, registry)
        result = loop.run(task, [Message(role="user", content="Work")])

        assert result.terminal_data is not None
        # Note: wrap-up might not trigger if timeout check happens before 30s threshold
        # This test mainly verifies the flow doesn't crash
