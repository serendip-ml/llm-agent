"""Tests for tool use infrastructure."""

import json
from unittest.mock import MagicMock

import pytest

from llm_agent import (
    BaseTool,
    CompletionResult,
    Message,
    ShellTool,
    Tool,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


pytestmark = pytest.mark.unit


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        result = ToolResult(success=True, output="hello")

        assert result.success is True
        assert result.output == "hello"
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, output="", error="something went wrong")

        assert result.success is False
        assert result.output == ""
        assert result.error == "something went wrong"


class TestToolCall:
    """Tests for ToolCall."""

    def test_tool_call(self):
        call = ToolCall(id="call-1", name="shell", arguments={"command": "ls"})

        assert call.id == "call-1"
        assert call.name == "shell"
        assert call.arguments == {"command": "ls"}

    def test_tool_call_empty_arguments(self):
        call = ToolCall(id="call-1", name="noop")

        assert call.arguments == {}


class TestBaseTool:
    """Tests for BaseTool."""

    def test_protocol_compliance(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}

            def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")

        tool = MyTool()
        assert isinstance(tool, Tool)

    def test_to_openai_function(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            }

            def execute(self, arg: str) -> ToolResult:
                return ToolResult(success=True, output=arg)

        tool = MyTool()
        func = tool.to_openai_function()

        assert func == {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
            },
        }


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ShellTool()

        registry.register(tool)

        assert registry.get("shell") is tool
        assert "shell" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(ShellTool())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ShellTool())

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(ShellTool())

        registry.unregister("shell")

        assert registry.get("shell") is None
        assert len(registry) == 0

    def test_unregister_unknown_raises(self):
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("unknown")

    def test_list_names(self):
        registry = ToolRegistry()
        registry.register(ShellTool())

        assert registry.list_names() == ["shell"]

    def test_list_tools(self):
        registry = ToolRegistry()
        tool = ShellTool()
        registry.register(tool)

        assert registry.list_tools() == [tool]

    def test_to_openai_tools(self):
        registry = ToolRegistry()
        registry.register(ShellTool())

        tools = registry.to_openai_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "shell"


class TestShellTool:
    """Tests for ShellTool."""

    def test_execute_success(self):
        tool = ShellTool()
        result = tool.execute(command="echo hello")

        assert result.success is True
        assert "hello" in result.output

    def test_execute_failure(self):
        tool = ShellTool()
        result = tool.execute(command="exit 1")

        assert result.success is False
        assert result.error is not None
        assert "exit" in result.error.lower() or "code 1" in result.error.lower()

    def test_execute_with_working_dir(self, tmp_path):
        tool = ShellTool(working_dir=str(tmp_path))
        result = tool.execute(command="pwd")

        assert result.success is True
        assert str(tmp_path) in result.output

    def test_execute_timeout(self):
        tool = ShellTool(timeout=0.1)
        result = tool.execute(command="sleep 10")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_execute_allowed_commands(self):
        tool = ShellTool(allowed_commands=["echo", "ls"])

        # Allowed command works
        result = tool.execute(command="echo hello")
        assert result.success is True

        # Disallowed command fails
        result = tool.execute(command="rm -rf /")
        assert result.success is False
        assert "not in allowed list" in result.error

    def test_output_truncation(self):
        tool = ShellTool(max_output_chars=50)
        result = tool.execute(command="echo " + "x" * 100)

        assert "truncated" in result.output
        assert len(result.output) < 150  # Some buffer for truncation message

    def test_tool_properties(self):
        tool = ShellTool()

        assert tool.name == "shell"
        assert "command" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "command" in tool.parameters["properties"]


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def registry_with_shell(self):
        registry = ToolRegistry()
        registry.register(ShellTool())
        return registry

    def test_run_no_tools_called(self, mock_llm, registry_with_shell):
        """LLM returns final response without calling tools."""
        mock_llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Here is the answer",
            model="test",
            tokens_used=10,
            latency_ms=100,
            tool_calls=None,
        )

        executor = ToolExecutor(mock_llm, registry_with_shell)
        result = executor.run(messages=[Message(role="user", content="Hello")])

        assert result.content == "Here is the answer"
        assert result.tool_calls == []
        assert result.iterations == 1

    def test_run_with_tool_call(self, mock_llm, registry_with_shell):
        """LLM calls a tool, then returns final response."""
        # First call: LLM wants to call shell tool
        mock_llm.complete.side_effect = [
            CompletionResult(
                id="resp-1",
                content="",
                model="test",
                tokens_used=10,
                latency_ms=100,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "echo hello"}),
                        },
                    }
                ],
            ),
            # Second call: LLM returns final response
            CompletionResult(
                id="resp-2",
                content="The command output was: hello",
                model="test",
                tokens_used=15,
                latency_ms=100,
                tool_calls=None,
            ),
        ]

        executor = ToolExecutor(mock_llm, registry_with_shell)
        result = executor.run(messages=[Message(role="user", content="Run echo hello")])

        assert result.content == "The command output was: hello"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[0].result.success is True
        assert "hello" in result.tool_calls[0].result.output
        assert result.iterations == 2
        assert result.total_tokens == 25

    def test_run_max_iterations_exceeded(self, mock_llm, registry_with_shell):
        """Error when max iterations exceeded."""
        # LLM keeps calling tools forever
        mock_llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="",
            model="test",
            tokens_used=10,
            latency_ms=100,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({"command": "echo loop"}),
                    },
                }
            ],
        )

        executor = ToolExecutor(mock_llm, registry_with_shell)

        with pytest.raises(RuntimeError, match="exceeded.*iterations"):
            executor.run(
                messages=[Message(role="user", content="Loop forever")],
                max_iterations=3,
            )

    def test_run_unknown_tool(self, mock_llm, registry_with_shell):
        """Handle unknown tool gracefully."""
        mock_llm.complete.side_effect = [
            CompletionResult(
                id="resp-1",
                content="",
                model="test",
                tokens_used=10,
                latency_ms=100,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "unknown_tool",
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            CompletionResult(
                id="resp-2",
                content="Tool not found",
                model="test",
                tokens_used=10,
                latency_ms=100,
                tool_calls=None,
            ),
        ]

        executor = ToolExecutor(mock_llm, registry_with_shell)
        result = executor.run(messages=[Message(role="user", content="Call unknown")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result.success is False
        assert "Unknown tool" in result.tool_calls[0].result.error

    def test_run_tool_execution_error(self, mock_llm):
        """Handle tool execution errors gracefully."""

        class FailingTool(BaseTool):
            name = "failing"
            description = "Always fails"
            parameters = {"type": "object", "properties": {}}

            def execute(self) -> ToolResult:
                raise RuntimeError("Tool exploded")

        registry = ToolRegistry()
        registry.register(FailingTool())

        mock_llm.complete.side_effect = [
            CompletionResult(
                id="resp-1",
                content="",
                model="test",
                tokens_used=10,
                latency_ms=100,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "failing", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(
                id="resp-2",
                content="Tool failed",
                model="test",
                tokens_used=10,
                latency_ms=100,
                tool_calls=None,
            ),
        ]

        executor = ToolExecutor(mock_llm, registry)
        result = executor.run(messages=[Message(role="user", content="Call failing")])

        assert result.tool_calls[0].result.success is False
        assert "Tool exploded" in result.tool_calls[0].result.error
