"""Tests for tool use infrastructure."""

import json
from unittest.mock import MagicMock

import pytest

from llm_agent import (
    BaseTool,
    CompletionResult,
    FileReadTool,
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

    def test_empty_command_with_allowlist(self):
        """Empty or whitespace-only commands return clear error."""
        tool = ShellTool(allowed_commands=["echo", "ls"])

        # Whitespace-only command gets clear error (empty string caught earlier)
        result = tool.execute(command="   ")
        assert result.success is False
        assert "Empty command" in result.error

    def test_shell_metacharacters_blocked_with_allowlist(self):
        """Shell metacharacters are blocked when allowlist is enabled to prevent bypass."""
        tool = ShellTool(allowed_commands=["echo", "ls"])

        # Command chaining with semicolon
        result = tool.execute(command="echo hello; rm -rf /")
        assert result.success is False
        assert "metacharacters" in result.error

        # Command chaining with &&
        result = tool.execute(command="echo hello && cat /etc/passwd")
        assert result.success is False
        assert "metacharacters" in result.error

        # Pipe
        result = tool.execute(command="ls | grep secret")
        assert result.success is False
        assert "metacharacters" in result.error

        # Command substitution
        result = tool.execute(command="echo $(whoami)")
        assert result.success is False
        assert "metacharacters" in result.error

        # Backticks
        result = tool.execute(command="echo `id`")
        assert result.success is False
        assert "metacharacters" in result.error

        # Variable expansion with braces ${VAR}
        result = tool.execute(command="echo ${PATH}")
        assert result.success is False
        assert "metacharacters" in result.error

        # Variable reference $VAR
        result = tool.execute(command="echo $HOME")
        assert result.success is False
        assert "metacharacters" in result.error

        # Output redirection
        result = tool.execute(command="echo x > /tmp/test")
        assert result.success is False
        assert "metacharacters" in result.error

        # Append redirection
        result = tool.execute(command="echo x >> /tmp/test")
        assert result.success is False
        assert "metacharacters" in result.error

        # Input redirection
        result = tool.execute(command="cat < /etc/passwd")
        assert result.success is False
        assert "metacharacters" in result.error

        # Heredoc
        result = tool.execute(command="cat << EOF")
        assert result.success is False
        assert "metacharacters" in result.error

    def test_invalid_command_type(self):
        """Non-string command argument is rejected."""
        tool = ShellTool()

        # Integer instead of string
        result = tool.execute(command=123)
        assert result.success is False
        assert "invalid" in result.error.lower()

        # None
        result = tool.execute(command=None)
        assert result.success is False
        assert "invalid" in result.error.lower()

        # List
        result = tool.execute(command=["echo", "hello"])
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_shell_metacharacters_allowed_without_allowlist(self):
        """Shell metacharacters work normally when no allowlist is configured."""
        tool = ShellTool()

        # Pipe should work
        result = tool.execute(command="echo hello | cat")
        assert result.success is True
        assert "hello" in result.output

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


class TestFileReadTool:
    """Tests for FileReadTool."""

    def test_read_file_success(self, tmp_path):
        """Read a simple file successfully."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt")

        assert result.success is True
        assert "line 1" in result.output
        assert "line 2" in result.output
        assert "line 3" in result.output

    def test_read_file_with_line_numbers(self, tmp_path):
        """Output includes line numbers."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("first\nsecond\nthird")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt")

        assert result.success is True
        assert "1│" in result.output or "1|" in result.output
        assert "2│" in result.output or "2|" in result.output
        assert "3│" in result.output or "3|" in result.output

    def test_read_file_absolute_path(self, tmp_path):
        """Read file using absolute path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content here")

        tool = FileReadTool()
        result = tool.execute(path=str(test_file))

        assert result.success is True
        assert "content here" in result.output

    def test_read_file_not_found(self, tmp_path):
        """Return error when file doesn't exist."""
        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="nonexistent.txt")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_directory_fails(self, tmp_path):
        """Return error when path is a directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="subdir")

        assert result.success is False
        assert "not a file" in result.error.lower()

    def test_read_with_offset(self, tmp_path):
        """Read from specific line offset."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", offset=3)

        assert result.success is True
        assert "line 3" in result.output
        assert "line 4" in result.output
        assert "line 5" in result.output
        assert "line 1" not in result.output
        assert "line 2" not in result.output

    def test_read_with_limit(self, tmp_path):
        """Read limited number of lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", limit=2)

        assert result.success is True
        assert "line 1" in result.output
        assert "line 2" in result.output
        assert "line 3" not in result.output

    def test_read_with_offset_and_limit(self, tmp_path):
        """Read specific range of lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", offset=2, limit=2)

        assert result.success is True
        assert "line 2" in result.output
        assert "line 3" in result.output
        assert "line 1" not in result.output
        assert "line 4" not in result.output

    def test_read_offset_past_end(self, tmp_path):
        """Handle offset past end of file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", offset=100)

        assert result.success is True
        assert "past end" in result.output.lower() or "2 lines" in result.output.lower()

    def test_read_empty_file(self, tmp_path):
        """Handle empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="empty.txt")

        assert result.success is True
        assert "Empty file" in result.output

    def test_max_lines_truncation(self, tmp_path):
        """Truncate output when exceeding max_lines."""
        test_file = tmp_path / "big.txt"
        test_file.write_text("\n".join(f"line {i}" for i in range(100)))

        tool = FileReadTool(working_dir=str(tmp_path), max_lines=10)
        result = tool.execute(path="big.txt")

        assert result.success is True
        assert "line 0" in result.output
        assert "line 9" in result.output
        assert "more lines" in result.output.lower()

    def test_max_chars_truncation(self, tmp_path):
        """Truncate output when exceeding max_chars."""
        test_file = tmp_path / "big.txt"
        test_file.write_text("x" * 1000)

        tool = FileReadTool(working_dir=str(tmp_path), max_chars=100)
        result = tool.execute(path="big.txt")

        assert result.success is True
        assert "truncated" in result.output.lower()

    def test_allowed_paths_success(self, tmp_path):
        """Allow reading from allowed paths."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("allowed content")

        tool = FileReadTool(allowed_paths=[str(tmp_path)])
        result = tool.execute(path=str(test_file))

        assert result.success is True
        assert "allowed content" in result.output

    def test_allowed_paths_blocked(self, tmp_path):
        """Block reading from paths outside allowed list."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        blocked_dir = tmp_path / "blocked"
        blocked_dir.mkdir()
        blocked_file = blocked_dir / "secret.txt"
        blocked_file.write_text("secret content")

        tool = FileReadTool(allowed_paths=[str(allowed_dir)])
        result = tool.execute(path=str(blocked_file))

        assert result.success is False
        assert "not under allowed" in result.error.lower()

    def test_symlink_escape_blocked(self, tmp_path):
        """Block symlinks that point outside allowed paths."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        secret_file = secret_dir / "password.txt"
        secret_file.write_text("SECRET")

        # Create symlink inside allowed dir pointing to secret file
        symlink = allowed_dir / "link"
        symlink.symlink_to(secret_file)

        tool = FileReadTool(allowed_paths=[str(allowed_dir)])
        result = tool.execute(path=str(symlink))

        assert result.success is False
        assert "not under allowed" in result.error.lower()

    def test_permission_denied(self, tmp_path):
        """Return error when file is not readable."""
        import os

        test_file = tmp_path / "noperm.txt"
        test_file.write_text("content")
        os.chmod(test_file, 0o000)

        try:
            tool = FileReadTool(working_dir=str(tmp_path))
            result = tool.execute(path="noperm.txt")

            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            os.chmod(test_file, 0o644)  # Restore for cleanup

    def test_binary_file_error(self, tmp_path):
        """Return error when file contains non-UTF-8 content."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x80\x81\x82\xff\xfe")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="binary.bin")

        assert result.success is False
        assert "utf-8" in result.error.lower() or "binary" in result.error.lower()

    def test_missing_path_argument(self):
        """Return error when path argument is missing."""
        tool = FileReadTool()

        result = tool.execute()
        assert result.success is False
        assert "missing" in result.error.lower()

        result = tool.execute(path="")
        assert result.success is False
        assert "missing" in result.error.lower() or "invalid" in result.error.lower()

    def test_invalid_path_type(self):
        """Return error when path is not a string."""
        tool = FileReadTool()

        result = tool.execute(path=123)
        assert result.success is False
        assert "invalid" in result.error.lower()

        result = tool.execute(path=None)
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_tool_properties(self):
        """Tool has correct properties."""
        tool = FileReadTool()

        assert tool.name == "read_file"
        assert "file" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "path" in tool.parameters["properties"]
        assert "offset" in tool.parameters["properties"]
        assert "limit" in tool.parameters["properties"]

    def test_to_openai_function(self):
        """Converts to OpenAI function format."""
        tool = FileReadTool()
        func = tool.to_openai_function()

        assert func["type"] == "function"
        assert func["function"]["name"] == "read_file"
        assert "path" in func["function"]["parameters"]["properties"]


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

    def test_run_malformed_json_arguments(self, mock_llm, registry_with_shell):
        """Malformed JSON arguments return parse error to LLM instead of silently failing."""
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
                            "arguments": "{invalid json",  # Malformed JSON
                        },
                    }
                ],
            ),
            CompletionResult(
                id="resp-2",
                content="Got parse error, will try differently",
                model="test",
                tokens_used=10,
                latency_ms=100,
                tool_calls=None,
            ),
        ]

        executor = ToolExecutor(mock_llm, registry_with_shell)
        result = executor.run(messages=[Message(role="user", content="Run command")])

        # The parse error should be returned as a tool error
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result.success is False
        assert "Failed to parse arguments" in result.tool_calls[0].result.error
