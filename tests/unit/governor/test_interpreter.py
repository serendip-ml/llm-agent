"""Tests for ResponseInterpreter."""

import json
from unittest.mock import MagicMock

import pytest

from llm_agent.core.governor.interpreter import ResponseInterpreter
from llm_agent.core.governor.types import ResponseEvent
from llm_agent.core.llm.types import CompletionResult
from llm_agent.core.tools.base import BaseTool, ToolResult
from llm_agent.core.tools.registry import ToolRegistry


pytestmark = pytest.mark.unit


class MockWorkTool(BaseTool):
    """Non-terminal work tool for testing."""

    name = "work_tool"
    description = "Does work"
    parameters = {"type": "object", "properties": {}}

    def execute(self) -> ToolResult:
        return ToolResult(success=True, output="done")


class MockTerminalTool(BaseTool):
    """Terminal tool for testing."""

    name = "complete_task"
    description = "Completes task"
    parameters = {"type": "object", "properties": {}}
    terminal = True

    def execute(self) -> ToolResult:
        return ToolResult(success=True, output="completed", terminal=True)


@pytest.fixture
def registry():
    """Registry with work and terminal tools."""
    reg = ToolRegistry()
    reg.register(MockWorkTool())
    reg.register(MockTerminalTool())
    return reg


@pytest.fixture
def interpreter(registry):
    """Interpreter with test registry."""
    return ResponseInterpreter(registry)


def make_completion(
    content: str = "",
    tool_calls: list | None = None,
) -> CompletionResult:
    """Helper to create CompletionResult."""
    return CompletionResult(
        id="test-id",
        content=content,
        model="test",
        tokens_used=10,
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


class TestResponseInterpreter:
    """Tests for ResponseInterpreter event classification."""

    def test_text_only_no_tool_calls(self, interpreter):
        """TEXT_ONLY when no tool calls in response."""
        result = make_completion(content="Here is the answer")

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.TEXT_ONLY
        assert interpreted.tool_calls == ()
        assert interpreted.terminal_call is None

    def test_text_only_empty_tool_calls(self, interpreter):
        """TEXT_ONLY when tool_calls is empty list."""
        result = make_completion(content="Answer", tool_calls=[])

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.TEXT_ONLY

    def test_work_tools_single(self, interpreter):
        """WORK_TOOLS when single non-terminal tool called."""
        result = make_completion(tool_calls=[make_tool_call("work_tool")])

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.WORK_TOOLS
        assert len(interpreted.tool_calls) == 1
        assert interpreted.tool_calls[0].name == "work_tool"
        assert interpreted.terminal_call is None

    def test_work_tools_multiple(self, interpreter):
        """WORK_TOOLS when multiple non-terminal tools called."""
        result = make_completion(
            tool_calls=[
                make_tool_call("work_tool", "call-1"),
                make_tool_call("work_tool", "call-2"),
            ]
        )

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.WORK_TOOLS
        assert len(interpreted.tool_calls) == 2

    def test_early_terminal_no_prior_work(self, interpreter):
        """EARLY_TERMINAL when terminal called without prior work."""
        result = make_completion(tool_calls=[make_tool_call("complete_task")])

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.EARLY_TERMINAL
        assert interpreted.terminal_call is not None
        assert interpreted.terminal_call.name == "complete_task"

    def test_terminal_after_work(self, interpreter):
        """TERMINAL_AFTER_WORK when terminal called with prior work."""
        result = make_completion(tool_calls=[make_tool_call("complete_task")])

        interpreted = interpreter.interpret(result, has_prior_work=True)

        assert interpreted.event == ResponseEvent.TERMINAL_AFTER_WORK
        assert interpreted.terminal_call is not None

    def test_mixed_batch_work_and_terminal(self, interpreter):
        """MIXED_BATCH when both work and terminal in same response."""
        result = make_completion(
            tool_calls=[
                make_tool_call("work_tool", "call-1"),
                make_tool_call("complete_task", "call-2"),
            ]
        )

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.MIXED_BATCH
        assert len(interpreted.tool_calls) == 2
        assert interpreted.terminal_call is not None
        assert interpreted.terminal_call.name == "complete_task"

    def test_unknown_tool_treated_as_work(self, interpreter):
        """Unknown tools are treated as work tools."""
        result = make_completion(tool_calls=[make_tool_call("unknown_tool")])

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.event == ResponseEvent.WORK_TOOLS
        assert interpreted.terminal_call is None

    def test_parse_errors_tracked(self, interpreter):
        """Parse errors for malformed arguments are tracked."""
        result = make_completion(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "work_tool",
                        "arguments": "{invalid json",
                    },
                }
            ]
        )

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert "call-1" in interpreted.parse_errors
        assert "Failed to parse" in interpreted.parse_errors["call-1"]

    def test_raw_result_preserved(self, interpreter):
        """Original CompletionResult is preserved."""
        result = make_completion(content="test content")

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.raw is result
        assert interpreted.raw.content == "test content"

    def test_arguments_parsed(self, interpreter):
        """Tool arguments are correctly parsed."""
        result = make_completion(
            tool_calls=[make_tool_call("work_tool", arguments={"key": "value", "num": 42})]
        )

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.tool_calls[0].arguments == {"key": "value", "num": 42}

    def test_empty_arguments(self, interpreter):
        """Empty arguments handled correctly."""
        result = make_completion(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "work_tool", "arguments": ""},
                }
            ]
        )

        interpreted = interpreter.interpret(result, has_prior_work=False)

        assert interpreted.tool_calls[0].arguments == {}

    def test_object_style_tool_call(self, interpreter):
        """Handle object-style tool calls (with attributes instead of dict).

        The interpreter's _parse_tool_call can handle both dict and object forms.
        We test this by directly calling _parse_tool_call with a MagicMock.
        """
        # Simulate object with attributes
        function_obj = MagicMock()
        function_obj.name = "work_tool"
        function_obj.arguments = '{"key": "value"}'

        tc_obj = MagicMock()
        tc_obj.id = "call-1"
        tc_obj.function = function_obj

        # Test the internal parse method directly
        parsed, error = interpreter._parse_tool_call(tc_obj)

        assert error is None
        assert parsed.name == "work_tool"
        assert parsed.arguments == {"key": "value"}
        assert parsed.id == "call-1"
