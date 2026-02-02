"""Tests for task execution."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llm_agent import (
    AgentConfig,
    CompletionResult,
    ConversationalAgent,
    ShellTool,
    Task,
    TaskResult,
    ToolCallResult,
    ToolRegistry,
    ToolResult,
)
from llm_agent.core.llm.backend import StructuredOutputError


pytestmark = pytest.mark.unit


class Answer(BaseModel):
    """Test output schema."""

    answer: str
    confidence: float


class TestTaskModel:
    """Tests for Task model."""

    def test_minimal_task(self):
        task = Task(name="test", description="Do something")

        assert task.name == "test"
        assert task.description == "Do something"
        assert task.context == {}
        assert task.output_schema is None
        assert task.system_prompt is None
        assert task.max_iterations == 10

    def test_full_task(self):
        task = Task(
            name="analyze",
            description="Analyze sentiment",
            context={"text": "Hello"},
            output_schema=Answer,
            system_prompt="Be precise.",
            max_iterations=5,
        )

        assert task.name == "analyze"
        assert task.description == "Analyze sentiment"
        assert task.context == {"text": "Hello"}
        assert task.output_schema is Answer
        assert task.system_prompt == "Be precise."
        assert task.max_iterations == 5


class TestTaskResultModel:
    """Tests for TaskResult model."""

    def test_success_result(self):
        result = TaskResult(
            success=True,
            content="The answer is 4",
            parsed=Answer(answer="4", confidence=0.95),
            iterations=2,
            tokens_used=100,
        )

        assert result.success is True
        assert result.content == "The answer is 4"
        assert result.parsed.answer == "4"
        assert result.parsed.confidence == 0.95
        assert result.tool_calls == []
        assert result.iterations == 2
        assert result.tokens_used == 100
        assert result.error is None

    def test_error_result(self):
        result = TaskResult(
            success=False,
            content="",
            error="LLM failed",
        )

        assert result.success is False
        assert result.content == ""
        assert result.parsed is None
        assert result.error == "LLM failed"

    def test_result_with_tool_calls(self):
        tool_call = ToolCallResult(
            call_id="call-1",
            name="shell",
            result=ToolResult(success=True, output="hello"),
        )
        result = TaskResult(
            success=True,
            content="Executed command",
            tool_calls=[tool_call],
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"


class TestAgentExecuteSimple:
    """Tests for Agent.execute() without tools."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="The answer is 4",
            model="default",
            tokens_used=50,
            latency_ms=100,
            parsed=None,
        )
        return trait

    @pytest.fixture
    def agent_with_llm(self, mock_logger, mock_llm_trait):
        from llm_agent.core.traits.llm import LLMTrait

        agent = ConversationalAgent(
            lg=mock_logger, config=AgentConfig(name="test", fact_injection="none")
        )
        agent._traits[LLMTrait] = mock_llm_trait
        return agent

    def test_execute_requires_llm_trait(self, mock_logger):
        agent = ConversationalAgent(lg=mock_logger, config=AgentConfig(name="test"))
        task = Task(name="test", description="Do something")

        with pytest.raises(RuntimeError, match="LLMTrait required"):
            agent.execute(task)

    def test_execute_simple_completion(self, agent_with_llm, mock_llm_trait):
        task = Task(name="math", description="What is 2+2?")

        result = agent_with_llm.execute(task)

        assert result.success is True
        assert result.content == "The answer is 4"
        assert result.iterations == 1
        assert result.tokens_used == 50
        assert result.tool_calls == []

    def test_execute_uses_task_system_prompt(self, agent_with_llm, mock_llm_trait):
        task = Task(
            name="test",
            description="Query",
            system_prompt="Be concise.",
        )

        agent_with_llm.execute(task)

        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        assert "Be concise." in messages[0].content

    def test_execute_injects_task_context(self, agent_with_llm, mock_llm_trait):
        task = Task(
            name="analyze",
            description="Analyze the text",
            context={"text": "Hello world", "language": "English"},
        )

        agent_with_llm.execute(task)

        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        system_prompt = messages[0].content
        assert "Task Context" in system_prompt
        assert "text: Hello world" in system_prompt
        assert "language: English" in system_prompt


class TestAgentExecuteStructured:
    """Tests for Agent.execute() with structured output."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        return trait

    @pytest.fixture
    def agent_with_llm(self, mock_logger, mock_llm_trait):
        from llm_agent.core.traits.llm import LLMTrait

        agent = ConversationalAgent(
            lg=mock_logger, config=AgentConfig(name="test", fact_injection="none")
        )
        agent._traits[LLMTrait] = mock_llm_trait
        return agent

    def test_execute_with_output_schema(self, agent_with_llm, mock_llm_trait):
        parsed_answer = Answer(answer="4", confidence=0.95)
        mock_llm_trait.complete.return_value = CompletionResult(
            id="resp-1",
            content='{"answer": "4", "confidence": 0.95}',
            model="default",
            tokens_used=60,
            latency_ms=100,
            parsed=parsed_answer,
        )

        task = Task(
            name="math",
            description="What is 2+2?",
            output_schema=Answer,
        )

        result = agent_with_llm.execute(task)

        assert result.success is True
        assert result.parsed.answer == "4"
        assert result.parsed.confidence == 0.95
        mock_llm_trait.complete.assert_called_once()
        assert mock_llm_trait.complete.call_args.kwargs["output_schema"] is Answer

    def test_execute_structured_output_error(self, agent_with_llm, mock_llm_trait):
        mock_llm_trait.complete.side_effect = StructuredOutputError("Invalid JSON")

        task = Task(
            name="math",
            description="What is 2+2?",
            output_schema=Answer,
        )

        result = agent_with_llm.execute(task)

        assert result.success is False
        assert "Structured output error" in result.error


class TestAgentExecuteWithTools:
    """Tests for Agent.execute() with ToolsTrait."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        return trait

    @pytest.fixture
    def mock_tools_trait(self):
        from llm_agent.core.traits.tools import ToolsTrait

        trait = MagicMock(spec=ToolsTrait)
        registry = ToolRegistry()
        registry.register(ShellTool())
        trait.registry = registry
        trait.has_tools.return_value = True
        return trait

    @pytest.fixture
    def agent_with_tools(self, mock_logger, mock_llm_trait, mock_tools_trait):
        from llm_agent.core.traits.llm import LLMTrait
        from llm_agent.core.traits.tools import ToolsTrait

        agent = ConversationalAgent(
            lg=mock_logger, config=AgentConfig(name="test", fact_injection="none")
        )
        agent._traits[LLMTrait] = mock_llm_trait
        agent._traits[ToolsTrait] = mock_tools_trait
        return agent

    def test_execute_calls_tools(self, agent_with_tools, mock_llm_trait, mock_tools_trait):
        # First call: LLM wants to call shell tool
        # Second call: LLM returns final response
        mock_llm_trait.complete.side_effect = [
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
            CompletionResult(
                id="resp-2",
                content="The output was: hello",
                model="test",
                tokens_used=15,
                latency_ms=100,
                tool_calls=None,
            ),
        ]

        task = Task(name="run", description="Run echo hello")

        result = agent_with_tools.execute(task)

        assert result.success is True
        assert result.content == "The output was: hello"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"
        assert result.iterations == 2
        assert result.tokens_used == 25

    def test_execute_no_tools_when_trait_empty(self, mock_logger, mock_llm_trait):
        """When ToolsTrait has no tools, use simple completion path."""
        from llm_agent.core.traits.llm import LLMTrait
        from llm_agent.core.traits.tools import ToolsTrait

        mock_llm_trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Simple answer",
            model="default",
            tokens_used=50,
            latency_ms=100,
        )

        empty_tools_trait = MagicMock(spec=ToolsTrait)
        empty_tools_trait.has_tools.return_value = False

        agent = ConversationalAgent(
            lg=mock_logger, config=AgentConfig(name="test", fact_injection="none")
        )
        agent._traits[LLMTrait] = mock_llm_trait
        agent._traits[ToolsTrait] = empty_tools_trait

        task = Task(name="test", description="Query")
        result = agent.execute(task)

        assert result.success is True
        assert result.content == "Simple answer"
        # Should not have called complete with tools
        assert mock_llm_trait.complete.call_args.kwargs.get("tools") is None

    def test_execute_max_iterations_exceeded(
        self, agent_with_tools, mock_llm_trait, mock_tools_trait
    ):
        """Error when tool loop exceeds max_iterations."""
        # LLM keeps calling tools forever
        mock_llm_trait.complete.return_value = CompletionResult(
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

        task = Task(name="loop", description="Loop forever", max_iterations=3)

        result = agent_with_tools.execute(task)

        assert result.success is False
        assert "exceeded" in result.error.lower()


class TestAgentExecuteToolsAndSchema:
    """Tests for Agent.execute() with both tools and structured output."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        return trait

    @pytest.fixture
    def mock_tools_trait(self):
        from llm_agent.core.traits.tools import ToolsTrait

        trait = MagicMock(spec=ToolsTrait)
        registry = ToolRegistry()
        registry.register(ShellTool())
        trait.registry = registry
        trait.has_tools.return_value = True
        return trait

    @pytest.fixture
    def agent_with_tools(self, mock_logger, mock_llm_trait, mock_tools_trait):
        from llm_agent.core.traits.llm import LLMTrait
        from llm_agent.core.traits.tools import ToolsTrait

        agent = ConversationalAgent(
            lg=mock_logger, config=AgentConfig(name="test", fact_injection="none")
        )
        agent._traits[LLMTrait] = mock_llm_trait
        agent._traits[ToolsTrait] = mock_tools_trait
        return agent

    def test_execute_tools_then_extract_schema(
        self, agent_with_tools, mock_llm_trait, mock_tools_trait
    ):
        """Two-phase execution: tool loop, then structured extraction."""
        parsed_answer = Answer(answer="hello", confidence=0.9)

        # Phase 1: Tool loop
        # Phase 2: Structured extraction
        mock_llm_trait.complete.side_effect = [
            # Tool call
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
            # Final tool loop response
            CompletionResult(
                id="resp-2",
                content="The shell output was: hello",
                model="test",
                tokens_used=15,
                latency_ms=100,
                tool_calls=None,
            ),
            # Structured extraction phase
            CompletionResult(
                id="resp-3",
                content='{"answer": "hello", "confidence": 0.9}',
                model="test",
                tokens_used=20,
                latency_ms=100,
                parsed=parsed_answer,
            ),
        ]

        task = Task(
            name="analyze",
            description="What does echo hello output?",
            output_schema=Answer,
        )

        result = agent_with_tools.execute(task)

        assert result.success is True
        assert result.parsed.answer == "hello"
        assert result.parsed.confidence == 0.9
        # Tool calls from phase 1
        assert len(result.tool_calls) == 1
        # Tokens from both phases
        assert result.tokens_used == 45  # 10 + 15 + 20

    def test_execute_extraction_error(self, agent_with_tools, mock_llm_trait, mock_tools_trait):
        """Handle structured extraction failure after successful tool loop."""
        mock_llm_trait.complete.side_effect = [
            # Tool call
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
            # Final tool loop response
            CompletionResult(
                id="resp-2",
                content="Done",
                model="test",
                tokens_used=15,
                latency_ms=100,
                tool_calls=None,
            ),
            # Extraction fails
            StructuredOutputError("Invalid schema"),
        ]

        task = Task(
            name="analyze",
            description="What does echo hello output?",
            output_schema=Answer,
        )

        result = agent_with_tools.execute(task)

        assert result.success is False
        assert "Structured output error" in result.error
        # Tool calls should still be present from phase 1
        assert len(result.tool_calls) == 1


class TestToolsTrait:
    """Tests for ToolsTrait."""

    def test_register_tool(self):
        from llm_agent.core.traits.tools import ToolsTrait

        trait = ToolsTrait()
        tool = ShellTool()

        trait.register(tool)

        assert trait.has_tools() is True
        assert trait.registry.get("shell") is tool

    def test_unregister_tool(self):
        from llm_agent.core.traits.tools import ToolsTrait

        trait = ToolsTrait()
        trait.register(ShellTool())

        trait.unregister("shell")

        assert trait.has_tools() is False

    def test_trait_lifecycle(self):
        from llm_agent.core.traits.tools import ToolsTrait

        trait = ToolsTrait()
        mock_agent = MagicMock()

        trait.attach(mock_agent)
        trait.on_start()
        trait.on_stop()

        assert trait._agent is mock_agent

    def test_empty_registry(self):
        from llm_agent.core.traits.tools import ToolsTrait

        trait = ToolsTrait()

        assert trait.has_tools() is False
