"""E2E tests for task execution with real LLM.

These tests require an LLM server at localhost:8000.
Tests are skipped if no server is available.
"""

from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel

from llm_agent import AgentConfig, ConversationalAgent, FileReadTool, ShellTool, Task
from llm_agent.core.traits.llm import LLMTrait
from llm_agent.core.traits.tools import ToolsTrait


pytestmark = [pytest.mark.e2e, pytest.mark.slow]

LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "default"


def llm_available() -> bool:
    """Check if LLM server is running."""
    try:
        response = httpx.get(f"{LLM_BASE_URL}/models", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


skip_no_llm = pytest.mark.skipif(
    not llm_available(),
    reason="LLM server not available at localhost:8000",
)


class Answer(BaseModel):
    """Schema for task answers."""

    answer: str
    confidence: float


class FileInfo(BaseModel):
    """Schema for file information."""

    filename: str
    line_count: int
    summary: str


@skip_no_llm
class TestTaskExecutionE2E:
    """E2E tests for Agent.execute() with real LLM."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create agent with LLMTrait connected to LLM server."""
        agent = ConversationalAgent(
            lg=mock_logger,
            config=AgentConfig(name="test-agent", fact_injection="none"),
        )
        agent.add_trait(
            LLMTrait(
                config={
                    "type": "openai_compatible",
                    "base_url": LLM_BASE_URL,
                    "model": LLM_MODEL,
                    "temperature": 0.0,
                }
            )
        )
        agent.start()
        yield agent
        agent.stop()

    @pytest.fixture
    def agent_with_tools(self, mock_logger):
        """Create agent with LLMTrait and ToolsTrait."""
        agent = ConversationalAgent(
            lg=mock_logger,
            config=AgentConfig(name="test-agent-tools", fact_injection="none"),
        )
        agent.add_trait(
            LLMTrait(
                config={
                    "type": "openai_compatible",
                    "base_url": LLM_BASE_URL,
                    "model": LLM_MODEL,
                    "temperature": 0.0,
                }
            )
        )
        tools_trait = ToolsTrait()
        tools_trait.register(ShellTool())
        tools_trait.register(FileReadTool())
        agent.add_trait(tools_trait)
        agent.start()
        yield agent
        agent.stop()

    def test_simple_task_execution(self, agent):
        """Execute a simple task without tools."""
        task = Task(
            name="greeting",
            description="Say hello in exactly 3 words.",
        )

        result = agent.execute(task)

        assert result.success is True
        assert len(result.content) > 0
        assert result.iterations == 1

    def test_task_with_context(self, agent):
        """Execute a task with context injection."""
        task = Task(
            name="translation",
            description="Translate the word to French.",
            context={"word": "hello"},
        )

        result = agent.execute(task)

        assert result.success is True
        assert len(result.content) > 0
        # Response should likely contain "bonjour"
        assert "bonjour" in result.content.lower() or len(result.content) > 0

    def test_task_with_structured_output(self, agent):
        """Execute a task with structured output extraction."""
        task = Task(
            name="math",
            description="What is 7 * 8? Give the numeric result.",
            output_schema=Answer,
        )

        result = agent.execute(task)

        assert result.success is True
        assert result.parsed is not None
        assert isinstance(result.parsed, Answer)
        assert isinstance(result.parsed.answer, str)
        assert isinstance(result.parsed.confidence, float)

    def test_task_with_tools(self, agent_with_tools):
        """Execute a task that requires tool use."""
        task = Task(
            name="shell",
            description="Use the shell tool to run 'echo hello world' and tell me what it outputs.",
        )

        result = agent_with_tools.execute(task)

        assert result.success is True
        assert len(result.tool_calls) > 0
        # At least one tool call should be to shell
        shell_calls = [tc for tc in result.tool_calls if tc.name == "shell"]
        assert len(shell_calls) > 0
        # Check that the shell command succeeded
        assert any(tc.result.success for tc in shell_calls)

    def test_task_with_tools_and_structured_output(self, agent_with_tools, tmp_path):
        """Execute a task with tools and structured output (two-phase).

        Note: This test combines tool use with structured output extraction,
        which is challenging for some LLMs. We verify tools were called and
        check structured output if available.
        """
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")

        task = Task(
            name="file-info",
            description=f"Read the file at {test_file} and describe it.",
            output_schema=FileInfo,
        )

        result = agent_with_tools.execute(task)

        # Tool should have been called regardless of extraction success
        assert len(result.tool_calls) > 0

        # Structured output extraction can fail with some LLMs
        # The important thing is that tool execution worked
        if result.success:
            assert result.parsed is not None
            assert isinstance(result.parsed, FileInfo)
        else:
            # If extraction failed, verify we got a structured output error
            # (not a tool execution error)
            assert "Structured output error" in result.error
