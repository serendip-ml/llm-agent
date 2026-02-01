"""Tests for PromptOnlyAgent."""

from unittest.mock import MagicMock

import pytest

from llm_agent.core.prompt_agent import (
    PromptOnlyAgent,
    PromptOnlyAgentConfig,
    _json_schema_to_pydantic,
    _substitute_variables,
)
from llm_agent.core.traits.directive import Directive
from llm_agent.core.traits.llm import LLMConfig


pytestmark = pytest.mark.unit


class TestVariableSubstitution:
    """Tests for variable substitution."""

    def test_substitute_from_dict(self):
        """Substitute variables from provided dict."""
        result = _substitute_variables(
            "Path: {{CODEBASE_PATH}}", {"CODEBASE_PATH": "/home/user/code"}
        )
        assert result == "Path: /home/user/code"

    def test_substitute_multiple(self):
        """Substitute multiple variables."""
        result = _substitute_variables(
            "{{FOO}} and {{BAR}}",
            {"FOO": "hello", "BAR": "world"},
        )
        assert result == "hello and world"

    def test_substitute_from_env(self, monkeypatch):
        """Fall back to environment variables."""
        monkeypatch.setenv("MY_VAR", "from-env")
        result = _substitute_variables("Value: {{MY_VAR}}", {})
        assert result == "Value: from-env"

    def test_substitute_dict_overrides_env(self, monkeypatch):
        """Dict values take precedence over environment."""
        monkeypatch.setenv("MY_VAR", "from-env")
        result = _substitute_variables("Value: {{MY_VAR}}", {"MY_VAR": "from-dict"})
        assert result == "Value: from-dict"

    def test_substitute_missing_raises(self):
        """Raise error for missing variables."""
        with pytest.raises(ValueError, match="MISSING_VAR"):
            _substitute_variables("Value: {{MISSING_VAR}}", {})

    def test_substitute_no_variables(self):
        """Text without variables passes through."""
        result = _substitute_variables("No variables here", {})
        assert result == "No variables here"


class TestJsonSchemaToPydantic:
    """Tests for JSON schema conversion."""

    def test_simple_schema(self):
        """Convert simple schema to Pydantic model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        Model = _json_schema_to_pydantic(schema, "Person")

        instance = Model(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_optional_fields(self):
        """Handle optional fields correctly."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }
        Model = _json_schema_to_pydantic(schema, "Test")

        instance = Model(required_field="value")
        assert instance.required_field == "value"
        assert instance.optional_field is None

    def test_all_types(self):
        """Support all basic JSON Schema types."""
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            },
        }
        Model = _json_schema_to_pydantic(schema, "AllTypes")

        instance = Model(s="text", i=42, n=3.14, b=True, a=[1, 2], o={"k": "v"})
        assert instance.s == "text"
        assert instance.i == 42
        assert instance.n == 3.14
        assert instance.b is True
        assert instance.a == [1, 2]
        assert instance.o == {"k": "v"}


class TestPromptOnlyAgentConfig:
    """Tests for PromptOnlyAgentConfig."""

    def test_minimal_config(self):
        """Create config with minimal fields."""
        config = PromptOnlyAgentConfig(
            name="test-agent",
            directive=Directive(prompt="Be helpful"),
            task={"description": "Do something"},
        )
        assert config.name == "test-agent"
        assert config.directive.prompt == "Be helpful"
        assert config.task.description == "Do something"
        assert config.tools == {}
        assert config.schedule is None

    def test_full_config(self):
        """Create config with all fields."""
        config = PromptOnlyAgentConfig(
            name="explorer",
            directive=Directive(prompt="Explore code"),
            task={"description": "Find insights", "output_schema": {"type": "object"}},
            tools={"shell": {"allowed_commands": ["ls"]}},
            schedule={"interval": 600},
        )
        assert config.name == "explorer"
        assert config.schedule.interval == 600
        assert "shell" in config.tools


class TestPromptOnlyAgent:
    """Tests for PromptOnlyAgent."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def llm_config(self):
        return LLMConfig(base_url="http://localhost:8000/v1")

    @pytest.fixture
    def minimal_config_dict(self):
        return {
            "name": "test-agent",
            "directive": {"prompt": "You are a test agent."},
            "task": {"description": "Run tests"},
        }

    def test_from_dict_minimal(self, mock_logger, llm_config, minimal_config_dict):
        """Create agent from minimal config dict."""
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=minimal_config_dict,
            llm_config=llm_config,
        )
        assert agent.name == "test-agent"
        assert agent.config.directive.prompt == "You are a test agent."

    def test_from_dict_with_variables(self, mock_logger, llm_config):
        """Variable substitution in config."""
        config_dict = {
            "name": "explorer",
            "directive": {"prompt": "Explore {{CODEBASE_PATH}}"},
            "task": {"description": "Find code in {{CODEBASE_PATH}}"},
        }
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=config_dict,
            llm_config=llm_config,
            variables={"CODEBASE_PATH": "/home/user/code"},
        )
        assert "/home/user/code" in agent.config.directive.prompt
        assert "/home/user/code" in agent.config.task.description

    def test_from_dict_with_tools(self, mock_logger, llm_config):
        """Configure tools from config."""
        config_dict = {
            "name": "tool-agent",
            "directive": {"prompt": "Use tools"},
            "task": {"description": "Do work"},
            "tools": {
                "shell": {"allowed_commands": ["echo", "ls"]},
            },
        }
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=config_dict,
            llm_config=llm_config,
        )

        # Check tools trait has shell registered
        from llm_agent.core.traits.tools import ToolsTrait

        tools_trait = agent.agent.get_trait(ToolsTrait)
        assert tools_trait is not None
        assert "shell" in tools_trait.registry

    def test_from_dict_with_learn_tools(self, mock_logger, llm_config):
        """Configure memory tools with LearnTrait."""
        mock_learn_trait = MagicMock()
        mock_learn_trait.has_embedder = False

        config_dict = {
            "name": "learning-agent",
            "directive": {"prompt": "Learn things"},
            "task": {"description": "Discover insights"},
            "tools": {
                "remember": {},
                "recall": {},
            },
        }
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=config_dict,
            llm_config=llm_config,
            learn_trait=mock_learn_trait,
        )

        from llm_agent.core.traits.tools import ToolsTrait

        tools_trait = agent.agent.get_trait(ToolsTrait)
        assert "remember" in tools_trait.registry
        assert "recall" in tools_trait.registry

    def test_learn_tools_require_trait(self, mock_logger, llm_config):
        """Memory tools fail without LearnTrait."""
        config_dict = {
            "name": "agent",
            "directive": {"prompt": "Test"},
            "task": {"description": "Test"},
            "tools": {"remember": {}},
        }
        with pytest.raises(ValueError, match="LearnTrait"):
            PromptOnlyAgent.from_dict(
                lg=mock_logger,
                config_dict=config_dict,
                llm_config=llm_config,
            )

    def test_get_recent_results(self, mock_logger, llm_config, minimal_config_dict):
        """Get recent task results."""
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=minimal_config_dict,
            llm_config=llm_config,
        )

        # Add some fake results
        from llm_agent.core.task import TaskResult

        for i in range(5):
            agent._results.append(TaskResult(success=True, content=f"Result {i}"))

        recent = agent.get_recent_results(limit=3)
        assert len(recent) == 3
        assert recent[-1].content == "Result 4"

    def test_run_once_stores_result(self, mock_logger, llm_config, minimal_config_dict):
        """run_once stores result in history."""
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=minimal_config_dict,
            llm_config=llm_config,
        )

        # Mock the internal conversation loop
        from llm_agent.core.task import TaskResult

        mock_result = TaskResult(success=True, content="Test output")
        agent._run_conversation_loop = MagicMock(return_value=mock_result)

        result = agent.run_once()

        assert result.content == "Test output"
        assert len(agent._results) == 1
        assert agent._results[0] is result

    def test_run_once_increments_cycle_count(self, mock_logger, llm_config, minimal_config_dict):
        """run_once increments cycle count and maintains conversation."""
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=minimal_config_dict,
            llm_config=llm_config,
        )

        from llm_agent.core.task import TaskResult

        mock_result = TaskResult(success=True, content="Test output")
        agent._run_conversation_loop = MagicMock(return_value=mock_result)

        # First run
        agent.run_once()
        assert agent.cycle_count == 1
        # Conversation should have system + user messages
        assert agent.conversation.message_count == 2

        # Second run
        agent.run_once()
        assert agent.cycle_count == 2
        # Should have added a continuation message
        assert agent.conversation.message_count == 3

    def test_reset_conversation(self, mock_logger, llm_config, minimal_config_dict):
        """reset_conversation clears state."""
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=minimal_config_dict,
            llm_config=llm_config,
        )

        from llm_agent.core.task import TaskResult

        mock_result = TaskResult(success=True, content="Test output")
        agent._run_conversation_loop = MagicMock(return_value=mock_result)

        agent.run_once()
        assert agent.cycle_count == 1
        assert agent.conversation.message_count > 0

        agent.reset_conversation()
        assert agent.cycle_count == 0
        assert agent.conversation.message_count == 0

    def test_ask_uses_context(self, mock_logger, llm_config, minimal_config_dict):
        """ask() includes context from results."""
        agent = PromptOnlyAgent.from_dict(
            lg=mock_logger,
            config_dict=minimal_config_dict,
            llm_config=llm_config,
        )

        # Add some results
        from llm_agent.core.task import TaskResult

        agent._results.append(TaskResult(success=True, content="Found a bug"))

        # Mock complete
        from llm_agent.core.llm.types import CompletionResult

        mock_completion = CompletionResult(
            id="test",
            content="I found a bug recently.",
            model="test",
            tokens_used=10,
            latency_ms=100,
        )
        agent.agent.complete = MagicMock(return_value=mock_completion)
        agent.agent._started = True

        response = agent.ask("What have you learned?")

        assert response == "I found a bug recently."
        # Verify context was included in the call
        call_args = agent.agent.complete.call_args
        assert "Recent Activity" in call_args.kwargs["system_prompt"]
