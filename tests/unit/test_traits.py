"""Tests for traits."""

from unittest.mock import MagicMock

import pytest

from llm_agent import Agent, AgentConfig, BaseTrait, CompletionResult, Directive, DirectiveTrait


pytestmark = pytest.mark.unit


class TestBaseTrait:
    """Tests for BaseTrait."""

    def test_init(self):
        trait = BaseTrait()

        assert trait._agent is None

    def test_agent_property_raises_if_not_attached(self):
        trait = BaseTrait()

        with pytest.raises(RuntimeError, match="not attached to agent"):
            _ = trait.agent

    def test_attach(self):
        trait = BaseTrait()
        mock_agent = MagicMock()

        trait.attach(mock_agent)

        assert trait._agent == mock_agent

    def test_agent_property_returns_agent_after_attach(self):
        trait = BaseTrait()
        mock_agent = MagicMock()
        trait.attach(mock_agent)

        assert trait.agent == mock_agent

    def test_lifecycle_methods_exist(self):
        trait = BaseTrait()

        # Should not raise - these are no-op by default
        trait.on_start()
        trait.on_stop()


class TestCustomTrait:
    """Tests for custom traits using BaseTrait."""

    def test_custom_trait_inherits_behavior(self):
        class MyTrait(BaseTrait):
            def __init__(self, value: str) -> None:
                super().__init__()
                self.value = value

            def get_value(self) -> str:
                return f"{self.agent.name}: {self.value}"

        trait = MyTrait("test")
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        trait.attach(mock_agent)

        assert trait.get_value() == "test-agent: test"


class TestDirective:
    """Tests for Directive."""

    def test_directive_with_prompt(self):
        directive = Directive(prompt="You are a code reviewer.")

        assert directive.prompt == "You are a code reviewer."
        assert directive.extensions == {}

    def test_directive_with_extensions(self):
        directive = Directive(
            prompt="You are a code reviewer.",
            extensions={"custom_field": "value"},
        )

        assert directive.prompt == "You are a code reviewer."
        assert directive.extensions == {"custom_field": "value"}


class TestDirectiveTrait:
    """Tests for DirectiveTrait."""

    def test_init(self):
        directive = Directive(prompt="Test directive")
        trait = DirectiveTrait(directive)

        assert trait.directive == directive

    def test_attach(self):
        directive = Directive(prompt="Test directive")
        trait = DirectiveTrait(directive)
        mock_agent = MagicMock()

        trait.attach(mock_agent)

        assert trait._agent == mock_agent

    def test_build_prompt(self):
        directive = Directive(prompt="You are a code reviewer. Be critical.")
        trait = DirectiveTrait(directive)

        result = trait.build_prompt("Base system prompt.")

        # Directive is prepended
        assert result.startswith("You are a code reviewer.")
        assert "Base system prompt." in result


class TestAgentTraits:
    """Tests for Agent trait support."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    def test_add_trait(self, mock_logger):
        config = AgentConfig(name="test")
        agent = Agent(lg=mock_logger, config=config)

        directive = Directive(prompt="Test directive")
        trait = DirectiveTrait(directive)
        agent.add_trait(trait)

        assert agent.has_trait(DirectiveTrait)
        assert agent.get_trait(DirectiveTrait) == trait

    def test_add_trait_attaches(self, mock_logger):
        config = AgentConfig(name="test")
        agent = Agent(lg=mock_logger, config=config)

        directive = Directive(prompt="Test directive")
        trait = DirectiveTrait(directive)
        agent.add_trait(trait)

        assert trait._agent == agent

    def test_add_duplicate_trait_raises(self, mock_logger):
        config = AgentConfig(name="test")
        agent = Agent(lg=mock_logger, config=config)

        directive = Directive(prompt="Test directive")
        agent.add_trait(DirectiveTrait(directive))

        with pytest.raises(ValueError, match="already added"):
            agent.add_trait(DirectiveTrait(directive))

    def test_get_trait_returns_none_if_not_added(self, mock_logger):
        config = AgentConfig(name="test")
        agent = Agent(lg=mock_logger, config=config)

        assert agent.get_trait(DirectiveTrait) is None

    def test_has_trait_returns_false_if_not_added(self, mock_logger):
        config = AgentConfig(name="test")
        agent = Agent(lg=mock_logger, config=config)

        assert not agent.has_trait(DirectiveTrait)

    def test_complete_with_directive_trait(self, mock_logger):
        from llm_agent.traits.llm import LLMTrait

        config = AgentConfig(name="test", fact_injection="none")
        agent = Agent(lg=mock_logger, config=config)

        # Add mock LLMTrait
        mock_llm_trait = MagicMock(spec=LLMTrait)
        mock_llm_trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent._traits[LLMTrait] = mock_llm_trait

        # Add DirectiveTrait
        directive = Directive(prompt="You are a helpful assistant. Be friendly.")
        agent.add_trait(DirectiveTrait(directive))

        agent.complete("Hello")

        # Check that directive was injected into prompt
        call_args = mock_llm_trait.complete.call_args
        messages = call_args.kwargs["messages"]
        system_prompt = messages[0].content

        assert "You are a helpful assistant." in system_prompt
        assert "Be friendly." in system_prompt
