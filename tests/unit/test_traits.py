"""Tests for traits."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agent import Agent, AgentConfig, CompletionResult, Directive, DirectiveTrait


pytestmark = pytest.mark.unit


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
    def mock_learn(self):
        """Create mock LearnClient."""
        learn = MagicMock()
        learn.facts = MagicMock()
        learn.facts.list_active.return_value = []
        learn.feedback = MagicMock()
        learn.preferences = MagicMock()
        return learn

    @pytest.fixture
    def mock_context_builder(self):
        """Patch ContextBuilder to avoid database calls."""
        with patch("llm_agent.agent.ContextBuilder") as mock_cls:
            mock_builder = MagicMock()
            mock_builder.build_system_prompt.side_effect = lambda base_prompt, **_: base_prompt
            mock_cls.return_value = mock_builder
            yield mock_cls

    def test_add_trait(self, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(config=config, llm=llm, learn=mock_learn)

        directive = Directive(prompt="Test directive")
        trait = DirectiveTrait(directive)
        agent.add_trait(trait)

        assert agent.has_trait(DirectiveTrait)
        assert agent.get_trait(DirectiveTrait) == trait

    def test_add_trait_attaches(self, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(config=config, llm=llm, learn=mock_learn)

        directive = Directive(prompt="Test directive")
        trait = DirectiveTrait(directive)
        agent.add_trait(trait)

        assert trait._agent == agent

    def test_add_duplicate_trait_raises(self, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(config=config, llm=llm, learn=mock_learn)

        directive = Directive(prompt="Test directive")
        agent.add_trait(DirectiveTrait(directive))

        with pytest.raises(ValueError, match="already added"):
            agent.add_trait(DirectiveTrait(directive))

    def test_get_trait_returns_none_if_not_added(self, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(config=config, llm=llm, learn=mock_learn)

        assert agent.get_trait(DirectiveTrait) is None

    def test_has_trait_returns_false_if_not_added(self, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(config=config, llm=llm, learn=mock_learn)

        assert not agent.has_trait(DirectiveTrait)

    def test_complete_with_directive_trait(self, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(config=config, llm=llm, learn=mock_learn)

        directive = Directive(prompt="You are a helpful assistant. Be friendly.")
        agent.add_trait(DirectiveTrait(directive))

        agent.complete("Hello")

        # Check that directive was injected into prompt
        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        system_prompt = messages[0].content

        assert "You are a helpful assistant." in system_prompt
        assert "Be friendly." in system_prompt
