"""Tests for Agent."""

from unittest.mock import MagicMock

from llm_agent import Agent, AgentConfig, CompletionResult


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        config = AgentConfig(name="test-agent")

        assert config.name == "test-agent"
        assert config.default_prompt == "You are a helpful assistant."
        assert config.model == "default"
        assert config.fact_injection == "all"
        assert config.max_facts == 20
        assert config.rag_top_k == 5
        assert config.rag_min_similarity == 0.3

    def test_custom_values(self):
        config = AgentConfig(
            name="custom",
            default_prompt="Be concise.",
            model="gpt-4",
            fact_injection="rag",
            max_facts=10,
        )

        assert config.name == "custom"
        assert config.default_prompt == "Be concise."
        assert config.model == "gpt-4"
        assert config.fact_injection == "rag"
        assert config.max_facts == 10


class TestAgent:
    """Tests for Agent."""

    def test_init(self):
        config = AgentConfig(name="test-agent")
        llm = MagicMock()

        agent = Agent(config=config, llm=llm)

        assert agent.name == "test-agent"
        assert agent.config == config

    def test_complete_uses_default_prompt(self):
        config = AgentConfig(name="test", default_prompt="Be helpful.")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Hello!",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(config=config, llm=llm)

        result = agent.complete("Hi there")

        assert result.content == "Hello!"
        llm.complete.assert_called_once()

        # Check messages passed to LLM
        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful."
        assert messages[1].role == "user"
        assert messages[1].content == "Hi there"

    def test_complete_with_custom_prompt(self):
        config = AgentConfig(name="test", default_prompt="Default prompt.")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(config=config, llm=llm)

        agent.complete("Query", system_prompt="Custom prompt.")

        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0].content == "Custom prompt."

    def test_complete_uses_config_model(self):
        config = AgentConfig(name="test", model="gpt-4")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="gpt-4",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(config=config, llm=llm)

        agent.complete("Query")

        call_args = llm.complete.call_args
        assert call_args.kwargs["model"] == "gpt-4"

    def test_complete_returns_result(self):
        config = AgentConfig(name="test")
        llm = MagicMock()
        expected = CompletionResult(
            id="resp-123",
            content="The answer is 42.",
            model="gpt-4",
            tokens_used=25,
            latency_ms=150,
        )
        llm.complete.return_value = expected
        agent = Agent(config=config, llm=llm)

        result = agent.complete("What is the answer?")

        assert result == expected
        assert result.id == "resp-123"
        assert result.content == "The answer is 42."
