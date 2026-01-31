"""Tests for Agent."""

from unittest.mock import MagicMock

import pytest

from llm_agent import Agent, AgentConfig, CompletionResult


pytestmark = pytest.mark.unit


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


class TestAgentLifecycle:
    """Tests for Agent lifecycle."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    def test_init(self, mock_logger):
        config = AgentConfig(name="test-agent")
        agent = Agent(lg=mock_logger, config=config)

        assert agent.name == "test-agent"
        assert agent.config == config

    def test_start_calls_trait_on_start(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        mock_trait = MagicMock()
        agent._traits[type(mock_trait)] = mock_trait

        agent.start()

        mock_trait.on_start.assert_called_once()

    def test_stop_calls_trait_on_stop(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        mock_trait = MagicMock()
        agent._traits[type(mock_trait)] = mock_trait
        agent._started = True

        agent.stop()

        mock_trait.on_stop.assert_called_once()


class TestAgentTraits:
    """Tests for Agent trait management."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    def test_add_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        mock_trait = MagicMock()

        agent.add_trait(mock_trait)

        mock_trait.attach.assert_called_once_with(agent)
        assert agent.has_trait(type(mock_trait))

    def test_add_duplicate_trait_raises(self, mock_logger):
        from llm_agent.core.traits.llm import LLMConfig, LLMTrait

        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        trait1 = LLMTrait(LLMConfig())
        agent.add_trait(trait1)

        with pytest.raises(ValueError, match="already added"):
            trait2 = LLMTrait(LLMConfig())
            agent.add_trait(trait2)

    def test_get_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        mock_trait = MagicMock()
        agent.add_trait(mock_trait)

        result = agent.get_trait(type(mock_trait))

        assert result is mock_trait

    def test_get_trait_not_found(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        result = agent.get_trait(MagicMock)

        assert result is None

    def test_require_trait_found(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        mock_trait = MagicMock()
        agent.add_trait(mock_trait)

        result = agent.require_trait(type(mock_trait))

        assert result is mock_trait

    def test_require_trait_not_found_raises(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        with pytest.raises(RuntimeError, match="required but not attached"):
            agent.require_trait(MagicMock)


class TestAgentComplete:
    """Tests for Agent.complete()."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        """Create mock LLMTrait."""
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Hello!",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        return trait

    @pytest.fixture
    def agent_with_llm(self, mock_logger, mock_llm_trait):
        """Create agent with LLMTrait attached."""
        from llm_agent.core.traits.llm import LLMTrait

        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        agent._traits[LLMTrait] = mock_llm_trait
        return agent

    def test_complete_requires_llm_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        with pytest.raises(RuntimeError, match="LLMTrait required"):
            agent.complete("Hello")

    def test_complete_uses_default_prompt(self, agent_with_llm, mock_llm_trait):
        agent_with_llm._config = AgentConfig(name="test", default_prompt="Be helpful.")

        result = agent_with_llm.complete("Hi there")

        assert result.content == "Hello!"
        mock_llm_trait.complete.assert_called_once()
        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful."
        assert messages[1].role == "user"
        assert messages[1].content == "Hi there"

    def test_complete_with_custom_prompt(self, agent_with_llm, mock_llm_trait):
        agent_with_llm.complete("Query", system_prompt="Custom prompt.")

        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        assert messages[0].content == "Custom prompt."

    def test_complete_uses_config_model(self, agent_with_llm, mock_llm_trait):
        agent_with_llm._config = AgentConfig(name="test", model="gpt-4")

        agent_with_llm.complete("Query")

        assert mock_llm_trait.complete.call_args.kwargs["model"] == "gpt-4"

    def test_complete_returns_result(self, agent_with_llm, mock_llm_trait):
        expected = CompletionResult(
            id="resp-123",
            content="The answer is 42.",
            model="gpt-4",
            tokens_used=25,
            latency_ms=150,
        )
        mock_llm_trait.complete.return_value = expected

        result = agent_with_llm.complete("What is the answer?")

        assert result == expected


class TestAgentMemory:
    """Tests for Agent memory operations (remember, forget, recall)."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_learn_trait(self):
        """Create mock LearnTrait."""
        from llm_agent.core.traits.learn import LearnTrait

        trait = MagicMock(spec=LearnTrait)
        trait.remember.return_value = 42
        trait.has_embedder = True
        return trait

    @pytest.fixture
    def agent_with_learn(self, mock_logger, mock_learn_trait):
        """Create agent with LearnTrait attached."""
        from llm_agent.core.traits.learn import LearnTrait

        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))
        agent._traits[LearnTrait] = mock_learn_trait
        return agent

    def test_remember_requires_learn_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        with pytest.raises(RuntimeError, match="LearnTrait required"):
            agent.remember("fact")

    def test_remember_delegates_to_trait(self, agent_with_learn, mock_learn_trait):
        fact_id = agent_with_learn.remember("User prefers Python", category="preferences")

        assert fact_id == 42
        mock_learn_trait.remember.assert_called_once_with(
            "User prefers Python", category="preferences"
        )

    def test_forget_requires_learn_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        with pytest.raises(RuntimeError, match="LearnTrait required"):
            agent.forget(42)

    def test_forget_delegates_to_trait(self, agent_with_learn, mock_learn_trait):
        agent_with_learn.forget(42)

        mock_learn_trait.forget.assert_called_once_with(42)

    def test_recall_requires_learn_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        with pytest.raises(RuntimeError, match="LearnTrait required"):
            agent.recall("query")

    def test_recall_delegates_to_trait(self, agent_with_learn, mock_learn_trait):
        mock_scored_fact = MagicMock()
        mock_learn_trait.recall.return_value = [mock_scored_fact]

        results = agent_with_learn.recall("programming languages")

        assert len(results) == 1
        mock_learn_trait.recall.assert_called_once()

    def test_recall_uses_config_defaults(self, agent_with_learn, mock_learn_trait):
        agent_with_learn._config = AgentConfig(name="test", rag_top_k=10, rag_min_similarity=0.5)
        mock_learn_trait.recall.return_value = []

        agent_with_learn.recall("query")

        mock_learn_trait.recall.assert_called_once_with(
            query="query",
            top_k=10,
            min_similarity=0.5,
            categories=None,
        )

    def test_recall_with_custom_params(self, agent_with_learn, mock_learn_trait):
        mock_learn_trait.recall.return_value = []

        agent_with_learn.recall("query", top_k=20, min_similarity=0.8, categories=["prefs"])

        mock_learn_trait.recall.assert_called_once_with(
            query="query",
            top_k=20,
            min_similarity=0.8,
            categories=["prefs"],
        )


class TestAgentFeedback:
    """Tests for Agent feedback operations."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        """Create mock LLMTrait."""
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        return trait

    @pytest.fixture
    def mock_learn_trait(self):
        """Create mock LearnTrait."""
        from llm_agent.core.traits.learn import LearnTrait

        trait = MagicMock(spec=LearnTrait)
        return trait

    @pytest.fixture
    def agent_with_traits(self, mock_logger, mock_llm_trait, mock_learn_trait):
        """Create agent with LLMTrait and LearnTrait attached."""
        from llm_agent.core.traits.learn import LearnTrait
        from llm_agent.core.traits.llm import LLMTrait

        # Use fact_injection="none" to avoid needing to mock build_prompt
        config = AgentConfig(name="test", fact_injection="none")
        agent = Agent(lg=mock_logger, config=config)
        agent._traits[LLMTrait] = mock_llm_trait
        agent._traits[LearnTrait] = mock_learn_trait
        return agent

    def test_feedback_requires_learn_trait(self, mock_logger):
        agent = Agent(lg=mock_logger, config=AgentConfig(name="test"))

        with pytest.raises(RuntimeError, match="LearnTrait required"):
            agent.feedback("resp-1", "positive")

    def test_feedback_unknown_response_raises(self, agent_with_traits):
        with pytest.raises(ValueError, match="Unknown response_id"):
            agent_with_traits.feedback("nonexistent", "positive")

    def test_feedback_positive(self, agent_with_traits, mock_llm_trait, mock_learn_trait):
        # Setup: complete a request to track the response
        mock_llm_trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Good response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent_with_traits.complete("What is Python?")

        # Act: provide feedback
        agent_with_traits.feedback("resp-1", "positive")

        # Assert
        mock_learn_trait.record_feedback.assert_called_once()
        call_kwargs = mock_learn_trait.record_feedback.call_args.kwargs
        assert call_kwargs["content"] == "Good response"
        assert call_kwargs["signal"] == "positive"

    def test_feedback_negative_with_correction(
        self, agent_with_traits, mock_llm_trait, mock_learn_trait
    ):
        mock_llm_trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Wrong answer",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )

        agent_with_traits.complete("What is 2+2?")
        agent_with_traits.feedback("resp-1", "negative", correction="The answer is 4")

        # Should record feedback
        mock_learn_trait.record_feedback.assert_called_once()

        # Should also create preference pair
        mock_learn_trait.record_preference.assert_called_once()
        pref_kwargs = mock_learn_trait.record_preference.call_args.kwargs
        assert pref_kwargs["chosen"] == "The answer is 4"
        assert pref_kwargs["rejected"] == "Wrong answer"

    def test_feedback_cleans_up_response_context(
        self, agent_with_traits, mock_llm_trait, mock_learn_trait
    ):
        """Verify feedback removes response context to prevent memory leaks."""
        mock_llm_trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )

        agent_with_traits.complete("Query")
        agent_with_traits.feedback("resp-1", "positive")

        # Second feedback on same response should fail
        with pytest.raises(ValueError, match="Unknown response_id"):
            agent_with_traits.feedback("resp-1", "positive")

    def test_response_context_evicts_oldest_when_full(
        self, agent_with_traits, mock_llm_trait, mock_learn_trait
    ):
        """Verify oldest response contexts are evicted when limit is reached."""
        agent_with_traits._config = AgentConfig(
            name="test", max_tracked_responses=3, fact_injection="none"
        )

        # Generate 4 completions with a limit of 3
        for i in range(4):
            mock_llm_trait.complete.return_value = CompletionResult(
                id=f"resp-{i}",
                content=f"Response {i}",
                model="default",
                tokens_used=10,
                latency_ms=100,
            )
            agent_with_traits.complete(f"Query {i}")

        # resp-0 should have been evicted (oldest)
        with pytest.raises(ValueError, match="Unknown response_id"):
            agent_with_traits.feedback("resp-0", "positive")

        # resp-1, resp-2, resp-3 should still be tracked
        agent_with_traits.feedback("resp-1", "positive")
        agent_with_traits.feedback("resp-2", "positive")
        agent_with_traits.feedback("resp-3", "positive")


class TestAgentFactInjection:
    """Tests for fact injection in prompts."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_llm_trait(self):
        """Create mock LLMTrait."""
        from llm_agent.core.traits.llm import LLMTrait

        trait = MagicMock(spec=LLMTrait)
        trait.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        return trait

    @pytest.fixture
    def mock_learn_trait(self):
        """Create mock LearnTrait."""
        from llm_agent.core.traits.learn import LearnTrait

        trait = MagicMock(spec=LearnTrait)
        trait.has_embedder = True
        trait.build_prompt.side_effect = lambda base_prompt, **_: base_prompt + "\n[FACTS]"
        trait.build_prompt_rag.side_effect = lambda base_prompt, **_: base_prompt + "\n[RAG_FACTS]"
        return trait

    def test_no_fact_injection(self, mock_logger, mock_llm_trait):
        """Verify fact_injection='none' skips fact injection."""
        from llm_agent.core.traits.llm import LLMTrait

        config = AgentConfig(name="test", fact_injection="none", default_prompt="Base.")
        agent = Agent(lg=mock_logger, config=config)
        agent._traits[LLMTrait] = mock_llm_trait

        agent.complete("Query")

        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        assert messages[0].content == "Base."

    def test_all_fact_injection(self, mock_logger, mock_llm_trait, mock_learn_trait):
        """Verify fact_injection='all' uses build_prompt."""
        from llm_agent.core.traits.learn import LearnTrait
        from llm_agent.core.traits.llm import LLMTrait

        config = AgentConfig(name="test", fact_injection="all", default_prompt="Base.")
        agent = Agent(lg=mock_logger, config=config)
        agent._traits[LLMTrait] = mock_llm_trait
        agent._traits[LearnTrait] = mock_learn_trait

        agent.complete("Query")

        mock_learn_trait.build_prompt.assert_called_once()
        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        assert "[FACTS]" in messages[0].content

    def test_rag_fact_injection(self, mock_logger, mock_llm_trait, mock_learn_trait):
        """Verify fact_injection='rag' uses build_prompt_rag."""
        from llm_agent.core.traits.learn import LearnTrait
        from llm_agent.core.traits.llm import LLMTrait

        config = AgentConfig(name="test", fact_injection="rag", default_prompt="Base.")
        agent = Agent(lg=mock_logger, config=config)
        agent._traits[LLMTrait] = mock_llm_trait
        agent._traits[LearnTrait] = mock_learn_trait

        agent.complete("Query")

        mock_learn_trait.build_prompt_rag.assert_called_once()
        messages = mock_llm_trait.complete.call_args.kwargs["messages"]
        assert "[RAG_FACTS]" in messages[0].content

    def test_rag_without_embedder_raises(self, mock_logger, mock_llm_trait, mock_learn_trait):
        """Verify RAG mode requires embedder."""
        from llm_agent.core.traits.learn import LearnTrait
        from llm_agent.core.traits.llm import LLMTrait

        config = AgentConfig(name="test", fact_injection="rag")
        agent = Agent(lg=mock_logger, config=config)
        agent._traits[LLMTrait] = mock_llm_trait
        mock_learn_trait.has_embedder = False
        agent._traits[LearnTrait] = mock_learn_trait

        with pytest.raises(ValueError, match="requires embedder"):
            agent.complete("Query")
