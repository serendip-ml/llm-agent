"""Tests for Agent."""

from unittest.mock import MagicMock, patch

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


class TestAgent:
    """Tests for Agent."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_learn(self):
        """Create mock LearnClient."""
        learn = MagicMock()
        learn.facts = MagicMock()
        learn.facts.list_active.return_value = []  # No facts by default
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

    def test_init(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test-agent")
        llm = MagicMock()

        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        assert agent.name == "test-agent"
        assert agent.config == config

    def test_complete_uses_default_prompt(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", default_prompt="Be helpful.")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Hello!",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

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

    def test_complete_with_custom_prompt(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", default_prompt="Default prompt.")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.complete("Query", system_prompt="Custom prompt.")

        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0].content == "Custom prompt."

    def test_complete_uses_config_model(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", model="gpt-4")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="gpt-4",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.complete("Query")

        call_args = llm.complete.call_args
        assert call_args.kwargs["model"] == "gpt-4"

    def test_complete_returns_result(self, mock_logger, mock_learn, mock_context_builder):
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
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        result = agent.complete("What is the answer?")

        assert result == expected
        assert result.id == "resp-123"
        assert result.content == "The answer is 42."

    def test_complete_no_fact_injection(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.complete("Query", system_prompt="Base prompt.")

        # ContextBuilder should not be called when fact_injection is "none"
        mock_context_builder.return_value.build_system_prompt.assert_not_called()

        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0].content == "Base prompt."

    def test_remember_delegates_to_learn(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        mock_learn.facts.add.return_value = 42
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        fact_id = agent.remember("User prefers Python", category="preferences")

        assert fact_id == 42
        mock_learn.facts.add.assert_called_once_with("User prefers Python", category="preferences")

    def test_forget_delegates_to_learn(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.forget(42)

        mock_learn.facts.delete.assert_called_once_with(42)

    def test_feedback_positive(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Good response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        # First complete a request to track the response
        agent.complete("What is Python?")

        # Then provide feedback
        agent.feedback("resp-1", "positive")

        mock_learn.feedback.record.assert_called_once()
        call_kwargs = mock_learn.feedback.record.call_args.kwargs
        assert call_kwargs["content_text"] == "Good response"
        assert call_kwargs["signal"] == "positive"

    def test_feedback_negative_with_correction(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Wrong answer",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.complete("What is 2+2?")
        agent.feedback("resp-1", "negative", correction="The answer is 4")

        # Should record feedback
        mock_learn.feedback.record.assert_called_once()

        # Should also create preference pair
        mock_learn.preferences.record.assert_called_once()
        pref_kwargs = mock_learn.preferences.record.call_args.kwargs
        assert pref_kwargs["chosen"] == "The answer is 4"
        assert pref_kwargs["rejected"] == "Wrong answer"

    def test_feedback_unknown_response_raises(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        with pytest.raises(ValueError, match="Unknown response_id"):
            agent.feedback("nonexistent", "positive")

    def test_load_adapter(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.load_adapter("/path/to/adapter")

        llm.load_adapter.assert_called_once_with("/path/to/adapter")

    def test_unload_adapter(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test")
        llm = MagicMock()
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.unload_adapter()

        llm.unload_adapter.assert_called_once()

    def test_rag_mode_without_embedder_raises(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", fact_injection="rag")
        llm = MagicMock()

        with pytest.raises(ValueError, match="fact_injection='rag' requires an embedder"):
            Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

    def test_rag_mode_with_embedder_succeeds(self, mock_logger, mock_learn, mock_context_builder):
        config = AgentConfig(name="test", fact_injection="rag")
        llm = MagicMock()
        embedder = MagicMock()

        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=embedder)

        assert agent.name == "test"

    def test_feedback_cleans_up_response_context(
        self, mock_logger, mock_learn, mock_context_builder
    ):
        """Verify feedback removes response context to prevent memory leaks."""
        config = AgentConfig(name="test")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        agent.complete("Query")
        agent.feedback("resp-1", "positive")

        # Second feedback on same response should fail (context was cleaned up)
        with pytest.raises(ValueError, match="Unknown response_id"):
            agent.feedback("resp-1", "positive")

    def test_response_context_evicts_oldest_when_full(
        self, mock_logger, mock_learn, mock_context_builder
    ):
        """Verify oldest response contexts are evicted when limit is reached."""
        config = AgentConfig(name="test", max_tracked_responses=3)
        llm = MagicMock()
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        # Generate 4 completions with a limit of 3
        for i in range(4):
            llm.complete.return_value = CompletionResult(
                id=f"resp-{i}",
                content=f"Response {i}",
                model="default",
                tokens_used=10,
                latency_ms=100,
            )
            agent.complete(f"Query {i}")

        # resp-0 should have been evicted (oldest)
        with pytest.raises(ValueError, match="Unknown response_id"):
            agent.feedback("resp-0", "positive")

        # resp-1, resp-2, resp-3 should still be tracked
        agent.feedback("resp-1", "positive")
        agent.feedback("resp-2", "positive")
        agent.feedback("resp-3", "positive")


class TestAgentRAG:
    """Tests for RAG functionality."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

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
            mock_builder.build_system_prompt_from_facts.side_effect = (
                lambda base_prompt, **_: base_prompt
            )
            mock_cls.return_value = mock_builder
            yield mock_cls

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder with standard return values."""
        embedder = MagicMock()
        embedder.model = "test-embed-model"
        embedding_result = MagicMock()
        embedding_result.embedding = [0.1, 0.2, 0.3]
        embedder.embed.return_value = embedding_result
        return embedder

    def test_recall_returns_similar_facts(
        self, mock_logger, mock_learn, mock_context_builder, mock_embedder
    ):
        """Verify recall() performs semantic search and returns results."""
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()

        # Setup mock scored facts
        mock_fact = MagicMock()
        mock_fact.content = "User prefers Python"
        mock_scored_fact = MagicMock()
        mock_scored_fact.fact = mock_fact
        mock_scored_fact.similarity = 0.85
        mock_learn.facts.search_similar.return_value = [mock_scored_fact]

        agent = Agent(
            lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=mock_embedder
        )

        results = agent.recall("programming languages")

        assert len(results) == 1
        assert results[0].fact.content == "User prefers Python"
        mock_embedder.embed.assert_called_once_with("programming languages")
        mock_learn.facts.search_similar.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            model_name="test-embed-model",
            top_k=5,  # default rag_top_k
            min_similarity=0.3,  # default rag_min_similarity
            categories=None,
        )

    def test_recall_with_custom_parameters(
        self, mock_logger, mock_learn, mock_context_builder, mock_embedder
    ):
        """Verify recall() uses custom parameters when provided."""
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()
        mock_learn.facts.search_similar.return_value = []

        agent = Agent(
            lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=mock_embedder
        )

        agent.recall("query", top_k=10, min_similarity=0.5, categories=["preferences"])

        mock_learn.facts.search_similar.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            model_name="test-embed-model",
            top_k=10,
            min_similarity=0.5,
            categories=["preferences"],
        )

    def test_recall_without_embedder_raises(self, mock_logger, mock_learn, mock_context_builder):
        """Verify recall() raises error when embedder not configured."""
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()

        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        with pytest.raises(ValueError, match="recall\\(\\) requires an embedder"):
            agent.recall("query")

    def test_remember_embeds_when_embedder_available(
        self, mock_logger, mock_learn, mock_context_builder, mock_embedder
    ):
        """Verify remember() embeds facts when embedder is configured."""
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()
        mock_learn.facts.add.return_value = 42

        agent = Agent(
            lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=mock_embedder
        )

        fact_id = agent.remember("User prefers Python", category="preferences")

        assert fact_id == 42
        mock_learn.facts.add.assert_called_once_with("User prefers Python", category="preferences")
        mock_embedder.embed.assert_called_once_with("User prefers Python")
        mock_learn.facts.set_embedding.assert_called_once_with(
            fact_id=42,
            embedding=[0.1, 0.2, 0.3],
            model_name="test-embed-model",
        )

    def test_remember_without_embedder_skips_embedding(
        self, mock_logger, mock_learn, mock_context_builder
    ):
        """Verify remember() skips embedding when embedder not configured."""
        config = AgentConfig(name="test", fact_injection="none")
        llm = MagicMock()
        mock_learn.facts.add.return_value = 42

        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

        fact_id = agent.remember("User prefers Python")

        assert fact_id == 42
        mock_learn.facts.add.assert_called_once()
        mock_learn.facts.set_embedding.assert_not_called()

    def test_complete_rag_mode_uses_semantic_search(
        self, mock_logger, mock_learn, mock_context_builder, mock_embedder
    ):
        """Verify complete() in RAG mode performs semantic search."""
        config = AgentConfig(name="test", fact_injection="rag")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )

        # Setup mock scored facts
        mock_fact = MagicMock()
        mock_scored_fact = MagicMock()
        mock_scored_fact.fact = mock_fact
        mock_learn.facts.search_similar.return_value = [mock_scored_fact]

        agent = Agent(
            lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=mock_embedder
        )

        agent.complete("What programming language should I use?")

        # Verify embedding was created for the query
        mock_embedder.embed.assert_called_once_with("What programming language should I use?")

        # Verify semantic search was performed
        mock_learn.facts.search_similar.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            model_name="test-embed-model",
            top_k=5,
            min_similarity=0.3,
        )

        # Verify build_system_prompt_from_facts was called (not build_system_prompt)
        mock_context_builder.return_value.build_system_prompt_from_facts.assert_called_once()
        mock_context_builder.return_value.build_system_prompt.assert_not_called()

    def test_complete_all_mode_does_not_use_semantic_search(
        self, mock_logger, mock_learn, mock_context_builder, mock_embedder
    ):
        """Verify complete() in 'all' mode does not use semantic search."""
        config = AgentConfig(name="test", fact_injection="all")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-1",
            content="Response",
            model="default",
            tokens_used=10,
            latency_ms=100,
        )

        agent = Agent(
            lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=mock_embedder
        )

        agent.complete("Query")

        # Embedder should not be called for fact retrieval
        mock_embedder.embed.assert_not_called()
        mock_learn.facts.search_similar.assert_not_called()

        # build_system_prompt should be called (not from_facts)
        mock_context_builder.return_value.build_system_prompt.assert_called_once()
        mock_context_builder.return_value.build_system_prompt_from_facts.assert_not_called()
