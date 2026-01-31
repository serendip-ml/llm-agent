"""Learn trait for agent memory and feedback capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from llm_infer.client import ChatResponse, LLMClient
from llm_learn import LearnClient
from llm_learn.collection import ScoredFact
from llm_learn.core import Database
from llm_learn.inference import ContextBuilder, Embedder

from llm_agent.core.llm.types import CompletionResult
from llm_agent.core.traits.llm import LLMConfig


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent


@dataclass
class LearnConfig:
    """Configuration for Learn trait.

    Attributes:
        profile_id: Profile ID for scoping all learn operations.
        llm: LLM configuration for learned completions.
        db_config_path: Path to database config file (default: etc/infra.yaml).
        db_key: Database config key (default: main).
        embedder_url: URL for embedding service (None = no RAG).
        embedder_model: Model name for embeddings.
        embedder_timeout: Embedder timeout in seconds.
    """

    profile_id: int
    llm: LLMConfig
    db_config_path: str = "etc/infra.yaml"
    db_key: str = "main"
    embedder_url: str | None = None
    embedder_model: str = "default"
    embedder_timeout: float = 30.0


@dataclass
class LearnTrait:
    """Learn capability trait for memory, feedback, and learned completions.

    Wraps llm_learn.LearnClient, LLMClient, and Embedder to provide
    learning-enabled completions with memory and feedback.

    Capabilities:
        - complete(): Generate completions with automatic fact injection
        - remember(): Store facts about the user
        - recall(): Search facts by semantic similarity
        - record_feedback(): Record feedback on responses
        - record_preference(): Record preference pairs for training

    Example:
        from llm_agent.core.traits import LearnTrait, LearnConfig, LLMConfig

        agent = Agent(lg, config)
        agent.add_trait(LearnTrait(LearnConfig(
            profile_id=123,
            llm=LLMConfig(base_url="http://localhost:8000/v1"),
            embedder_url="http://localhost:8001/v1",
        )))
        agent.start()

        # Learned completion with fact injection
        result = agent.get_trait(LearnTrait).complete("What do I prefer?")

        # Store a fact
        agent.get_trait(LearnTrait).remember("User prefers concise answers")

    Lifecycle:
        - attach(): Stores agent reference
        - on_start(): Creates LearnClient, LLMClient, Embedder, ContextBuilder
        - on_stop(): Closes LLMClient and Embedder
    """

    config: LearnConfig

    _agent: Agent | None = field(default=None, repr=False, compare=False)
    _database: Database | None = field(default=None, repr=False, compare=False)
    _learn: LearnClient | None = field(default=None, repr=False, compare=False)
    _embedder: Embedder | None = field(default=None, repr=False, compare=False)
    _context: ContextBuilder | None = field(default=None, repr=False, compare=False)
    _client: LLMClient | None = field(default=None, repr=False, compare=False)

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent.

        Args:
            agent: The agent this trait is attached to.
        """
        self._agent = agent

    def on_start(self) -> None:
        """Create learn client, LLM client, and embedder on agent start."""
        # Create database and learn client
        self._database = Database.from_config(self.config.db_config_path, self.config.db_key)
        self._learn = LearnClient(
            profile_id=self.config.profile_id,
            database=self._database,
        )

        # Create context builder
        self._context = ContextBuilder(self._learn.facts)

        # Create LLM client for completions
        self._client = LLMClient.openai(
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            timeout=self.config.llm.timeout,
        )

        # Create embedder if URL provided
        if self.config.embedder_url:
            self._embedder = Embedder(
                base_url=self.config.embedder_url,
                model=self.config.embedder_model,
                timeout=self.config.embedder_timeout,
            )

    def on_stop(self) -> None:
        """Close LLM client and embedder on agent stop."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None
        self._learn = None
        self._context = None
        self._database = None

    @property
    def learn(self) -> LearnClient:
        """Access the learn client.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._learn is None:
            raise RuntimeError("LearnTrait not started - ensure agent.start() was called")
        return self._learn

    @property
    def embedder(self) -> Embedder | None:
        """Access the embedder (None if not configured)."""
        return self._embedder

    @property
    def has_embedder(self) -> bool:
        """Check if embedder is available for RAG."""
        return self._embedder is not None

    @property
    def client(self) -> LLMClient:
        """Access the LLM client.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._client is None:
            raise RuntimeError("LearnTrait not started - ensure agent.start() was called")
        return self._client

    # =========================================================================
    # Completions
    # =========================================================================

    def complete(
        self,
        query: str,
        system_prompt: str = "",
        include_facts: bool = True,
        rag: bool = False,
        rag_top_k: int = 10,
        rag_min_similarity: float = 0.5,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a completion with optional fact injection.

        Args:
            query: User query.
            system_prompt: Base system prompt.
            include_facts: Whether to inject facts into prompt.
            rag: Use RAG-based fact retrieval (requires embedder).
            rag_top_k: Max facts for RAG.
            rag_min_similarity: Min similarity for RAG.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            CompletionResult with response and metadata.
        """
        prompt = self._build_completion_prompt(
            system_prompt, query, include_facts, rag, rag_top_k, rag_min_similarity
        )

        response = self.client.chat_full(
            messages=[{"role": "user", "content": query}],
            system=prompt if prompt else None,
            model=self.config.llm.model,
            temperature=temperature if temperature is not None else self.config.llm.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.llm.max_tokens,
        )

        return self._response_to_result(response)

    def _build_completion_prompt(
        self,
        base_prompt: str,
        query: str,
        include_facts: bool,
        rag: bool,
        rag_top_k: int,
        rag_min_similarity: float,
    ) -> str:
        """Build system prompt with optional fact injection."""
        if not include_facts:
            return base_prompt

        if rag:
            if self._embedder is None:
                raise ValueError("RAG requires embedder - configure embedder_url in LearnConfig")
            return self.build_prompt_rag(
                base_prompt=base_prompt,
                query=query,
                top_k=rag_top_k,
                min_similarity=rag_min_similarity,
            )

        return self.build_prompt(base_prompt=base_prompt)

    def _response_to_result(self, response: ChatResponse) -> CompletionResult:
        """Convert LLM response to CompletionResult."""
        import uuid

        tokens_used = 0
        if response.usage:
            tokens_used = response.usage.total_tokens or (
                (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
            )

        return CompletionResult(
            id=str(uuid.uuid4()),
            content=response.content,
            model=response.model or self.config.llm.model,
            tokens_used=tokens_used,
            latency_ms=0,
            tool_calls=None,
        )

    # =========================================================================
    # Memory operations
    # =========================================================================

    def remember(
        self,
        fact: str,
        category: str = "general",
        source: Literal["user", "inferred", "conversation", "system"] = "user",
        confidence: float = 1.0,
    ) -> int:
        """Store a fact about the user.

        If embedder is configured, the fact is also embedded for semantic search.

        Args:
            fact: The fact to store.
            category: Category for organization.
            source: How the fact was obtained.
            confidence: Confidence level 0.0-1.0.

        Returns:
            Fact ID.
        """
        fact_id = self.learn.facts.add(
            fact, category=category, source=source, confidence=confidence
        )

        # Embed for semantic search if embedder available
        if self._embedder is not None:
            embedding = self._embedder.embed(fact)
            self.learn.facts.set_embedding(
                fact_id=fact_id,
                embedding=embedding.embedding,
                model_name=self._embedder.model,
            )

        return fact_id

    def forget(self, fact_id: int) -> None:
        """Remove a stored fact.

        Args:
            fact_id: ID of the fact to remove.
        """
        self.learn.facts.delete(fact_id)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
        categories: list[str] | None = None,
    ) -> list[ScoredFact]:
        """Search facts by semantic similarity.

        Args:
            query: Text to search for similar facts.
            top_k: Maximum results.
            min_similarity: Minimum similarity threshold (0-1).
            categories: Filter to these categories (None = all).

        Returns:
            List of ScoredFact sorted by similarity.

        Raises:
            ValueError: If embedder not configured.
        """
        if self._embedder is None:
            raise ValueError("recall() requires embedder - configure embedder_url in LearnConfig")

        embedding = self._embedder.embed(query)
        return self.learn.facts.search_similar(
            embedding=embedding.embedding,
            model_name=self._embedder.model,
            top_k=top_k,
            min_similarity=min_similarity,
            categories=categories,
        )

    # =========================================================================
    # Context building
    # =========================================================================

    def build_prompt(
        self,
        base_prompt: str,
        max_facts: int = 100,
        categories: list[str] | None = None,
    ) -> str:
        """Build system prompt with injected facts (all mode).

        Args:
            base_prompt: Base system prompt.
            max_facts: Maximum facts to include.
            categories: Filter to these categories.

        Returns:
            System prompt with facts appended.
        """
        if self._context is None:
            raise RuntimeError("LearnTrait not started")

        return self._context.build_system_prompt(
            base_prompt=base_prompt,
            categories=categories,
            max_facts=max_facts,
        )

    def build_prompt_rag(
        self,
        base_prompt: str,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> str:
        """Build system prompt with RAG-injected facts.

        Args:
            base_prompt: Base system prompt.
            query: Query for semantic fact retrieval.
            top_k: Maximum facts to retrieve.
            min_similarity: Minimum similarity threshold.

        Returns:
            System prompt with relevant facts appended.

        Raises:
            ValueError: If embedder not configured.
        """
        if self._context is None:
            raise RuntimeError("LearnTrait not started")
        if self._embedder is None:
            raise ValueError("build_prompt_rag() requires embedder")

        scored_facts = self.recall(query, top_k=top_k, min_similarity=min_similarity)
        facts = [sf.fact for sf in scored_facts]

        return self._context.build_system_prompt_from_facts(
            base_prompt=base_prompt,
            facts=facts,
        )

    # =========================================================================
    # Feedback
    # =========================================================================

    def record_feedback(
        self,
        content: str,
        signal: Literal["positive", "negative"],
        context: dict[str, Any] | None = None,
    ) -> int:
        """Record feedback on content.

        Args:
            content: The content that received feedback.
            signal: Whether it was good or bad.
            context: Additional context (query, model, etc.).

        Returns:
            Feedback ID.
        """
        return self.learn.feedback.record(
            content_text=content,
            signal=signal,
            context=context,
        )

    def record_preference(
        self,
        context: str,
        chosen: str,
        rejected: str,
    ) -> int:
        """Record a preference pair (chosen over rejected).

        Args:
            context: The context/prompt for the pair.
            chosen: The preferred response.
            rejected: The rejected response.

        Returns:
            Preference ID.
        """
        return self.learn.preferences.record(
            context=context,
            chosen=chosen,
            rejected=rejected,
        )
