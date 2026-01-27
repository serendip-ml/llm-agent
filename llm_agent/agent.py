"""Core agent implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

from llm_learn import LearnClient
from llm_learn.collection import ScoredFact
from llm_learn.inference import ContextBuilder, Embedder

from llm_agent.config import AgentConfig
from llm_agent.llm import CompletionResult, LLMBackend, Message
from llm_agent.traits.base import Trait


if TYPE_CHECKING:
    from llm_agent.protocol.base import Request, Response


TraitT = TypeVar("TraitT", bound=Trait)


class Agent:
    """Learning agent that improves through feedback.

    Coordinates LLM backend with llm-learn for facts, feedback, and preferences.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMBackend,
        learn: LearnClient,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            config: Agent configuration.
            llm: LLM backend for completions.
            learn: Learning client from llm-learn.
            embedder: Embedder for RAG (optional, enables semantic search).
        """
        # Validate RAG mode configuration
        if config.fact_injection == "rag" and embedder is None:
            raise ValueError(
                "fact_injection='rag' requires an embedder. "
                "Either provide an embedder or use fact_injection='all' or 'none'."
            )

        self._config = config
        self._llm = llm
        self._learn = learn
        self._embedder = embedder
        self._context = ContextBuilder(learn.facts)
        self._response_contexts: dict[str, _ResponseContext] = {}
        self._traits: dict[type[Trait], Trait] = {}

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    # === Traits ===

    def add_trait(self, trait: Trait) -> None:
        """Add a trait to this agent.

        Traits are attached immediately upon adding.

        Args:
            trait: The trait instance to add.

        Raises:
            ValueError: If a trait of this type is already added.
        """
        trait_type = type(trait)
        if trait_type in self._traits:
            raise ValueError(f"Trait {trait_type.__name__} already added")
        self._traits[trait_type] = trait
        trait.attach(self)

    def get_trait(self, trait_type: type[TraitT]) -> TraitT | None:
        """Get an attached trait by its type.

        Args:
            trait_type: The trait class to look up.

        Returns:
            The trait instance, or None if not attached.
        """
        return self._traits.get(trait_type)  # type: ignore[return-value]

    def has_trait(self, trait_type: type[Trait]) -> bool:
        """Check if a trait is attached.

        Args:
            trait_type: The trait class to check.

        Returns:
            True if the trait is attached.
        """
        return trait_type in self._traits

    # === Core operations ===

    def complete(
        self,
        query: str,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a response with context-augmented prompt.

        Facts are automatically injected based on config.fact_injection mode.

        Args:
            query: User input.
            system_prompt: System prompt (uses default if None).

        Returns:
            Completion result with response and metadata.
        """
        base_prompt = system_prompt or self._config.default_prompt

        # Build prompt with fact injection (pass query for RAG mode)
        prompt = self._build_prompt(base_prompt, query=query)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=query),
        ]

        result = self._llm.complete(
            messages=messages,
            model=self._config.model,
        )

        # Track response for feedback (bounded to prevent memory leaks)
        if len(self._response_contexts) >= self._config.max_tracked_responses:
            # Evict oldest entry (dicts maintain insertion order in Python 3.7+)
            oldest_key = next(iter(self._response_contexts))
            del self._response_contexts[oldest_key]
        self._response_contexts[result.id] = _ResponseContext(
            system_prompt=prompt,
            query=query,
            response=result.content,
            model=result.model,
        )

        return result

    def _build_prompt(self, base_prompt: str, query: str | None = None) -> str:
        """Build system prompt with directive and fact injection."""
        from llm_agent.traits.directive import DirectiveTrait

        prompt = base_prompt

        # Inject directive if DirectiveTrait is attached
        directive_trait = self.get_trait(DirectiveTrait)
        if directive_trait is not None:
            prompt = directive_trait.build_prompt(prompt)

        # Inject facts based on config
        if self._config.fact_injection == "rag":
            assert query is not None  # RAG mode only called from complete() which provides query
            prompt = self._inject_rag_facts(prompt, query)
        elif self._config.fact_injection == "all":
            prompt = self._context.build_system_prompt(
                base_prompt=prompt,
                max_facts=self._config.max_facts,
            )
        # "none" mode: no fact injection

        return prompt

    def _inject_rag_facts(self, prompt: str, query: str) -> str:
        """Inject semantically relevant facts into prompt using RAG."""
        assert self._embedder is not None  # Validated at __init__

        embedding = self._embedder.embed(query)
        scored_facts = self._learn.facts.search_similar(
            embedding=embedding.embedding,
            model_name=self._embedder.model,
            top_k=self._config.rag_top_k,
            min_similarity=self._config.rag_min_similarity,
        )
        facts = [sf.fact for sf in scored_facts]
        return self._context.build_system_prompt_from_facts(
            base_prompt=prompt,
            facts=facts,
        )

    def recall(
        self,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        categories: list[str] | None = None,
    ) -> list[ScoredFact]:
        """Search facts by semantic similarity to query.

        Args:
            query: Text to search for similar facts.
            top_k: Max results (defaults to config.rag_top_k).
            min_similarity: Minimum similarity threshold (defaults to config.rag_min_similarity).
            categories: Filter to these categories.

        Returns:
            List of ScoredFact sorted by similarity (highest first).

        Raises:
            ValueError: If embedder not configured.
        """
        if self._embedder is None:
            raise ValueError("recall() requires an embedder")

        embedding = self._embedder.embed(query)
        return self._learn.facts.search_similar(
            embedding=embedding.embedding,
            model_name=self._embedder.model,
            top_k=self._config.rag_top_k if top_k is None else top_k,
            min_similarity=self._config.rag_min_similarity
            if min_similarity is None
            else min_similarity,
            categories=categories,
        )

    # === Memory (delegates to llm-learn) ===

    def remember(self, fact: str, category: str = "general") -> int:
        """Store a fact about the user.

        If an embedder is configured, the fact is also embedded for semantic search.

        Args:
            fact: The fact to store.
            category: Category for organization (preferences, rules, background, etc.)

        Returns:
            Fact ID.
        """
        fact_id = self._learn.facts.add(fact, category=category)

        # Embed for semantic search if embedder available
        if self._embedder is not None:
            embedding = self._embedder.embed(fact)
            self._learn.facts.set_embedding(
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
        self._learn.facts.delete(fact_id)

    # === Feedback (delegates to llm-learn) ===

    def feedback(
        self,
        response_id: str,
        signal: Literal["positive", "negative"],
        correction: str | None = None,
    ) -> None:
        """Record feedback on a response.

        Args:
            response_id: ID from CompletionResult.
            signal: Whether response was good or bad.
            correction: If negative, the preferred response (creates preference pair).

        Raises:
            ValueError: If response_id is not found.
        """
        # Pop to remove after use - prevents unbounded memory growth
        ctx = self._response_contexts.pop(response_id, None)
        if ctx is None:
            raise ValueError(f"Unknown response_id: {response_id}")

        # Record feedback signal
        self._learn.feedback.record(
            content_text=ctx.response,
            signal=signal,
            context={"query": ctx.query, "model": ctx.model},
        )

        # If negative with correction, create preference pair
        if signal == "negative" and correction is not None:
            full_context = f"{ctx.system_prompt}\n\nUser: {ctx.query}"
            self._learn.preferences.record(
                context=full_context,
                chosen=correction,
                rejected=ctx.response,
            )

    # === HTTP Request Handling ===

    def handle_request(self, request: Request) -> Response:
        """Handle an HTTP protocol request.

        Dispatches to the appropriate handler based on message type.

        Args:
            request: Protocol request message.

        Returns:
            Protocol response message.
        """
        from llm_agent.protocol.v1 import (
            CompleteRequest,
            FeedbackRequest,
            ForgetRequest,
            HealthRequest,
            RecallRequest,
            RememberRequest,
        )

        handlers: dict[str, Callable[[Request], Response]] = {
            HealthRequest.message_type: self._handle_health,
            CompleteRequest.message_type: self._handle_complete,
            RememberRequest.message_type: self._handle_remember,
            ForgetRequest.message_type: self._handle_forget,
            RecallRequest.message_type: self._handle_recall,
            FeedbackRequest.message_type: self._handle_feedback,
        }

        handler = handlers.get(request.message_type)
        if handler is None:
            # Return generic error response
            from llm_agent.protocol.base import Response

            return Response(
                id=request.id,
                success=False,
                error=f"Unknown message type: {request.message_type}",
            )

        return handler(request)

    def _handle_health(self, request: Request) -> Response:
        """Handle health check request."""
        from llm_agent.protocol.v1 import HealthResponse

        return HealthResponse(id=request.id, status="ok", agent_name=self.name)

    def _handle_complete(self, request: Request) -> Response:
        """Handle completion request."""
        from llm_agent.protocol.v1 import CompleteRequest, CompleteResponse

        req = (
            request
            if isinstance(request, CompleteRequest)
            else CompleteRequest(**request.model_dump())
        )
        try:
            result = self.complete(query=req.query, system_prompt=req.system_prompt)
            return CompleteResponse(
                id=req.id,
                response_id=result.id,
                content=result.content,
                model=result.model,
                tokens_used=result.tokens_used,
            )
        except Exception as e:
            return CompleteResponse(
                id=req.id,
                success=False,
                error=str(e),
                response_id="",
                content="",
                model="",
                tokens_used=0,
            )

    def _handle_remember(self, request: Request) -> Response:
        """Handle remember request."""
        from llm_agent.protocol.v1 import RememberRequest, RememberResponse

        req = (
            request
            if isinstance(request, RememberRequest)
            else RememberRequest(**request.model_dump())
        )
        try:
            fact_id = self.remember(fact=req.fact, category=req.category)
            return RememberResponse(id=req.id, fact_id=fact_id)
        except Exception as e:
            return RememberResponse(id=req.id, success=False, error=str(e), fact_id=-1)

    def _handle_forget(self, request: Request) -> Response:
        """Handle forget request."""
        from llm_agent.protocol.v1 import ForgetRequest, ForgetResponse

        req = (
            request if isinstance(request, ForgetRequest) else ForgetRequest(**request.model_dump())
        )
        try:
            self.forget(fact_id=req.fact_id)
            return ForgetResponse(id=req.id)
        except Exception as e:
            return ForgetResponse(id=req.id, success=False, error=str(e))

    def _handle_recall(self, request: Request) -> Response:
        """Handle recall request."""
        from llm_agent.protocol.v1 import RecallRequest, RecallResponse

        req = (
            request if isinstance(request, RecallRequest) else RecallRequest(**request.model_dump())
        )
        try:
            scored_facts = self.recall(
                query=req.query,
                top_k=req.top_k,
                min_similarity=req.min_similarity,
                categories=req.categories,
            )
            facts = [
                {
                    "fact_id": sf.fact.id,
                    "content": sf.fact.content,
                    "category": sf.fact.category,
                    "similarity": sf.similarity,
                }
                for sf in scored_facts
            ]
            return RecallResponse(id=req.id, facts=facts)
        except Exception as e:
            return RecallResponse(id=req.id, success=False, error=str(e))

    def _handle_feedback(self, request: Request) -> Response:
        """Handle feedback request."""
        from llm_agent.protocol.v1 import FeedbackRequest, FeedbackResponse

        req = (
            request
            if isinstance(request, FeedbackRequest)
            else FeedbackRequest(**request.model_dump())
        )
        try:
            self.feedback(
                response_id=req.response_id,
                signal=req.signal,
                correction=req.correction,
            )
            return FeedbackResponse(id=req.id)
        except Exception as e:
            return FeedbackResponse(id=req.id, success=False, error=str(e))

    # === Adapter ===

    def load_adapter(self, adapter_path: str) -> None:
        """Load a fine-tuned adapter.

        Args:
            adapter_path: Path to the adapter.
        """
        self._llm.load_adapter(adapter_path)

    def unload_adapter(self) -> None:
        """Revert to base model."""
        self._llm.unload_adapter()


class _ResponseContext:
    """Internal: tracks response context for feedback."""

    __slots__ = ("system_prompt", "query", "response", "model")

    def __init__(
        self,
        system_prompt: str,
        query: str,
        response: str,
        model: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.query = query
        self.response = response
        self.model = model
