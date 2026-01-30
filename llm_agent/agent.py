"""Core agent implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

from appinfra.log import Logger
from llm_learn.collection import ScoredFact

from llm_agent.config import AgentConfig
from llm_agent.llm import CompletionResult, Message
from llm_agent.traits.base import Trait


if TYPE_CHECKING:
    from llm_agent.protocol.base import Request, Response
    from llm_agent.traits.learn import LearnTrait


TraitT = TypeVar("TraitT", bound=Trait)


class Agent:
    """Learning agent that improves through feedback.

    Coordinates capabilities via traits:
    - LLMTrait: LLM completions
    - LearnTrait: Memory (facts), feedback, preferences
    - DirectiveTrait: Directive/persona injection
    - HTTPTrait: HTTP server

    Example:
        from appinfra.log import Logger
        from llm_agent import Agent, AgentConfig
        from llm_agent.traits import LLMTrait, LLMConfig, LearnTrait, LearnConfig

        lg = Logger.create("agent")
        config = AgentConfig(name="my-agent")

        agent = Agent(lg, config)
        agent.add_trait(LLMTrait(LLMConfig(base_url="http://localhost:8000/v1")))
        agent.add_trait(LearnTrait(LearnConfig(profile_id=1)))
        agent.start()

        result = agent.complete("Hello!")
    """

    def __init__(
        self,
        lg: Logger,
        config: AgentConfig,
    ) -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            config: Agent configuration.
        """
        self._lg = lg
        self._config = config
        self._response_contexts: dict[str, _ResponseContext] = {}
        self._traits: dict[type[Trait], Trait] = {}
        self._started = False

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the agent and all traits."""
        if self._started:
            return

        for trait in self._traits.values():
            trait.on_start()

        self._started = True
        self._lg.info("Agent started", extra={"name": self.name})

    def stop(self) -> None:
        """Stop the agent and all traits."""
        if not self._started:
            return

        for trait in self._traits.values():
            trait.on_stop()

        self._started = False
        self._lg.info("Agent stopped", extra={"name": self.name})

    # =========================================================================
    # Traits
    # =========================================================================

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

    def require_trait(self, trait_type: type[TraitT]) -> TraitT:
        """Get a required trait, raising if not attached.

        Args:
            trait_type: The trait class to look up.

        Returns:
            The trait instance.

        Raises:
            RuntimeError: If the trait is not attached.
        """
        trait = self.get_trait(trait_type)
        if trait is None:
            raise RuntimeError(
                f"{trait_type.__name__} required but not attached - "
                f"add it with agent.add_trait({trait_type.__name__}(...))"
            )
        return trait

    # =========================================================================
    # Core operations
    # =========================================================================

    def complete(
        self,
        query: str,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a response with context-augmented prompt.

        Requires LLMTrait. Facts are automatically injected based on
        config.fact_injection mode (requires LearnTrait for 'all' or 'rag').

        Args:
            query: User input.
            system_prompt: System prompt (uses default if None).

        Returns:
            Completion result with response and metadata.

        Raises:
            RuntimeError: If LLMTrait not attached.
        """
        from llm_agent.traits.llm import LLMTrait

        llm_trait = self.require_trait(LLMTrait)

        base_prompt = system_prompt or self._config.default_prompt
        prompt = self._build_prompt(base_prompt, query=query)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=query),
        ]

        result = llm_trait.complete(
            messages=messages,
            model=self._config.model,
        )

        # Track response for feedback (bounded to prevent memory leaks)
        if len(self._response_contexts) >= self._config.max_tracked_responses:
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
        from llm_agent.traits.learn import LearnTrait

        prompt = base_prompt

        # Inject directive if DirectiveTrait is attached
        directive_trait = self.get_trait(DirectiveTrait)
        if directive_trait is not None:
            prompt = directive_trait.build_prompt(prompt)

        # Inject facts based on config (requires LearnTrait)
        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is not None and self._config.fact_injection != "none":
            prompt = self._inject_facts(prompt, learn_trait, query)

        return prompt

    def _inject_facts(self, prompt: str, learn_trait: LearnTrait, query: str | None) -> str:
        """Inject facts into prompt based on config mode."""
        if self._config.fact_injection == "rag":
            if query is None:
                raise ValueError("RAG mode requires query for semantic search")
            if not learn_trait.has_embedder:
                raise ValueError(
                    "fact_injection='rag' requires embedder - configure embedder_url in LearnConfig"
                )
            return learn_trait.build_prompt_rag(
                base_prompt=prompt,
                query=query,
                top_k=self._config.rag_top_k,
                min_similarity=self._config.rag_min_similarity,
            )
        elif self._config.fact_injection == "all":
            return learn_trait.build_prompt(
                base_prompt=prompt,
                max_facts=self._config.max_facts,
            )
        return prompt

    # =========================================================================
    # Memory (delegates to LearnTrait)
    # =========================================================================

    def remember(self, fact: str, category: str = "general") -> int:
        """Store a fact about the user.

        Requires LearnTrait.

        Args:
            fact: The fact to store.
            category: Category for organization.

        Returns:
            Fact ID.
        """
        from llm_agent.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)
        return learn_trait.remember(fact, category=category)

    def forget(self, fact_id: int) -> None:
        """Remove a stored fact.

        Requires LearnTrait.

        Args:
            fact_id: ID of the fact to remove.
        """
        from llm_agent.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)
        learn_trait.forget(fact_id)

    def recall(
        self,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        categories: list[str] | None = None,
    ) -> list[ScoredFact]:
        """Search facts by semantic similarity to query.

        Requires LearnTrait with embedder configured.

        Args:
            query: Text to search for similar facts.
            top_k: Max results (defaults to config.rag_top_k).
            min_similarity: Minimum similarity (defaults to config.rag_min_similarity).
            categories: Filter to these categories.

        Returns:
            List of ScoredFact sorted by similarity (highest first).
        """
        from llm_agent.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)
        return learn_trait.recall(
            query=query,
            top_k=top_k if top_k is not None else self._config.rag_top_k,
            min_similarity=min_similarity
            if min_similarity is not None
            else self._config.rag_min_similarity,
            categories=categories,
        )

    # =========================================================================
    # Feedback (delegates to LearnTrait)
    # =========================================================================

    def feedback(
        self,
        response_id: str,
        signal: Literal["positive", "negative"],
        correction: str | None = None,
    ) -> None:
        """Record feedback on a response.

        Requires LearnTrait.

        Args:
            response_id: ID from CompletionResult.
            signal: Whether response was good or bad.
            correction: If negative, the preferred response (creates preference pair).

        Raises:
            ValueError: If response_id is not found.
        """
        from llm_agent.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)

        # Pop to remove after use - prevents unbounded memory growth
        ctx = self._response_contexts.pop(response_id, None)
        if ctx is None:
            raise ValueError(f"Unknown response_id: {response_id}")

        # Record feedback signal
        learn_trait.record_feedback(
            content=ctx.response,
            signal=signal,
            context={"query": ctx.query, "model": ctx.model},
        )

        # If negative with correction, create preference pair
        if signal == "negative" and correction is not None:
            full_context = f"{ctx.system_prompt}\n\nUser: {ctx.query}"
            learn_trait.record_preference(
                context=full_context,
                chosen=correction,
                rejected=ctx.response,
            )

    # =========================================================================
    # HTTP Request Handling
    # =========================================================================

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
            self._lg.warning("complete request failed", extra={"exception": e})
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
            self._lg.warning("remember request failed", extra={"exception": e})
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
            self._lg.warning("forget request failed", extra={"exception": e})
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
            self._lg.warning("recall request failed", extra={"exception": e})
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
            self._lg.warning("feedback request failed", extra={"exception": e})
            return FeedbackResponse(id=req.id, success=False, error=str(e))


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
