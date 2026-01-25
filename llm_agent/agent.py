"""Core agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from llm_learn import LearnClient
from llm_learn.inference import ContextBuilder

from llm_agent.config import AgentConfig
from llm_agent.llm import CompletionResult, LLMBackend, Message


if TYPE_CHECKING:
    from llm_learn.inference import Embedder


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
        self._config = config
        self._llm = llm
        self._learn = learn
        self._embedder = embedder
        self._context = ContextBuilder(learn.facts)
        self._response_contexts: dict[str, _ResponseContext] = {}

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

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

        # Build prompt with fact injection
        prompt = self._build_prompt(base_prompt)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=query),
        ]

        result = self._llm.complete(
            messages=messages,
            model=self._config.model,
        )

        # Track response for feedback
        self._response_contexts[result.id] = _ResponseContext(
            system_prompt=prompt,
            query=query,
            response=result.content,
            model=result.model,
        )

        return result

    def _build_prompt(self, base_prompt: str) -> str:
        """Build system prompt with fact injection based on config."""
        mode = self._config.fact_injection

        if mode == "none":
            return base_prompt

        if mode == "all":
            return cast(
                str,
                self._context.build_system_prompt(
                    base_prompt=base_prompt,
                    max_facts=self._config.max_facts,
                ),
            )

        # RAG mode - requires embedder (Phase 3)
        # For now, fall back to "all" mode
        return cast(
            str,
            self._context.build_system_prompt(
                base_prompt=base_prompt,
                max_facts=self._config.max_facts,
            ),
        )

    # === Memory (delegates to llm-learn) ===

    def remember(self, fact: str, category: str = "general") -> int:
        """Store a fact about the user.

        Args:
            fact: The fact to store.
            category: Category for organization (preferences, rules, background, etc.)

        Returns:
            Fact ID.
        """
        return cast(int, self._learn.facts.add(fact, category=category))

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
        ctx = self._response_contexts.get(response_id)
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
