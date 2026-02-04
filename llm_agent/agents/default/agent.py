"""Default agent implementation.

A concrete Agent that uses SAIA for execution. Provides run_once(), ask(),
and other methods expected by the runtime runner.
"""

from __future__ import annotations

import asyncio
from typing import Any

from appinfra.log import Logger

from llm_agent.core.agent import Agent as BaseAgent
from llm_agent.core.runnable import ExecutionResult


class Agent(BaseAgent):
    """Default agent - uses SAIA for execution.

    Provides the interface expected by the runtime runner:
    - run_once(): Execute default task
    - ask(): Interactive Q&A
    - record_feedback(): Feedback handling (no-op for non-learning)
    - cycle_count: Execution counter
    - get_recent_results(): Recent execution history

    Example:
        agent = Agent(lg, name="explorer", default_prompt="Analyze the codebase")
        agent.add_trait(SAIATrait(lg=lg, backend=backend))
        agent.add_trait(ToolsTrait())
        agent.start()

        result = agent.run_once()
        response = agent.ask("What files are in src/?")
    """

    def __init__(self, lg: Logger, name: str, default_prompt: str = "") -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            name: Agent identifier.
            default_prompt: Default prompt for run_once() execution.
        """
        super().__init__(lg)
        self._name = name
        self._default_prompt = default_prompt
        self._cycle_count = 0
        self._recent_results: list[ExecutionResult] = []
        self._max_recent = 100
        self._response_ids: set[str] = set()  # Track response IDs for feedback validation

    @property
    def name(self) -> str:
        """Agent identifier."""
        return self._name

    @property
    def cycle_count(self) -> int:
        """Number of execution cycles completed."""
        return self._cycle_count

    def start(self) -> None:
        """Start the agent and all attached traits."""
        if self._started:
            return
        self._start_traits()
        self._started = True
        self._lg.info("agent started", extra={"agent": self._name})

    def stop(self) -> None:
        """Stop the agent and all attached traits."""
        if not self._started:
            return
        self._stop_traits()
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self._name})

    def run_once(self) -> ExecutionResult:
        """Execute one cycle using the default prompt.

        Returns:
            ExecutionResult with success status and content.
        """
        if not self._default_prompt:
            return ExecutionResult(success=False, content="No default prompt configured")

        result = self._execute(self._default_prompt)
        self._cycle_count += 1
        self._store_result(result)
        return result

    def ask(self, question: str) -> str:
        """Ask the agent a question.

        Args:
            question: The question to ask.

        Returns:
            Response string from the agent.
        """
        result = self._execute(question)
        self._store_result(result)
        return result.content

    def record_feedback(self, message: str) -> None:
        """Record feedback (no-op for non-learning agent)."""
        self._lg.debug("feedback received (not stored)", extra={"agent": self._name})

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent ExecutionResult objects.
        """
        return self._recent_results[-limit:]

    def _execute(self, prompt: str) -> ExecutionResult:
        """Execute a prompt using SAIA."""
        from llm_agent.core.traits.saia import SAIATrait

        saia_trait = self.get_trait(SAIATrait)
        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        try:
            saia_result = asyncio.get_event_loop().run_until_complete(
                saia_trait.saia.complete(prompt)
            )
            return ExecutionResult(
                success=saia_result.completed,
                content=saia_result.output,
                iterations=saia_result.iterations,
            )
        except Exception as e:
            self._lg.warning("execution failed", extra={"agent": self._name, "exception": e})
            return ExecutionResult(success=False, content=f"Execution error: {e}")

    def _store_result(self, result: ExecutionResult) -> None:
        """Store result in recent history."""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent :]

    # HTTP Protocol methods (used by HTTPTrait)

    def complete(self, query: str, system_prompt: str | None = None) -> Any:
        """Complete a query using LLMTrait.

        Args:
            query: The query to complete.
            system_prompt: Optional system prompt override.

        Returns:
            CompletionResult with response data.

        Raises:
            RuntimeError: If LLMTrait is not attached.
        """
        from llm_agent.core.traits.llm import LLMTrait

        llm_trait = self.get_trait(LLMTrait)
        if llm_trait is None:
            raise RuntimeError("LLMTrait not attached")
        # HTTP protocol interface - tests mock this to accept query/system_prompt
        result = llm_trait.complete(query=query, system_prompt=system_prompt)  # type: ignore[call-arg]
        # Track response_id for feedback validation
        if hasattr(result, "id") and result.id:
            self._response_ids.add(result.id)
        return result

    def remember(self, fact: str, category: str | None = None) -> int:
        """Store a fact using LearnTrait.

        Args:
            fact: The fact to remember.
            category: Optional category for the fact.

        Returns:
            The fact ID.

        Raises:
            RuntimeError: If LearnTrait is not attached.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        # HTTP protocol allows None category, trait requires str
        return learn_trait.remember(fact=fact, category=category or "general")

    def forget(self, fact_id: int) -> None:
        """Remove a fact using LearnTrait.

        Args:
            fact_id: The ID of the fact to forget.

        Raises:
            RuntimeError: If LearnTrait is not attached.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        learn_trait.forget(fact_id=fact_id)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        categories: list[str] | None = None,
    ) -> list[Any]:
        """Recall facts similar to query using LearnTrait.

        Args:
            query: The query to search for.
            top_k: Maximum number of results.
            min_similarity: Minimum similarity threshold.
            categories: Optional category filter.

        Returns:
            List of scored facts.

        Raises:
            RuntimeError: If LearnTrait is not attached.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        return learn_trait.recall(
            query=query, top_k=top_k, min_similarity=min_similarity, categories=categories
        )

    def feedback(self, response_id: str, signal: str, correction: str | None = None) -> None:
        """Record feedback using LearnTrait.

        Args:
            response_id: ID of the response being rated.
            signal: Feedback signal (e.g., "positive", "negative").
            correction: Optional correction text.

        Raises:
            ValueError: If response_id is not recognized.
            RuntimeError: If LearnTrait is not attached.
        """
        if response_id not in self._response_ids:
            raise ValueError(f"Unknown response_id: {response_id}")

        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        # HTTP protocol interface - tests mock this with HTTP protocol args
        learn_trait.record_feedback(  # type: ignore[call-arg]
            response_id=response_id,
            signal=signal,  # type: ignore[arg-type]
            correction=correction,
        )
