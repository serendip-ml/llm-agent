"""Default agent implementation.

A concrete Agent that uses SAIA for execution. Provides run_once(), ask(),
and other methods expected by the runtime runner.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger
from appinfra.time import time

from llm_agent.core.agent import Agent as BaseAgent
from llm_agent.core.runnable import ExecutionResult


@dataclass
class _ConclusionSummary:
    """Concise summary of findings from agent execution."""

    summary: str


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
        # Track response IDs for feedback validation (bounded FIFO to prevent memory leaks)
        self._response_ids: OrderedDict[str, None] = OrderedDict()
        self._max_response_ids = 1000

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

        start_t = time.start()
        result = asyncio.run(self._run_once_pipeline())
        self._cycle_count += 1
        self._store_result(result)

        self._lg.info(
            "execution completed",
            extra={
                "after": time.since(start_t),
                "agent": self._name,
                "success": result.success,
                "iters": result.iterations,
                "output": result.content,
            },
        )

        return result

    async def _run_once_pipeline(self) -> ExecutionResult:
        """Run execution and optional conclusion persistence in one event loop.

        Both SAIA complete() and extract() share the same async backend,
        so they must run within a single asyncio.run() to avoid stale
        event-loop references on the httpx client.
        """
        result = await self._execute_async(self._default_prompt)
        if result.success:
            await self._persist_conclusion(result)
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
        """Execute a prompt using SAIA (sync wrapper for ask() etc.)."""
        return asyncio.run(self._execute_async(prompt))

    async def _execute_async(self, prompt: str) -> ExecutionResult:
        """Execute a prompt using SAIA."""
        from llm_agent.core.traits.saia import SAIATrait

        saia_trait = self.get_trait(SAIATrait)
        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        try:
            saia_result = await saia_trait.saia.complete(prompt)
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

    async def _persist_conclusion(self, result: ExecutionResult) -> None:
        """Persist a summarized conclusion from a successful run.

        Uses SAIA extract verb to distill the raw execution output into a
        concise summary before storing it as a fact. This avoids storing
        noisy tool-call traces and intermediate reasoning.
        """
        from llm_agent.core.traits.learn import LearnTrait
        from llm_agent.core.traits.saia import SAIATrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            return

        content = result.content.strip()
        if not content:
            return

        saia_trait = self.get_trait(SAIATrait)
        if saia_trait is None:
            return

        try:
            summary = await self._summarize_output(saia_trait.saia, content)
            if not summary:
                return

            fact_id = learn_trait.remember(fact=summary, category="conclusion", source="inferred")
            self._lg.debug(
                "conclusion persisted",
                extra={"agent": self._name, "fact_id": fact_id},
            )
        except Exception as e:
            self._lg.warning(
                "failed to persist conclusion",
                extra={"agent": self._name, "exception": e},
            )

    async def _summarize_output(self, saia: Any, content: str) -> str | None:
        """Summarize raw SAIA output into a concise conclusion via extract verb."""
        result = await saia.extract(
            content,
            _ConclusionSummary,
            instructions=(
                "Summarize the key findings and conclusions from this agent "
                "execution. Be concise — focus on what was discovered, not "
                "the steps taken. Output only factual findings."
            ),
        )
        return result.summary.strip() or None

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
        from llm_agent.core.llm.types import Message
        from llm_agent.core.traits.llm import LLMTrait

        llm_trait = self.get_trait(LLMTrait)
        if llm_trait is None:
            raise RuntimeError("LLMTrait not attached")

        # Build messages from HTTP API format
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=query))

        result = llm_trait.complete(messages=messages)
        # Track response_id for feedback validation (bounded FIFO)
        if hasattr(result, "id") and result.id:
            self._response_ids[result.id] = None
            # Evict oldest entries if over limit
            while len(self._response_ids) > self._max_response_ids:
                self._response_ids.popitem(last=False)
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
            signal: Feedback signal ("positive" or "negative").
            correction: Optional correction text.

        Raises:
            ValueError: If response_id is not recognized or signal is invalid.
            RuntimeError: If LearnTrait is not attached.
        """
        if response_id not in self._response_ids:
            raise ValueError(f"Unknown response_id: {response_id}")

        if signal not in ("positive", "negative"):
            raise ValueError(f"Invalid signal: {signal}. Must be 'positive' or 'negative'")

        from typing import Literal, cast

        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")

        # Convert HTTP API format to LearnTrait.record_feedback format
        context: dict[str, Any] = {"response_id": response_id}
        if correction:
            context["correction"] = correction

        learn_trait.record_feedback(
            content=correction or "",
            signal=cast(Literal["positive", "negative"], signal),
            context=context,
        )
