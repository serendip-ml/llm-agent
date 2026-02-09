"""Default agent implementation.

A concrete Agent that uses SAIA for execution. Provides run_once(), ask(),
and other methods expected by the runtime runner.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from appinfra.time import time

from ...core.agent import Agent as BaseAgent
from ...core.dispatcher import Dispatcher
from ...core.memory import (
    format_solutions_context,
    recall_chronological,
    recall_semantic,
)
from ...core.runnable import ExecutionResult


if TYPE_CHECKING:
    from llm_agent.core.agent import Identity


# Lazy import helpers to avoid circular dependencies
# These functions delay imports until runtime, preventing import cycles
def _get_learn_trait_class() -> type:
    from ...core.traits.builtin.learn import LearnTrait

    return LearnTrait


def _get_saia_trait_class() -> type:
    from ...core.traits.builtin.saia import SAIATrait

    return SAIATrait


def _get_llm_trait_class() -> type:
    from ...core.traits.builtin.llm import LLMTrait

    return LLMTrait


def _get_tools_trait_class() -> type:
    from ...core.traits.builtin.tools import ToolsTrait

    return ToolsTrait


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
        from llm_agent.core.agent import Identity

        identity = Identity.from_name("explorer")
        agent = Agent(lg, identity=identity, default_prompt="Analyze the codebase")
        agent.add_trait(SAIATrait(agent, backend=backend))
        agent.add_trait(ToolsTrait(agent))
        agent.start()

        result = agent.run_once()
        response = agent.ask("What files are in src/?")
    """

    def __init__(self, lg: Logger, identity: Identity, default_prompt: str = "") -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            identity: Agent identity (domain/workspace/name).
            default_prompt: Default prompt for run_once() execution.
        """
        super().__init__(lg)
        self.identity = identity
        self._default_prompt = default_prompt
        self._cycle_count = 0
        self._recent_results: list[ExecutionResult] = []
        self._max_recent = 100
        # Track response IDs for feedback validation (bounded FIFO to prevent memory leaks)
        self._response_ids: OrderedDict[str, None] = OrderedDict()
        self._max_response_ids = 1000
        # Event dispatcher for declarative behavior
        self._dispatcher = Dispatcher()

    @property
    def name(self) -> str:
        """Agent name (from identity)."""
        return str(self.identity.name)

    @property
    def cycle_count(self) -> int:
        """Number of execution cycles completed."""
        return self._cycle_count

    def start(self) -> None:
        """Start the agent and all attached traits."""
        if self._started:
            return
        self._start_traits()
        self._register_event_handlers()
        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop the agent and all attached traits."""
        if not self._started:
            return
        self._stop_traits()
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self.name})

    def _register_event_handlers(self) -> None:
        """Register default event handlers for schedule and question events.

        Uses chronological recall for scheduled tasks (repetitive execution).
        Uses semantic recall for ad-hoc questions (varied queries).

        Only registers handlers if not already registered, allowing factory
        or subclasses to override with custom handlers.
        """
        # Register schedule event handler (chronological recall) if not already set
        if not self._dispatcher.has_handler("schedule"):
            self._dispatcher.on("schedule", self._on_schedule)

        # Register question event handler (semantic recall) if not already set
        if not self._dispatcher.has_handler("question"):
            self._dispatcher.on("question", self._on_question)

    async def _on_schedule(
        self,
        task: str,
        agent_name: str,
        saia_trait: Any,
        learn_trait: Any | None,
    ) -> dict[str, Any]:
        """Handle scheduled execution with chronological recall.

        Args:
            task: The task to execute.
            agent_name: Name of this agent.
            saia_trait: SAIATrait instance.
            learn_trait: Optional LearnTrait instance.

        Returns:
            Dict with execution result fields.
        """
        # Recall recent solutions chronologically
        context = ""
        if learn_trait is not None:
            past = recall_chronological(learn_trait, agent_name, limit=5)
            context = format_solutions_context(past)
            if context:
                self._lg.debug(
                    "recalled past solutions (chronological)",
                    extra={"agent": agent_name, "count": len(past)},
                )

        # Compose prompt using SAIA
        prompt = saia_trait.saia.compose(context, task)

        # Execute
        saia_result = await saia_trait.saia.complete(prompt)
        return {
            "success": saia_result.completed,
            "content": saia_result.output,
            "iterations": saia_result.iterations,
            "tokens_used": saia_result.score.total_tokens if saia_result.score else 0,
            "trace_id": saia_result.trace_id,
        }

    async def _on_question(
        self,
        question: str,
        agent_name: str,
        saia_trait: Any,
        learn_trait: Any | None,
    ) -> dict[str, Any]:
        """Handle ad-hoc questions with semantic recall.

        Args:
            question: The question to answer.
            agent_name: Name of this agent.
            saia_trait: SAIATrait instance.
            learn_trait: Optional LearnTrait instance.

        Returns:
            Dict with execution result fields.
        """
        # Recall semantically similar solutions
        context = ""
        if learn_trait is not None:
            past = recall_semantic(learn_trait, query=question, limit=5, agent_name=agent_name)
            context = format_solutions_context(past)
            if context:
                self._lg.debug(
                    "recalled past solutions (semantic)",
                    extra={"agent": agent_name, "count": len(past)},
                )

        # Compose prompt using SAIA
        prompt = saia_trait.saia.compose(context, question)

        # Execute
        saia_result = await saia_trait.saia.complete(prompt)
        return {
            "success": saia_result.completed,
            "content": saia_result.output,
            "iterations": saia_result.iterations,
            "tokens_used": saia_result.score.total_tokens if saia_result.score else 0,
            "trace_id": saia_result.trace_id,
        }

    def run_once(self) -> ExecutionResult:
        """Execute one cycle using the default prompt.

        Returns:
            ExecutionResult with success status and content.
        """
        if not self._default_prompt:
            return ExecutionResult(success=False, content="No default prompt configured")

        start_t = time.start()
        result = asyncio.run(self._run_once_pipeline())
        result.latency_ms = int(time.since(start_t))
        self._cycle_count += 1
        self._store_result(result)

        self._lg.info(
            "execution completed",
            extra={
                "after": result.latency_ms,
                "agent": self.name,
                "success": result.success,
                "iters": result.iterations,
                "tokens": result.tokens_used,
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
        if self._dispatcher.has_handler("schedule"):
            result = await self._run_with_dispatcher()
        else:
            result = await self._run_legacy()

        if result.success:
            await self._persist_conclusion(result)
        return result

    async def _run_with_dispatcher(self) -> ExecutionResult:
        """Run execution using event orchestrator."""
        LearnTrait = _get_learn_trait_class()
        SAIATrait = _get_saia_trait_class()

        saia_trait = self.get_trait(SAIATrait)
        learn_trait = self.get_trait(LearnTrait)

        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        result_dict = await self._dispatcher.trigger(
            "schedule",
            task=self._default_prompt,
            agent_name=self.name,
            saia_trait=saia_trait,
            learn_trait=learn_trait,
        )

        return self._convert_to_execution_result(result_dict)

    async def _run_legacy(self) -> ExecutionResult:
        """Run execution using legacy approach (no orchestrator)."""
        past_solutions = self._recall_past_solutions()
        context = self._format_past_solutions_context(past_solutions)

        prompt = self._default_prompt
        if context:
            prompt = f"{context}\n\n{self._default_prompt}"

        return await self._execute_async(prompt)

    def _convert_to_execution_result(self, result_dict: Any) -> ExecutionResult:
        """Convert orchestrator result dict to ExecutionResult."""
        if isinstance(result_dict, dict):
            return ExecutionResult(
                success=result_dict.get("success", False),
                content=result_dict.get("content", ""),
                iterations=result_dict.get("iterations", 0),
                tokens_used=result_dict.get("tokens_used", 0),
                trace_id=result_dict.get("trace_id", ""),
            )
        else:
            # If handler returned ExecutionResult directly
            return result_dict  # type: ignore[no-any-return]

    def ask(self, question: str) -> str:
        """Ask the agent a question.

        Args:
            question: The question to ask.

        Returns:
            Response string from the agent.
        """
        # Use event-based orchestration if handler is registered
        result: ExecutionResult
        if self._dispatcher.has_handler("question"):
            result = asyncio.run(self._ask_async(question))
        else:
            result = self._execute(question)

        self._store_result(result)
        return result.content

    async def _ask_async(self, question: str) -> ExecutionResult:
        """Async implementation of ask() using event orchestration."""
        LearnTrait = _get_learn_trait_class()
        SAIATrait = _get_saia_trait_class()

        saia_trait = self.get_trait(SAIATrait)
        learn_trait = self.get_trait(LearnTrait)

        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        result_dict = await self._dispatcher.trigger(
            "question",
            question=question,
            agent_name=self.name,
            saia_trait=saia_trait,
            learn_trait=learn_trait,
        )

        return self._convert_to_execution_result(result_dict)

    def record_feedback(self, message: str) -> None:
        """Record feedback (no-op for non-learning agent)."""
        self._lg.debug("feedback received (not stored)", extra={"agent": self.name})

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
        SAIATrait = _get_saia_trait_class()

        saia_trait = self.get_trait(SAIATrait)
        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        try:
            saia_result = await saia_trait.saia.complete(prompt)
            return ExecutionResult(
                success=saia_result.completed,
                content=saia_result.output,
                iterations=saia_result.iterations,
                tokens_used=saia_result.score.total_tokens if saia_result.score else 0,
                trace_id=saia_result.trace_id,
            )
        except Exception as e:
            self._lg.warning("execution failed", extra={"agent": self.name, "exception": e})
            return ExecutionResult(success=False, content=f"Execution error: {e}")

    def _store_result(self, result: ExecutionResult) -> None:
        """Store result in recent history."""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent :]

    def _format_solution_summaries(self, solutions: list[Any]) -> list[dict[str, Any]]:
        """Format solution facts into summary dicts for logging."""
        return [
            {
                "problem": f.solution_details.problem[:100] if f.solution_details else "",
                "success": (
                    f.solution_details.answer.get("success", False) if f.solution_details else False
                ),
                "tokens": f.solution_details.tokens_used if f.solution_details else 0,
            }
            for f in solutions
        ]

    def _log_recalled_solutions(self, solutions: list[Any]) -> None:
        """Log recalled solutions for analysis."""
        self._lg.info(
            "recalled past solutions",
            extra={
                "agent": self.name,
                "count": len(solutions),
                "solutions": self._format_solution_summaries(solutions),
            },
        )

    def _format_past_solutions_context(self, solutions: list[Any]) -> str:
        """Format past solutions into context string for prompt injection."""
        if not solutions:
            return ""

        lines = ["## Previously Completed Tasks\n"]
        for i, sol in enumerate(solutions, 1):
            if sol.solution_details:
                output = sol.solution_details.answer.get("output", "")
                if output:
                    # Extract just the actual output, not metadata
                    lines.append(f"{i}. {output[:200]}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _recall_past_solutions(self, limit: int = 5) -> list[Any]:
        """Recall past solutions for similar problems.

        Args:
            limit: Maximum number of past solutions to recall.

        Returns:
            List of past solution facts.
        """
        LearnTrait = _get_learn_trait_class()

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            return []

        try:
            past_solutions = learn_trait.learn.solutions.search(
                query=self._default_prompt,
                limit=limit,
                active_only=True,
            )

            if not past_solutions:
                self._lg.debug(
                    "no past solutions found",
                    extra={"agent": self.name, "query": self._default_prompt[:100]},
                )
            else:
                self._log_recalled_solutions(past_solutions)

            return past_solutions

        except Exception as e:
            self._lg.debug(
                "failed to recall past solutions",
                extra={"agent": self.name, "exception": e},
            )
            return []

    def _build_problem_context(self, result: ExecutionResult) -> dict[str, Any]:
        """Build problem context dict including iterations, trace ID, and tools."""
        ToolsTrait = _get_tools_trait_class()

        context: dict[str, Any] = {
            "iterations": result.iterations,
            "trace_id": result.trace_id,
        }

        # Include tool names if ToolsTrait is attached
        tools_trait = self.get_trait(ToolsTrait)
        if tools_trait is not None:
            tool_names = [t.name for t in tools_trait.registry.list_tools()]
            context["tools_available"] = tool_names

        return context

    def _build_answer_payload(self, result: ExecutionResult) -> dict[str, Any]:
        """Build answer payload dict from execution result."""
        return {
            "success": result.success,
            "output": result.content,
            "iterations": result.iterations,
        }

    def _record_solution(self, learn_trait: Any, result: ExecutionResult, summary: str) -> int:
        """Record solution to database and log success."""
        fact_id = learn_trait.learn.solutions.record(
            agent_name=self.name,
            problem=self._default_prompt,
            problem_context=self._build_problem_context(result),
            answer=self._build_answer_payload(result),
            answer_text=summary,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            category="execution",
            source="agent",
        )

        self._lg.debug(
            "solution persisted",
            extra={
                "agent": self.name,
                "fact_id": fact_id,
                "tokens": result.tokens_used,
                "latency_ms": result.latency_ms,
            },
        )
        return fact_id  # type: ignore[no-any-return]

    async def _persist_conclusion(self, result: ExecutionResult) -> None:
        """Persist a complete solution record from a successful run."""
        LearnTrait = _get_learn_trait_class()
        SAIATrait = _get_saia_trait_class()

        learn_trait = self.get_trait(LearnTrait)
        saia_trait = self.get_trait(SAIATrait)
        if learn_trait is None or saia_trait is None:
            return

        content = result.content.strip()
        if not content:
            return

        try:
            summary = await self._summarize_output(saia_trait.saia, content)
            if summary:
                self._record_solution(learn_trait, result, summary)
        except Exception as e:
            self._lg.warning(
                "failed to persist solution",
                extra={"agent": self.name, "exception": e},
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

        LLMTrait = _get_llm_trait_class()

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
        LearnTrait = _get_learn_trait_class()

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
        LearnTrait = _get_learn_trait_class()

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
        LearnTrait = _get_learn_trait_class()

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

        LearnTrait = _get_learn_trait_class()

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
