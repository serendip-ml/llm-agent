"""Default agent implementation.

A concrete Agent that uses SAIA for execution. Provides run_once(), ask(),
and other methods expected by the runtime runner.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from appinfra.time import time

from ...core.agent import Agent as BaseAgent
from ...core.dispatcher import Dispatcher
from ...core.runnable import ExecutionResult
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.saia import SAIATrait


if TYPE_CHECKING:
    from llm_agent.core.agent import Identity


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
        """Register event handlers.

        For default agents, handlers are registered by the factory.
        This method exists as a hook for subclasses to customize behavior.
        """
        pass

    async def handle_task(
        self,
        task: str,
        recall_strategy: str = "chronological",
        recall_limit: int = 5,
    ) -> dict[str, Any]:
        """Handle task execution with configurable memory recall.

        This is the core event handler used by schedule/question events.

        Args:
            task: The task or question to execute.
            recall_strategy: "chronological" or "semantic" recall.
            recall_limit: Maximum number of past solutions to recall.

        Returns:
            Dict with execution result fields.
        """
        saia_trait = self.get_trait(SAIATrait)
        learn_trait = self.get_trait(LearnTrait)

        if saia_trait is None:
            return {
                "success": False,
                "content": "SAIATrait not attached",
                "iterations": 0,
                "tokens_used": 0,
            }

        # Recall past solutions for context
        context = self._recall_context(learn_trait, task, recall_strategy, recall_limit)

        # Execute task
        prompt = saia_trait.saia.compose(context, task)
        saia_result = await saia_trait.saia.complete(prompt)

        result = self._build_result_dict(saia_result)

        # Persist successful outcomes
        if result["success"] and learn_trait is not None:
            await self._persist_outcome(learn_trait, saia_trait, task, result)

        return result

    def _build_result_dict(self, saia_result: Any) -> dict[str, Any]:
        """Build result dict from SAIA execution result."""
        return {
            "success": saia_result.completed,
            "content": saia_result.output,
            "iterations": saia_result.iterations,
            "tokens_used": saia_result.score.total_tokens if saia_result.score else 0,
            "trace_id": saia_result.trace_id,
        }

    def _recall_context(
        self,
        learn_trait: LearnTrait | None,
        task: str,
        recall_strategy: str,
        recall_limit: int,
    ) -> str:
        """Recall past solutions and format as context string."""
        from ...core.memory import (
            format_solutions_context,
            recall_chronological,
            recall_semantic,
        )

        if learn_trait is None:
            return ""

        if recall_strategy == "semantic":
            past = recall_semantic(
                learn_trait, query=task, limit=recall_limit, agent_name=self.name
            )
        else:  # chronological
            past = recall_chronological(learn_trait, self.name, limit=recall_limit)

        return format_solutions_context(past)

    async def _persist_outcome(
        self,
        learn_trait: LearnTrait,
        saia_trait: SAIATrait,
        task: str,
        result: dict[str, Any],
    ) -> None:
        """Persist execution outcome to learning database."""
        content = result.get("content", "").strip()
        if not content:
            return

        try:
            summary = await self._summarize_outcome(saia_trait, content)
            if summary:
                self._record_solution(learn_trait, task, result, content, summary)
        except Exception as e:
            self._lg.warning(
                "failed to persist solution", extra={"agent": self.name, "exception": e}
            )

    async def _summarize_outcome(self, saia_trait: SAIATrait, content: str) -> str:
        """Summarize execution outcome using SAIA extract."""
        from dataclasses import dataclass

        @dataclass
        class ConclusionSummary:
            """Summary of agent execution findings."""

            summary: str

        summary_result = await saia_trait.saia.extract(
            content,
            ConclusionSummary,
            instructions=(
                "Summarize the key findings and conclusions from this agent "
                "execution. Be concise — focus on what was discovered, not "
                "the steps taken. Output only factual findings."
            ),
        )
        return summary_result.summary.strip()

    def _record_solution(
        self,
        learn_trait: LearnTrait,
        task: str,
        result: dict[str, Any],
        content: str,
        summary: str,
    ) -> None:
        """Record solution to learning database."""
        learn_trait.learn.solutions.record(
            agent_name=self.name,
            problem=task,
            problem_context={
                "iterations": result.get("iterations", 0),
                "trace_id": result.get("trace_id", ""),
            },
            answer={
                "success": result.get("success", False),
                "output": content,
                "iterations": result.get("iterations", 0),
            },
            answer_text=summary,
            tokens_used=result.get("tokens_used", 0),
            latency_ms=0,
            category="execution",
            source="agent",
        )
        self._lg.debug(
            "solution persisted",
            extra={"agent": self.name, "tokens": result.get("tokens_used", 0)},
        )

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
        """Run execution pipeline.

        Executes the agent's task using the event dispatcher system.
        """
        return await self._run_with_dispatcher()

    async def _run_with_dispatcher(self) -> ExecutionResult:
        """Run execution using event orchestrator."""
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
        result = asyncio.run(self._ask_async(question))
        self._store_result(result)
        return result.content

    async def _ask_async(self, question: str) -> ExecutionResult:
        """Async implementation of ask() using event orchestration."""
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

    def _store_result(self, result: ExecutionResult) -> None:
        """Store result in recent history."""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent :]
