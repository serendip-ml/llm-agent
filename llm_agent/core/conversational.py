"""Conversational agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from appinfra.log import Logger
from llm_learn.core.types import ScoredEntity
from llm_learn.memory.atomic import Fact

from llm_agent.core.agent import Agent
from llm_agent.core.config import AgentConfig
from llm_agent.core.conversation import (
    Compactor,
    Conversation,
    SlidingWindowCompactor,
)
from llm_agent.core.helpers import ResponseContext
from llm_agent.core.llm import CompletionResult, Message
from llm_agent.core.task import Task, TaskResult


if TYPE_CHECKING:
    from llm_agent.core.conversation.runner import ConversationRunner
    from llm_agent.core.governor import GovernorLoop
    from llm_agent.core.tools.base import ToolCallResult
    from llm_agent.core.traits.learn import LearnTrait


class ConversationalAgent(Agent):
    """Agent with conversational LLM capabilities and conversation management.

    Provides:
    - complete(): Single-turn LLM completion
    - execute(): Task execution with tools and structured output
    - run_once(): Scheduled execution cycle with conversation persistence
    - ask(): Conversational question answering
    - Memory operations (remember/forget/recall)
    - Feedback recording

    Coordinates capabilities via traits:
    - LLMTrait: LLM completions
    - LearnTrait: Memory (facts), feedback, preferences
    - IdentityTrait: Identity/persona injection
    - MethodTrait: Operating method injection
    - ToolsTrait: Tool use
    - HTTPTrait: HTTP server

    Example:
        from appinfra.log import quick_console_logger
        from llm_agent import ConversationalAgent, AgentConfig
        from llm_agent.core.traits import LLMTrait, LLMConfig

        lg = quick_console_logger("agent", "info")
        config = AgentConfig(name="my-agent")

        agent = ConversationalAgent(lg, config)
        agent.add_trait(LLMTrait(LLMConfig(base_url="http://localhost:8000/v1")))
        agent.start()

        result = agent.complete("Hello!")
    """

    def __init__(
        self,
        lg: Logger,
        config: AgentConfig,
        conversation: Conversation | None = None,
        compactor: Compactor | None = None,
        default_task: Task | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            config: Agent configuration.
            conversation: Optional conversation for persistent context.
            compactor: Optional compaction strategy (defaults to SlidingWindowCompactor).
            default_task: Default task for execution (used when no task is provided).
        """
        super().__init__(lg)
        self._config = config
        self._response_contexts: dict[str, ResponseContext] = {}

        # Conversation management
        self._conversation = conversation
        self._compactor = compactor or SlidingWindowCompactor()
        self._results: list[TaskResult] = []
        self._max_results = 100
        self._cycle_count = 0
        self._pending_task: Task | None = None

        # Default task (created from config if not provided)
        self._default_task = default_task or Task(
            name="default",
            description=config.default_prompt,
        )

        # Governor loop (lazy-created when tools are available)
        self._governor_loop: GovernorLoop | None = None

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def conversation(self) -> Conversation | None:
        """Conversation for persistent context (if configured)."""
        return self._conversation

    @property
    def cycle_count(self) -> int:
        """Number of run_once() cycles completed."""
        return self._cycle_count

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the agent and all traits."""
        if self._started:
            return

        self._start_traits()
        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop the agent and all traits."""
        if not self._started:
            return

        self._stop_traits()
        self._started = False
        self._lg.info("Agent stopped", extra={"agent": self.name})

    def submit(self, task: Task) -> None:
        """Submit a task for execution.

        The task will be executed on the next run_once() cycle.
        For immediate execution, use execute() or run_task().

        Args:
            task: The task to execute.
        """
        self._pending_task = task
        self._lg.debug("task submitted", extra={"agent": self.name, "task": task.name})

    def _get_governor_loop(self) -> GovernorLoop | None:
        """Get or create the governor loop if tools are available.

        Returns:
            GovernorLoop if ToolsTrait is attached and has tools, None otherwise.
        """
        from llm_agent.core.governor import GovernorLoop
        from llm_agent.core.traits.llm import LLMTrait, LLMTraitBackend
        from llm_agent.core.traits.tools import ToolsTrait

        tools_trait = self.get_trait(ToolsTrait)
        if tools_trait is None or not tools_trait.has_tools():
            return None

        # Create loop lazily on first use
        if self._governor_loop is None:
            llm_trait = self.require_trait(LLMTrait)
            self._governor_loop = GovernorLoop(
                lg=self._lg,
                llm=LLMTraitBackend(llm_trait),
                registry=tools_trait.registry,
                model=self._config.model,
            )

        return self._governor_loop

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
        from llm_agent.core.traits.llm import LLMTrait

        llm_trait = self.require_trait(LLMTrait)

        base_prompt = system_prompt or self._config.default_prompt
        prompt = self._build_prompt(base_prompt, query=query)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=query),
        ]

        self._lg.debug(
            "completing query",
            extra={"agent": self.name, "query_len": len(query)},
        )

        result = llm_trait.complete(
            messages=messages,
            model=self._config.model,
        )

        self._lg.debug(
            "completion finished",
            extra={"agent": self.name, "model": result.model, "tokens": result.tokens_used},
        )

        self._track_response(result, prompt, query)
        return result

    def _track_response(self, result: CompletionResult, prompt: str, query: str) -> None:
        """Track response for feedback (bounded FIFO to prevent memory leaks).

        Uses FIFO eviction when limit is reached. The default limit of 100 responses
        should be sufficient for typical interactive use. For high-throughput scenarios,
        submit feedback promptly before responses are evicted.
        """
        if len(self._response_contexts) >= self._config.max_tracked_responses:
            oldest_key = next(iter(self._response_contexts))
            del self._response_contexts[oldest_key]
        self._response_contexts[result.id] = ResponseContext(
            system_prompt=prompt, query=query, response=result.content, model=result.model
        )

    def _build_prompt(self, base_prompt: str, query: str | None = None) -> str:
        """Build system prompt with identity and fact injection."""
        from llm_agent.core.traits.identity import IdentityTrait, MethodTrait
        from llm_agent.core.traits.learn import LearnTrait

        prompt = base_prompt

        # Inject identity if IdentityTrait is attached
        identity_trait = self.get_trait(IdentityTrait)
        if identity_trait is not None:
            prompt = identity_trait.build_prompt(prompt)

        # Inject method if MethodTrait is attached
        method_trait = self.get_trait(MethodTrait)
        if method_trait is not None:
            prompt = method_trait.build_prompt(prompt)

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
    # Task Execution
    # =========================================================================

    def execute(self, task: Task) -> TaskResult:
        """Execute a task with optional tools and structured output.

        This method provides a higher-level interface than complete(), supporting:
        - Tool use via ToolsTrait (iterative tool loop)
        - Structured output extraction via output_schema
        - Task context injection into prompts

        Requires LLMTrait. ToolsTrait is optional but required for tool use.

        Args:
            task: Task definition with description, context, and optional schema.

        Returns:
            TaskResult with final content, parsed output (if schema provided),
            and execution metadata.

        Raises:
            RuntimeError: If LLMTrait is not attached.

        Example:
            from pydantic import BaseModel

            class Answer(BaseModel):
                answer: str
                confidence: float

            result = agent.execute(Task(
                name="question",
                description="What is 2+2?",
                output_schema=Answer,
            ))
            print(result.parsed.answer)  # "4"
        """
        from llm_agent.core.task_executor import TaskExecutor
        from llm_agent.core.traits.llm import LLMTrait

        executor = TaskExecutor(
            lg=self._lg,
            llm_trait=self.require_trait(LLMTrait),
            governor_loop=self._get_governor_loop(),
            prompt_builder=self._build_prompt,
        )
        result = executor.execute(task)
        self._record_solution(task, result)
        return result

    # =========================================================================
    # Scheduled Execution (run_once)
    # =========================================================================

    def run_once(self) -> TaskResult:
        """Execute one cycle, maintaining conversation context.

        Requires conversation to be configured (via constructor or from_config).
        Maintains persistent conversation across calls for continuous learning.

        Returns:
            TaskResult from this execution cycle.

        Raises:
            RuntimeError: If conversation not configured.
        """
        if self._conversation is None:
            raise RuntimeError(
                "run_once() requires conversation - use from_config() or pass conversation to constructor"
            )

        self._cycle_count += 1
        is_first_run = self._cycle_count == 1

        # Determine which task will be executed
        task = self._pending_task or self._default_task

        self._log_run_start(is_first_run)
        result = self._create_conversation_runner().run(
            task=self._pending_task, is_first_run=is_first_run
        )
        self._log_run_complete(result)

        self._store_result(result)
        if task is not None:
            self._record_solution(task, result)
        self._pending_task = None
        return result

    def run_task(self, task: Task) -> TaskResult:
        """Execute a specific task immediately via run_once().

        Args:
            task: The task to execute.

        Returns:
            TaskResult from execution.
        """
        self.submit(task)
        return self.run_once()

    def _create_conversation_runner(self) -> ConversationRunner:
        """Create a ConversationRunner for this agent."""
        from llm_agent.core.conversation.runner import ConversationRunner
        from llm_agent.core.traits.llm import LLMTrait

        assert self._conversation is not None

        return ConversationRunner(
            lg=self._lg,
            conversation=self._conversation,
            compactor=self._compactor,
            llm_trait=self.require_trait(LLMTrait),
            governor_loop=self._get_governor_loop(),
            default_task=self._default_task,
            identity_builder=self._build_identity_prompt,
            _agent_name=self.name,
        )

    def _build_identity_prompt(self) -> str:
        """Build identity/system prompt for conversation initialization."""
        from llm_agent.core.traits.identity import IdentityTrait, MethodTrait

        prompt = ""
        identity_trait = self.get_trait(IdentityTrait)
        if identity_trait is not None:
            prompt = identity_trait.build_prompt("")

        method_trait = self.get_trait(MethodTrait)
        if method_trait is not None:
            prompt = method_trait.build_prompt(prompt)

        return prompt

    def _log_run_start(self, is_first_run: bool) -> None:
        """Log task start with cycle info."""
        assert self._conversation is not None
        self._lg.debug(
            "running task",
            extra={
                "agent": self.name,
                "cycle": self._cycle_count,
                "is_first": is_first_run,
                "conv_tokens": self._conversation.token_count,
            },
        )

    def _log_run_complete(self, result: TaskResult) -> None:
        """Log task completion with result info."""
        assert self._conversation is not None
        self._lg.info(
            "task completed",
            extra={
                "agent": self.name,
                "cycle": self._cycle_count,
                "success": result.success,
                "iterations": result.iterations,
                "tool_calls": len(result.tool_calls),
                "conv_tokens": self._conversation.token_count,
            },
        )

    def _store_result(self, result: TaskResult) -> None:
        """Store result in bounded history."""
        self._results.append(result)
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results :]

    def _record_solution(self, task: Task, result: TaskResult) -> None:
        """Record task execution as a solution in llm-learn.

        Only records if LearnTrait is available. Logs but doesn't raise on failure.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            return

        try:
            learn_trait.learn.solutions.record(
                agent_name=self.name,
                problem=task.description,
                problem_context=self._build_problem_context(task),
                answer=self._build_answer_dict(result),
                answer_text=result.content[:1000] if result.content else None,
                tokens_used=result.tokens_used,
                latency_ms=0,
                tool_calls=self._serialize_tool_calls(result.tool_calls),
                category=task.name,
            )
            self._lg.debug("solution recorded", extra={"agent": self.name, "task": task.name})
        except Exception as e:
            self._lg.warning(
                "failed to record solution",
                extra={"agent": self.name, "task": task.name, "exception": e},
            )

    def _build_answer_dict(self, result: TaskResult) -> dict[str, Any]:
        """Build answer dict from task result for solution recording."""
        answer: dict[str, Any] = {"success": result.success, "content": result.content}
        if result.parsed is not None:
            if hasattr(result.parsed, "model_dump"):
                answer["parsed"] = result.parsed.model_dump()
            else:
                answer["parsed"] = result.parsed
        if result.completion is not None:
            answer["completion"] = result.completion.model_dump()
        if result.error:
            answer["error"] = result.error
        return answer

    def _build_problem_context(self, task: Task) -> dict[str, Any]:
        """Build problem context dict for solution recording."""
        context: dict[str, Any] = {"task_name": task.name, "task_context": task.context}
        if task.output_schema is not None:
            context["output_schema"] = task.output_schema.__name__
        return context

    def _serialize_tool_calls(
        self, tool_calls: list[ToolCallResult] | None
    ) -> list[dict[str, Any]] | None:
        """Convert tool calls to serializable format for solution recording."""
        if not tool_calls:
            return None
        return [{"name": tc.name, "success": tc.result.success} for tc in tool_calls]

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def reset_conversation(self) -> None:
        """Reset conversation to start fresh.

        Use this when you want the agent to start a new exploration
        without any prior context.
        """
        if self._conversation is not None:
            self._conversation.clear()
        self._cycle_count = 0
        self._lg.info("conversation reset", extra={"agent": self.name})

    def get_recent_results(self, limit: int = 10) -> list[TaskResult]:
        """Get recent task execution results.

        Args:
            limit: Maximum results to return.

        Returns:
            List of recent TaskResults, most recent last.
        """
        return self._results[-limit:]

    # =========================================================================
    # Conversational Interface
    # =========================================================================

    def ask(self, question: str) -> str:
        """Ask the agent a conversational question.

        Uses the agent's LLM with knowledge of its purpose and
        recent discoveries to answer questions like:
        - "What have you learned?"
        - "What are you working on?"

        Args:
            question: Question to ask.

        Returns:
            Agent's response.
        """
        self._lg.debug("ask received", extra={"agent": self.name, "question_len": len(question)})

        # Build context from recent results
        context = self._build_context_for_ask()

        # Build base prompt - identity is injected by _build_prompt() via complete()
        system_prompt = f"""You are {self.name}, a learning agent.

{context}

Answer the user's question based on your purpose and what you've learned."""

        result = self.complete(question, system_prompt=system_prompt)

        self._lg.debug(
            "ask completed",
            extra={"agent": self.name, "response_len": len(result.content)},
        )
        return result.content

    def _build_context_for_ask(self) -> str:
        """Build context string from recent results for ask()."""
        if not self._results:
            return "You haven't completed any tasks yet."

        lines = ["## Recent Activity"]
        for i, result in enumerate(self._results[-5:], 1):
            status = "completed" if result.success else "failed"
            content_preview = (
                result.content[:200] + "..." if len(result.content) > 200 else result.content
            )
            lines.append(f"{i}. Task {status}: {content_preview}")

        return "\n".join(lines)

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
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)
        fact_id = learn_trait.remember(fact, category=category)
        self._lg.debug(
            "fact stored",
            extra={"agent": self.name, "fact_id": fact_id, "category": category},
        )
        return fact_id

    def forget(self, fact_id: int) -> None:
        """Remove a stored fact.

        Requires LearnTrait.

        Args:
            fact_id: ID of the fact to remove.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)
        learn_trait.forget(fact_id)
        self._lg.debug("fact removed", extra={"agent": self.name, "fact_id": fact_id})

    def recall(
        self,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        categories: list[str] | None = None,
    ) -> list[ScoredEntity[Fact]]:
        """Search facts by semantic similarity to query.

        Requires LearnTrait with embedder configured.

        Args:
            query: Text to search for similar facts.
            top_k: Max results (defaults to config.rag_top_k).
            min_similarity: Minimum similarity (defaults to config.rag_min_similarity).
            categories: Filter to these categories.

        Returns:
            List of ScoredEntity[Fact] sorted by similarity (highest first).
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)
        results = learn_trait.recall(
            query=query,
            top_k=top_k if top_k is not None else self._config.rag_top_k,
            min_similarity=min_similarity
            if min_similarity is not None
            else self._config.rag_min_similarity,
            categories=categories,
        )
        self._lg.debug(
            "facts recalled",
            extra={"agent": self.name, "query_len": len(query), "results": len(results)},
        )
        return results

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
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.require_trait(LearnTrait)

        # Pop to remove after use - prevents unbounded memory growth
        ctx = self._response_contexts.pop(response_id, None)
        if ctx is None:
            raise ValueError(f"Unknown response_id: {response_id}")

        # Record feedback signal
        learn_trait.record_feedback(
            content=ctx.response, signal=signal, context={"query": ctx.query, "model": ctx.model}
        )
        self._lg.debug("feedback recorded", extra={"agent": self.name, "signal": signal})

        # If negative with correction, create preference pair
        if signal == "negative" and correction is not None:
            self._record_preference_pair(learn_trait, ctx, correction)

    def _record_preference_pair(
        self, learn_trait: LearnTrait, ctx: ResponseContext, correction: str
    ) -> None:
        """Record a preference pair for learning."""
        full_context = f"{ctx.system_prompt}\n\nUser: {ctx.query}"
        learn_trait.record_preference(
            context=full_context, chosen=correction, rejected=ctx.response
        )
        self._lg.debug("preference pair recorded", extra={"agent": self.name})

    def record_feedback(self, signal: str, context: dict[str, Any] | None = None) -> None:
        """Record feedback on the agent's behavior (simple version).

        Args:
            signal: Feedback signal (e.g., "helpful", "wrong", "good insight").
            context: Additional context about the feedback.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.get_trait(LearnTrait)
        if learn_trait is None:
            self._lg.warning("feedback ignored - LearnTrait not configured")
            return

        learn_trait.record_feedback(
            content=signal,
            signal="positive"
            if "good" in signal.lower() or "helpful" in signal.lower()
            else "negative",
            context=context or {},
        )
