"""Core agent implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from appinfra.log import Logger
from llm_learn.collection import ScoredFact

from llm_agent.core.config import AgentConfig
from llm_agent.core.llm import CompletionResult, Message
from llm_agent.core.traits.base import Trait


if TYPE_CHECKING:
    from pydantic import BaseModel

    from llm_agent.core.task import Task, TaskResult
    from llm_agent.core.traits.learn import LearnTrait
    from llm_agent.core.traits.llm import LLMTrait
    from llm_agent.core.traits.tools import ToolsTrait
    from llm_agent.runtime.server.protocol.base import Request, Response


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
        from llm_agent.core.traits import LLMTrait, LLMConfig, LearnTrait, LearnConfig

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
        self._lg.info("Agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop the agent and all traits."""
        if not self._started:
            return

        for trait in self._traits.values():
            trait.on_stop()

        self._started = False
        self._lg.info("Agent stopped", extra={"agent": self.name})

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
        """Track response for feedback (bounded to prevent memory leaks)."""
        if len(self._response_contexts) >= self._config.max_tracked_responses:
            oldest_key = next(iter(self._response_contexts))
            del self._response_contexts[oldest_key]
        self._response_contexts[result.id] = _ResponseContext(
            system_prompt=prompt, query=query, response=result.content, model=result.model
        )

    def _build_prompt(self, base_prompt: str, query: str | None = None) -> str:
        """Build system prompt with directive and fact injection."""
        from llm_agent.core.traits.directive import DirectiveTrait
        from llm_agent.core.traits.learn import LearnTrait

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
        from llm_agent.core.traits.llm import LLMTrait
        from llm_agent.core.traits.tools import ToolsTrait

        self._lg.debug(
            "executing task",
            extra={
                "agent": self.name,
                "task": task.name,
                "has_schema": task.output_schema is not None,
            },
        )

        llm_trait = self.require_trait(LLMTrait)
        tools_trait = self.get_trait(ToolsTrait)

        # Build system prompt with directive/facts and task context
        base_prompt = task.system_prompt or self._config.default_prompt
        prompt = self._build_task_prompt(base_prompt, task)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=task.description),
        ]

        # Phase 1: Tool loop (if tools available)
        if tools_trait is not None and tools_trait.has_tools():
            return self._execute_with_tools(task, messages, llm_trait, tools_trait)

        # No tools - simple completion path
        return self._execute_simple(task, messages, llm_trait)

    def _build_task_prompt(self, base_prompt: str, task: Task) -> str:
        """Build system prompt with directive, facts, and task context."""
        # Start with directive/fact injection
        prompt = self._build_prompt(base_prompt, query=task.description)

        # Add task context if provided
        if task.context:
            context_lines = [f"- {k}: {v}" for k, v in task.context.items()]
            context_str = "\n".join(context_lines)
            prompt = f"{prompt}\n\n## Task Context\n{context_str}"

        return prompt

    def _execute_simple(
        self, task: Task, messages: list[Message], llm_trait: LLMTrait
    ) -> TaskResult:
        """Execute task without tools (simple completion)."""
        from llm_agent.core.llm.backend import StructuredOutputError
        from llm_agent.core.task import TaskResult

        try:
            result = llm_trait.complete(
                messages=messages,
                model=self._config.model,
                output_schema=task.output_schema,
            )
            return TaskResult(
                success=True,
                content=result.content,
                parsed=result.parsed,
                tool_calls=[],
                iterations=1,
                tokens_used=result.tokens_used,
            )
        except StructuredOutputError as e:
            return TaskResult(
                success=False,
                content="",
                error=f"Structured output error: {e}",
            )

    def _execute_with_tools(
        self,
        task: Task,
        messages: list[Message],
        llm_trait: LLMTrait,
        tools_trait: ToolsTrait,
    ) -> TaskResult:
        """Execute task with tool loop and optional structured extraction."""
        from llm_agent.core.task import TaskResult

        # Phase 1: Run tool loop
        exec_result, error = self._run_tool_loop(task, messages, llm_trait, tools_trait)
        if error:
            return TaskResult(success=False, content="", error=error)

        # Phase 2: Structured extraction if output_schema provided
        return self._build_tool_result(task, exec_result, llm_trait)

    def _run_tool_loop(
        self,
        task: Task,
        messages: list[Message],
        llm_trait: LLMTrait,
        tools_trait: ToolsTrait,
    ) -> tuple[Any, str | None]:
        """Run the tool execution loop. Returns (result, error)."""
        from llm_agent.core.tools.executor import ToolExecutor

        self._lg.debug("starting tool loop", extra={"agent": self.name, "task": task.name})

        executor = ToolExecutor(
            lg=self._lg,
            llm=_LLMTraitBackend(llm_trait),
            registry=tools_trait.registry,
            model=self._config.model,
        )

        try:
            result = executor.run(messages=messages, max_iterations=task.max_iterations)
            self._lg.debug(
                "tool loop completed",
                extra={"agent": self.name, "iterations": result.iterations},
            )
            return result, None
        except RuntimeError as e:
            self._lg.warning("tool loop failed", extra={"agent": self.name, "exception": e})
            return None, str(e)

    def _build_tool_result(
        self,
        task: Task,
        exec_result: Any,
        llm_trait: LLMTrait,
    ) -> TaskResult:
        """Build TaskResult from tool execution, with optional structured extraction."""
        from llm_agent.core.task import TaskResult

        parsed, final_content, extra_tokens = None, exec_result.content, 0

        if task.output_schema is not None:
            extraction = self._extract_structured_output(
                exec_result.content, task.output_schema, llm_trait
            )
            if extraction["error"]:
                return self._tool_result_with_error(exec_result, extraction["error"])
            parsed, final_content, extra_tokens = (
                extraction["parsed"],
                extraction["content"],
                extraction["tokens"],
            )

        return TaskResult(
            success=True,
            content=final_content,
            parsed=parsed,
            tool_calls=exec_result.tool_calls,
            iterations=exec_result.iterations,
            tokens_used=exec_result.total_tokens + extra_tokens,
        )

    def _tool_result_with_error(self, exec_result: Any, error: str) -> TaskResult:
        """Build a failed TaskResult from tool execution with an error."""
        from llm_agent.core.task import TaskResult

        return TaskResult(
            success=False,
            content=exec_result.content,
            tool_calls=exec_result.tool_calls,
            iterations=exec_result.iterations,
            tokens_used=exec_result.total_tokens,
            error=error,
        )

    def _extract_structured_output(
        self,
        tool_loop_content: str,
        output_schema: type[BaseModel],
        llm_trait: LLMTrait,
    ) -> dict[str, Any]:
        """Extract structured output from tool loop result.

        Returns:
            Dict with keys: parsed, content, tokens, error
        """
        from llm_agent.core.llm.backend import StructuredOutputError

        messages = self._build_extraction_messages(tool_loop_content)

        try:
            result = llm_trait.complete(
                messages=messages,
                model=self._config.model,
                output_schema=output_schema,
            )
            return {
                "parsed": result.parsed,
                "content": result.content,
                "tokens": result.tokens_used,
                "error": None,
            }
        except StructuredOutputError as e:
            return {
                "parsed": None,
                "content": "",
                "tokens": 0,
                "error": f"Structured output error: {e}",
            }

    def _build_extraction_messages(self, tool_loop_content: str) -> list[Message]:
        """Build messages for structured output extraction."""
        system_prompt = (
            "You are a data extraction assistant. "
            "Your task is to extract information and return it in the exact JSON format requested."
        )
        user_prompt = (
            "Based on the following information, extract the relevant data.\n\n"
            f"Information:\n{tool_loop_content}"
        )
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

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
        self, learn_trait: LearnTrait, ctx: _ResponseContext, correction: str
    ) -> None:
        """Record a preference pair for learning."""
        full_context = f"{ctx.system_prompt}\n\nUser: {ctx.query}"
        learn_trait.record_preference(
            context=full_context, chosen=correction, rejected=ctx.response
        )
        self._lg.debug("preference pair recorded", extra={"agent": self.name})

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
        from llm_agent.runtime.server.protocol.v1 import (
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
            from llm_agent.runtime.server.protocol.base import Response

            return Response(
                id=request.id,
                success=False,
                error=f"Unknown message type: {request.message_type}",
            )

        return handler(request)

    def _handle_health(self, request: Request) -> Response:
        """Handle health check request."""
        from llm_agent.runtime.server.protocol.v1 import HealthResponse

        return HealthResponse(id=request.id, status="ok", agent_name=self.name)

    def _handle_complete(self, request: Request) -> Response:
        """Handle completion request."""
        from llm_agent.runtime.server.protocol.v1 import CompleteRequest, CompleteResponse

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
        from llm_agent.runtime.server.protocol.v1 import RememberRequest, RememberResponse

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
        from llm_agent.runtime.server.protocol.v1 import ForgetRequest, ForgetResponse

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
        from llm_agent.runtime.server.protocol.v1 import RecallRequest, RecallResponse

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
        from llm_agent.runtime.server.protocol.v1 import FeedbackRequest, FeedbackResponse

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


class _LLMTraitBackend:
    """Adapter that wraps LLMTrait to satisfy LLMBackend protocol.

    Used internally by Agent.execute() to provide ToolExecutor with
    an LLMBackend-compatible interface.
    """

    def __init__(self, llm_trait: LLMTrait) -> None:
        self._trait = llm_trait

    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Delegate to LLMTrait.complete()."""
        return self._trait.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

    def load_adapter(self, adapter_path: str) -> None:
        """Not supported through trait adapter."""
        raise NotImplementedError("Adapter loading not supported via trait")

    def unload_adapter(self) -> None:
        """Not supported through trait adapter."""
        raise NotImplementedError("Adapter unloading not supported via trait")
