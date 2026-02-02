"""Conversation-based task execution runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from llm_agent.core.conversation import Compactor, Conversation
from llm_agent.core.llm.types import Message
from llm_agent.core.task import Task, TaskResult


if TYPE_CHECKING:
    from llm_agent.core.traits.llm import LLMTrait
    from llm_agent.core.traits.tools import ToolsTrait


# Type alias for identity prompt builder
IdentityBuilder = Callable[[], str]
"""Function that builds the identity/system prompt."""


@dataclass
class ConversationRunner:
    """Runs conversation-based task execution with tool support.

    Manages the conversation lifecycle:
    - Prepares conversation (compaction, initialization/continuation)
    - Runs tool execution loop
    - Builds results with optional structured output extraction

    Example:
        runner = ConversationRunner(
            lg=lg,
            conversation=conversation,
            compactor=compactor,
            llm_trait=llm_trait,
            tools_trait=tools_trait,
            model="gpt-4",
            default_task=Task(name="default", description="..."),
            identity_builder=lambda: agent._build_identity_prompt(),
        )
        result = runner.run(task=task, is_first_run=True)
    """

    lg: Logger
    conversation: Conversation
    compactor: Compactor
    llm_trait: LLMTrait
    tools_trait: ToolsTrait | None
    model: str | None
    default_task: Task
    identity_builder: IdentityBuilder

    _agent_name: str = field(default="agent", repr=False)

    def run(self, task: Task | None, is_first_run: bool) -> TaskResult:
        """Run one conversation cycle.

        Args:
            task: Task to execute (uses default prompt if None).
            is_first_run: Whether this is the first run (initializes vs continues).

        Returns:
            TaskResult from this execution cycle.
        """
        self._prepare_conversation(is_first_run, task)
        return self._run_conversation_loop(task)

    # -------------------------------------------------------------------------
    # Conversation Preparation
    # -------------------------------------------------------------------------

    def _prepare_conversation(self, is_first_run: bool, task: Task | None) -> None:
        """Prepare conversation for this run."""
        if self.conversation.needs_compaction():
            self._compact_conversation()

        if is_first_run:
            self._initialize_conversation(task)
        else:
            self._continue_conversation(task)

    def _initialize_conversation(self, task: Task | None) -> None:
        """Initialize conversation with system prompt and task for first run."""
        # Build system prompt using identity builder
        system_prompt = self.identity_builder()
        self.conversation.add_system(system_prompt)

        # Use task description or default task description
        effective_task = task or self.default_task
        self.conversation.add_user(effective_task.description)

    def _continue_conversation(self, task: Task | None) -> None:
        """Add continuation prompt for subsequent runs."""
        if task:
            self.conversation.add_user(task.description)
        else:
            continue_prompt = (
                "Continue your exploration and analysis. Build on what you've learned so far. "
                "Focus on areas you haven't fully explored yet, or dive deeper into interesting "
                "findings. Use your tools to gather more information."
            )
            self.conversation.add_user(continue_prompt)

    def _compact_conversation(self) -> None:
        """Compact conversation when approaching token limit."""
        before_tokens = self.conversation.token_count
        self.compactor.compact(self.conversation)
        after_tokens = self.conversation.token_count

        self.lg.info(
            "conversation compacted",
            extra={
                "agent": self._agent_name,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "reduction": before_tokens - after_tokens,
            },
        )

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def _run_conversation_loop(self, task: Task | None) -> TaskResult:
        """Run tool executor with current conversation state."""
        if self.tools_trait is None or not self.tools_trait.has_tools():
            return self._run_simple_completion(task)
        return self._run_with_tools(task)

    def _run_simple_completion(self, task: Task | None) -> TaskResult:
        """Run simple completion without tools."""
        result = self.llm_trait.complete(
            messages=self.conversation.messages(),
            model=self.model,
        )
        self.conversation.add_assistant(result.content)

        # Handle structured output extraction if requested
        parsed, extra_tokens = None, 0
        if task is not None and task.output_schema is not None:
            extraction = self._try_extract_structured_simple(result, task.output_schema)
            if isinstance(extraction, TaskResult):
                return extraction
            parsed, extra_tokens = extraction

        return TaskResult(
            success=True,
            content=result.content,
            parsed=parsed,
            tool_calls=[],
            iterations=1,
            tokens_used=result.tokens_used + extra_tokens,
        )

    def _run_with_tools(self, task: Task | None) -> TaskResult:
        """Run tool executor loop."""
        from llm_agent.core.tools.executor import ToolExecutor
        from llm_agent.core.traits.llm import LLMTraitBackend

        assert self.tools_trait is not None

        effective_task = task or self.default_task
        executor = ToolExecutor(
            lg=self.lg,
            llm=LLMTraitBackend(self.llm_trait),
            registry=self.tools_trait.registry,
            task=effective_task,
            model=self.model,
        )

        try:
            messages = self.conversation.messages()
            exec_result = executor.run(messages=messages)
            self._update_conversation_from_execution(exec_result.messages, len(messages))
            return self._build_task_result(effective_task, exec_result)
        except RuntimeError as e:
            self.lg.warning("tool loop failed", extra={"agent": self._agent_name, "exception": e})
            return TaskResult(success=False, content="", error=str(e))

    def _update_conversation_from_execution(
        self, messages: list[Message], original_count: int
    ) -> None:
        """Update conversation with messages added during tool execution."""
        for msg in messages[original_count:]:
            self.conversation.add(msg)

    # -------------------------------------------------------------------------
    # Result Building
    # -------------------------------------------------------------------------

    def _build_task_result(self, task: Task, exec_result: Any) -> TaskResult:
        """Build TaskResult from tool execution."""
        parsed, final_content, extra_tokens = None, exec_result.content, 0

        if task.output_schema is not None:
            extraction = self._try_extract_structured(exec_result, task.output_schema)
            if isinstance(extraction, TaskResult):
                return extraction
            parsed, final_content, extra_tokens = extraction

        completion = self._extract_completion(exec_result.terminal_data)

        return TaskResult(
            success=True,
            content=final_content,
            parsed=parsed,
            completion=completion,
            tool_calls=exec_result.tool_calls,
            iterations=exec_result.iterations,
            tokens_used=exec_result.total_tokens + extra_tokens,
        )

    def _extract_completion(self, terminal_data: dict[str, Any] | None) -> Any:
        """Extract TaskCompletion from terminal_data if valid."""
        from llm_agent.core.task import TaskCompletion, TaskStatus

        if not terminal_data:
            return None
        status = terminal_data.get("status")
        conclusion = terminal_data.get("conclusion")
        if not (status and conclusion):
            return None
        try:
            return TaskCompletion(status=TaskStatus(status), conclusion=conclusion)
        except ValueError:
            self.lg.warning(
                "invalid terminal status", extra={"status": status, "agent": self._agent_name}
            )
            return None

    def _try_extract_structured(
        self, exec_result: Any, output_schema: type
    ) -> tuple[Any, str, int] | TaskResult:
        """Try to extract structured output, return tuple or error TaskResult."""
        from llm_agent.core.llm.backend import StructuredOutputError

        messages = self._build_extraction_messages(exec_result.content)

        try:
            result = self.llm_trait.complete(
                messages=messages,
                model=self.model,
                output_schema=output_schema,
            )
            return result.parsed, result.content, result.tokens_used
        except StructuredOutputError as e:
            return TaskResult(
                success=False,
                content=exec_result.content,
                tool_calls=exec_result.tool_calls,
                iterations=exec_result.iterations,
                tokens_used=exec_result.total_tokens,
                error=f"Structured output error: {e}",
            )

    def _try_extract_structured_simple(
        self, completion_result: Any, output_schema: type
    ) -> tuple[Any, int] | TaskResult:
        """Try to extract structured output from simple completion."""
        from llm_agent.core.llm.backend import StructuredOutputError

        messages = self._build_extraction_messages(completion_result.content)

        try:
            result = self.llm_trait.complete(
                messages=messages,
                model=self.model,
                output_schema=output_schema,
            )
            return result.parsed, result.tokens_used
        except StructuredOutputError as e:
            return TaskResult(
                success=False,
                content=completion_result.content,
                tool_calls=[],
                iterations=1,
                tokens_used=completion_result.tokens_used,
                error=f"Structured output error: {e}",
            )

    def _build_extraction_messages(self, content: str) -> list[Message]:
        """Build messages for structured output extraction."""
        system_prompt = (
            "You are a data extraction assistant. "
            "Your task is to extract information and return it in the exact JSON format requested."
        )
        user_prompt = (
            "Based on the following information, extract the relevant data.\n\n"
            f"Information:\n{content}"
        )
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
