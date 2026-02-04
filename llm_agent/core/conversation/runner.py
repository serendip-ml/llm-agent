"""Conversation-based task execution runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from llm_agent.core.conversation import Compactor, Conversation
from llm_agent.core.llm.types import Message
from llm_agent.core.task import Task, TaskResult


if TYPE_CHECKING:
    from llm_saia import SAIA

    from llm_agent.core.traits.llm import LLMTrait


# Type alias for identity prompt builder
IdentityBuilder = Callable[[], str]
"""Function that builds the identity/system prompt."""


@dataclass
class ConversationRunner:
    """Runs conversation-based task execution with optional SAIA integration.

    Manages the conversation lifecycle:
    - Prepares conversation (compaction, initialization/continuation)
    - Runs task execution via SAIA (if available) or simple LLM completion
    - Builds results with optional structured output extraction

    Example:
        runner = ConversationRunner(
            lg=lg,
            conversation=conversation,
            compactor=compactor,
            llm_trait=llm_trait,
            saia=saia,  # Optional - enables tool use
            default_task=Task(name="default", description="..."),
            identity_builder=lambda: agent._build_identity_prompt(),
        )
        result = runner.run(task=task, is_first_run=True)
    """

    lg: Logger
    conversation: Conversation
    compactor: Compactor
    llm_trait: LLMTrait | None
    saia: SAIA | None
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
        """Run task execution with current conversation state."""
        if self.saia is not None:
            return self._run_with_saia(task)
        return self._run_simple_completion(task)

    def _run_simple_completion(self, task: Task | None) -> TaskResult:
        """Run simple completion without tools (requires llm_trait)."""
        if self.llm_trait is None:
            raise RuntimeError("Either SAIA or LLMTrait required for execution")

        result = self.llm_trait.complete(messages=self.conversation.messages())
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

    def _run_with_saia(self, task: Task | None) -> TaskResult:
        """Run task execution with SAIA."""
        assert self.saia is not None

        effective_task = task or self.default_task

        try:
            # Build task description from conversation context
            messages = self.conversation.messages()
            task_context = self._build_task_from_conversation(messages, effective_task)

            # Run SAIA complete (async)
            saia_result = asyncio.get_event_loop().run_until_complete(
                self.saia.complete(task_context)
            )

            # Update conversation with result
            self.conversation.add_assistant(saia_result.output)

            # Build result
            return self._build_saia_result(effective_task, saia_result)

        except Exception as e:
            self.lg.warning(
                "SAIA execution failed",
                extra={"agent": self._agent_name, "exception": e},
            )
            return TaskResult(success=False, content="", error=str(e))

    def _build_task_from_conversation(self, messages: list[Message], task: Task) -> str:
        """Build task description incorporating conversation context."""
        # Extract system prompt and recent messages for context
        context_parts = []

        for msg in messages[-5:]:  # Last 5 messages for context
            if msg.role == "system":
                context_parts.append(f"Context: {msg.content[:500]}")
            elif msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                context_parts.append(f"Previous response: {msg.content[:200]}...")

        context = "\n".join(context_parts)
        return f"{context}\n\nCurrent task: {task.description}"

    def _build_saia_result(self, task: Task, saia_result: Any) -> TaskResult:
        """Build TaskResult from SAIA execution result."""
        parsed = None

        # Handle structured output extraction if requested
        if task.output_schema is not None and saia_result.completed:
            parsed = self._try_extract_with_saia(saia_result.output, task.output_schema)

        # Extract completion info from terminal data
        completion = self._extract_completion(saia_result.terminal_data)

        return TaskResult(
            success=saia_result.completed,
            content=saia_result.output,
            parsed=parsed,
            completion=completion,
            tool_calls=[],  # SAIA tracks tool calls differently
            iterations=saia_result.iterations,
            tokens_used=0,  # SAIA tracks tokens differently
        )

    def _try_extract_with_saia(self, content: str, output_schema: type) -> Any:
        """Try to extract structured output using SAIA extract verb."""
        if self.saia is None:
            return None

        try:
            return asyncio.get_event_loop().run_until_complete(
                self.saia.extract(content, output_schema)
            )
        except Exception as e:
            self.lg.warning(
                "SAIA extraction failed",
                extra={"agent": self._agent_name, "exception": e},
            )
            return None

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
                "invalid terminal status",
                extra={"status": status, "agent": self._agent_name},
            )
            return None

    # -------------------------------------------------------------------------
    # Simple completion helpers (when no SAIA)
    # -------------------------------------------------------------------------

    def _try_extract_structured_simple(
        self, completion_result: Any, output_schema: type
    ) -> tuple[Any, int] | TaskResult:
        """Try to extract structured output from simple completion."""
        if self.llm_trait is None:
            return TaskResult(
                success=False,
                content=completion_result.content,
                error="LLMTrait required for structured extraction",
            )

        from llm_agent.core.llm.backend import StructuredOutputError

        messages = self._build_extraction_messages(completion_result.content)

        try:
            result = self.llm_trait.complete(
                messages=messages,
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
