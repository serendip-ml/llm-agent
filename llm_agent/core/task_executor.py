"""Task execution with tools and structured output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from pydantic import BaseModel

from llm_agent.core.llm.types import Message
from llm_agent.core.task import Task, TaskResult


if TYPE_CHECKING:
    from llm_agent.core.traits.llm import LLMTrait
    from llm_agent.core.traits.tools import ToolsTrait


# Type alias for prompt builder function
PromptBuilder = Callable[[str, str], str]
"""Function that builds a full prompt from (base_prompt, query) -> enriched_prompt."""


@dataclass
class TaskExecutor:
    """Executes tasks with optional tools and structured output.

    Provides task execution with:
    - Tool use via ToolsTrait (iterative tool loop)
    - Structured output extraction via output_schema
    - Task context injection into prompts

    The executor is decoupled from the agent - it receives traits and a prompt
    builder function rather than depending on the agent directly.

    Example:
        from llm_agent.core.task_executor import TaskExecutor

        executor = TaskExecutor(
            lg=lg,
            llm_trait=llm_trait,
            tools_trait=tools_trait,
            model="gpt-4",
            prompt_builder=agent._build_prompt,
        )
        result = executor.execute(task)
    """

    lg: Logger
    llm_trait: LLMTrait
    tools_trait: ToolsTrait | None
    model: str | None
    prompt_builder: PromptBuilder

    def execute(self, task: Task) -> TaskResult:
        """Execute a task with optional tools and structured output.

        Args:
            task: Task definition with description, context, and optional schema.

        Returns:
            TaskResult with final content, parsed output (if schema provided),
            and execution metadata.
        """
        self.lg.debug(
            "executing task",
            extra={
                "task": task.name,
                "has_schema": task.output_schema is not None,
            },
        )

        # Build system prompt with identity/facts and task context
        base_prompt = task.system_prompt or ""
        prompt = self._build_task_prompt(base_prompt, task)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=task.description),
        ]

        # Use tool loop if tools available, otherwise simple completion
        if self.tools_trait is not None and self.tools_trait.has_tools():
            return self._execute_with_tools(task, messages)
        return self._execute_simple(task, messages)

    def _build_task_prompt(self, base_prompt: str, task: Task) -> str:
        """Build system prompt with identity, facts, and task context."""
        # Start with identity/fact injection via the prompt builder
        prompt = self.prompt_builder(base_prompt, task.description)

        # Add task context if provided
        if task.context:
            context_lines = [f"- {k}: {v}" for k, v in task.context.items()]
            context_str = "\n".join(context_lines)
            prompt = f"{prompt}\n\n## Task Context\n{context_str}"

        return prompt

    def _execute_simple(self, task: Task, messages: list[Message]) -> TaskResult:
        """Execute task without tools (simple completion)."""
        from llm_agent.core.llm.backend import StructuredOutputError

        try:
            result = self.llm_trait.complete(
                messages=messages,
                model=self.model,
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

    def _execute_with_tools(self, task: Task, messages: list[Message]) -> TaskResult:
        """Execute task with tool loop and optional structured extraction."""
        # Phase 1: Run tool loop
        exec_result, error = self._run_tool_loop(task, messages)
        if error:
            return TaskResult(success=False, content="", error=error)

        # Phase 2: Structured extraction if output_schema provided
        return self._build_tool_result(task, exec_result)

    def _run_tool_loop(self, task: Task, messages: list[Message]) -> tuple[Any, str | None]:
        """Run the tool execution loop. Returns (result, error)."""
        from llm_agent.core.tools.executor import ToolExecutor
        from llm_agent.core.traits.llm import LLMTraitBackend

        assert self.tools_trait is not None

        self.lg.debug("starting tool loop", extra={"task": task.name})

        executor = ToolExecutor(
            lg=self.lg,
            llm=LLMTraitBackend(self.llm_trait),
            registry=self.tools_trait.registry,
            task=task,
            model=self.model,
        )

        try:
            result = executor.run(messages=messages)
            self.lg.debug("tool loop completed", extra={"iterations": result.iterations})
            return result, None
        except RuntimeError as e:
            self.lg.warning("tool loop failed", extra={"exception": e})
            return None, str(e)

    def _build_tool_result(self, task: Task, exec_result: Any) -> TaskResult:
        """Build TaskResult from tool execution, with optional structured extraction."""
        parsed, final_content, extra_tokens = None, exec_result.content, 0

        if task.output_schema is not None:
            extraction = self._extract_structured_output(exec_result.content, task.output_schema)
            if extraction["error"]:
                return self._tool_result_with_error(exec_result, extraction["error"])
            parsed = extraction["parsed"]
            final_content = extraction["content"]
            extra_tokens = extraction["tokens"]

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
        return TaskResult(
            success=False,
            content=exec_result.content,
            tool_calls=exec_result.tool_calls,
            iterations=exec_result.iterations,
            tokens_used=exec_result.total_tokens,
            error=error,
        )

    def _extract_structured_output(
        self, tool_loop_content: str, output_schema: type[BaseModel]
    ) -> dict[str, Any]:
        """Extract structured output from tool loop result.

        Returns:
            Dict with keys: parsed, content, tokens, error
        """
        from llm_agent.core.llm.backend import StructuredOutputError

        messages = self._build_extraction_messages(tool_loop_content)

        try:
            result = self.llm_trait.complete(
                messages=messages,
                model=self.model,
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
