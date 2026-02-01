"""Prompt-only agent framework for YAML-configured agents.

This module provides the infrastructure for Class 2 agents - agents that are
configured entirely via YAML with no agent-specific Python code.

Example YAML config:
    name: codebase-explorer
    directive:
      prompt: |
        You are a codebase exploration agent...
    task:
      description: Explore the codebase and discover insights.
      output_schema:
        type: object
        properties:
          insight: {type: string}
          confidence: {type: number}
    tools:
      shell: {allowed_commands: [grep, find, git, ls]}
      remember: {}
      recall: {}
    schedule:
      interval: 600  # Run every 10 minutes
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from llm_agent.core.agent import Agent
from llm_agent.core.config import AgentConfig
from llm_agent.core.conversation import (
    Compactor,
    Conversation,
    ConversationConfig,
    SlidingWindowCompactor,
)
from llm_agent.core.llm import Message
from llm_agent.core.task import Task, TaskResult
from llm_agent.core.tools.registry import ToolRegistry
from llm_agent.core.traits.directive import Directive, DirectiveTrait
from llm_agent.core.traits.llm import LLMConfig, LLMTrait
from llm_agent.core.traits.tools import ToolsTrait


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.traits.learn import LearnTrait


class ToolConfig(BaseModel):
    """Configuration for a single tool."""

    type: str
    """Tool type (e.g., 'shell', 'read_file', 'remember')."""

    config: dict[str, Any] = {}
    """Tool-specific configuration."""


class TaskConfig(BaseModel):
    """Configuration for agent's primary task."""

    description: str
    """What the agent should do."""

    output_schema: dict[str, Any] | None = None
    """JSON Schema for structured output (optional)."""


class ScheduleConfig(BaseModel):
    """Configuration for scheduled execution."""

    interval: int
    """Seconds between executions."""


class ConversationYAMLConfig(BaseModel):
    """YAML-friendly conversation configuration."""

    max_tokens: int = 32000
    """Maximum tokens before compaction is required."""

    compact_threshold: float = 0.8
    """Trigger compaction when usage exceeds this fraction (0.0-1.0)."""

    min_recent_messages: int = 4
    """Minimum recent messages to preserve during compaction."""


class PromptOnlyAgentConfig(BaseModel):
    """Configuration for a prompt-only agent loaded from YAML."""

    name: str
    """Agent identifier."""

    directive: Directive
    """Agent's purpose and behavior."""

    task: TaskConfig
    """Primary task configuration."""

    tools: dict[str, dict[str, Any]] = {}
    """Tool configurations keyed by tool type."""

    schedule: ScheduleConfig | None = None
    """Optional schedule for automatic execution."""

    conversation: ConversationYAMLConfig = ConversationYAMLConfig()
    """Conversation management configuration."""


def _substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Substitute {{VAR}} patterns in text with variable values.

    Uses {{VAR}} syntax to avoid conflicts with appinfra's ${VAR} resolution.
    Falls back to environment variables if not in variables dict.

    Args:
        text: Text containing {{VAR}} patterns.
        variables: Variable name to value mapping.

    Returns:
        Text with variables substituted.

    Raises:
        ValueError: If a variable is not found.
    """
    pattern = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return variables[var_name]
        if var_name in os.environ:
            return os.environ[var_name]
        raise ValueError(f"Variable {{{{{var_name}}}}} not found in variables or environment")

    return pattern.sub(replacer, text)


def _substitute_in_dict(data: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute variables in a dict/list structure."""
    if isinstance(data, str):
        return _substitute_variables(data, variables)
    if isinstance(data, dict):
        return {k: _substitute_in_dict(v, variables) for k, v in data.items()}
    if isinstance(data, list):
        return [_substitute_in_dict(item, variables) for item in data]
    return data


def _create_agent_with_traits(
    lg: Logger,
    config: PromptOnlyAgentConfig,
    llm_config: LLMConfig,
    learn_trait: LearnTrait | None,
) -> Agent:
    """Create core Agent with all traits configured."""
    agent_config = AgentConfig(name=config.name, default_prompt="")
    agent = Agent(lg, agent_config)

    agent.add_trait(LLMTrait(llm_config))
    agent.add_trait(DirectiveTrait(config.directive))

    if learn_trait is not None:
        agent.add_trait(learn_trait)

    tools_trait = ToolsTrait()
    _setup_tools(tools_trait.registry, config.tools, learn_trait)
    agent.add_trait(tools_trait)

    return agent


def _create_conversation(config: PromptOnlyAgentConfig) -> Conversation:
    """Create Conversation with config from agent config."""
    conv_config = ConversationConfig(
        max_tokens=config.conversation.max_tokens,
        compact_threshold=config.conversation.compact_threshold,
        min_recent_messages=config.conversation.min_recent_messages,
    )
    return Conversation(config=conv_config)


@dataclass
class PromptOnlyAgent:
    """Framework class for YAML-configured agents (Class 2).

    PromptOnlyAgent wraps the core Agent class, providing:
    - Variable substitution ({{VAR}})
    - Automatic tool setup from config
    - Task execution with structured output
    - Conversational interface via ask()
    - Persistent conversation across scheduled runs

    The agent maintains conversation context between run_once() calls,
    allowing it to build understanding over time rather than starting
    fresh each cycle.

    Example:
        from appinfra.log import Logger

        lg = Logger.create("agent")
        config_dict = {"name": "explorer", "directive": {...}, "task": {...}}
        agent = PromptOnlyAgent.from_dict(
            lg=lg,
            config_dict=config_dict,
            llm_config=LLMConfig(base_url="http://localhost:8000/v1"),
            variables={"CODEBASE_PATH": "/path/to/code"},
        )
        agent.start()
        result = agent.run_once()  # First exploration
        result = agent.run_once()  # Continues from where it left off
        print(result.parsed)
    """

    lg: Logger = field(repr=False)
    """Logger instance."""

    config: PromptOnlyAgentConfig
    """Loaded configuration."""

    agent: Agent = field(repr=False)
    """Underlying Agent instance."""

    conversation: Conversation = field(repr=False)
    """Persistent conversation across run_once() calls."""

    compactor: Compactor = field(repr=False)
    """Compaction strategy for when conversation gets too long."""

    _results: list[TaskResult] = field(default_factory=list, repr=False)
    """Recent task results (bounded)."""

    _max_results: int = field(default=100, repr=False)
    """Maximum results to keep."""

    _cycle_count: int = field(default=0, repr=False)
    """Number of run_once() cycles completed."""

    @classmethod
    def from_dict(
        cls,
        lg: Logger,
        config_dict: dict[str, Any],
        llm_config: LLMConfig,
        learn_trait: LearnTrait | None = None,
        variables: dict[str, str] | None = None,
        compactor: Compactor | None = None,
    ) -> PromptOnlyAgent:
        """Create agent from configuration dictionary."""
        variables = variables or {}
        config_dict = _substitute_in_dict(config_dict, variables)
        config = PromptOnlyAgentConfig(**config_dict)

        agent = _create_agent_with_traits(lg, config, llm_config, learn_trait)
        conversation = _create_conversation(config)

        return cls(
            lg=lg,
            config=config,
            agent=agent,
            conversation=conversation,
            compactor=compactor or SlidingWindowCompactor(),
        )

    @property
    def name(self) -> str:
        """Agent name."""
        return self.config.name

    @property
    def cycle_count(self) -> int:
        """Number of run_once() cycles completed."""
        return self._cycle_count

    def reset_conversation(self) -> None:
        """Reset conversation to start fresh.

        Use this when you want the agent to start a new exploration
        without any prior context.
        """
        self.conversation.clear()
        self._cycle_count = 0
        self.lg.info("conversation reset", extra={"agent": self.name})

    def start(self) -> None:
        """Start the agent and all traits."""
        self.agent.start()
        self.lg.info("prompt agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop the agent and all traits."""
        self.agent.stop()
        self.lg.info("prompt agent stopped", extra={"agent": self.name})

    def run_once(self) -> TaskResult:
        """Execute the agent's task once, maintaining conversation context."""
        self._cycle_count += 1
        is_first_run = self._cycle_count == 1

        self._log_run_start(is_first_run)
        self._prepare_conversation(is_first_run)

        result = self._run_conversation_loop()

        self._log_run_complete(result)
        self._store_result(result)

        return result

    def _log_run_start(self, is_first_run: bool) -> None:
        """Log task start with cycle info."""
        self.lg.debug(
            "running task",
            extra={
                "agent": self.name,
                "cycle": self._cycle_count,
                "is_first": is_first_run,
                "conv_tokens": self.conversation.token_count,
            },
        )

    def _log_run_complete(self, result: TaskResult) -> None:
        """Log task completion with result info."""
        self.lg.info(
            "task completed",
            extra={
                "agent": self.name,
                "cycle": self._cycle_count,
                "success": result.success,
                "iterations": result.iterations,
                "tool_calls": len(result.tool_calls),
                "conv_tokens": self.conversation.token_count,
            },
        )

    def _prepare_conversation(self, is_first_run: bool) -> None:
        """Prepare conversation for this run (compact if needed, init or continue)."""
        if self.conversation.needs_compaction():
            self._compact_conversation()

        if is_first_run:
            self._initialize_conversation()
        else:
            self._continue_conversation()

    def _store_result(self, result: TaskResult) -> None:
        """Store result in bounded history."""
        self._results.append(result)
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results :]

    def _initialize_conversation(self) -> None:
        """Initialize conversation with system prompt and task for first run."""
        # Build system prompt using agent's directive
        from llm_agent.core.traits.directive import DirectiveTrait

        directive_trait = self.agent.get_trait(DirectiveTrait)
        base_prompt = ""
        if directive_trait is not None:
            base_prompt = directive_trait.build_prompt("")

        self.conversation.add_system(base_prompt)
        self.conversation.add_user(self.config.task.description)

    def _continue_conversation(self) -> None:
        """Add continuation prompt for subsequent runs."""
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
                "agent": self.name,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "reduction": before_tokens - after_tokens,
            },
        )

    def _run_conversation_loop(self) -> TaskResult:
        """Run tool executor with current conversation state."""
        from llm_agent.core.traits.llm import LLMTrait
        from llm_agent.core.traits.tools import ToolsTrait

        llm_trait = self.agent.require_trait(LLMTrait)
        tools_trait = self.agent.get_trait(ToolsTrait)

        if tools_trait is None or not tools_trait.has_tools():
            return self._run_simple_completion(llm_trait)

        return self._run_with_tools(llm_trait, tools_trait)

    def _run_simple_completion(self, llm_trait: LLMTrait) -> TaskResult:
        """Run simple completion without tools."""
        from llm_agent.core.task import TaskResult

        result = llm_trait.complete(messages=self.conversation.messages())
        self.conversation.add_assistant(result.content)
        return TaskResult(
            success=True,
            content=result.content,
            tool_calls=[],
            iterations=1,
            tokens_used=result.tokens_used,
        )

    def _run_with_tools(self, llm_trait: LLMTrait, tools_trait: ToolsTrait) -> TaskResult:
        """Run tool executor loop."""
        from llm_agent.core.agent import _LLMTraitBackend
        from llm_agent.core.task import TaskResult
        from llm_agent.core.tools.executor import ToolExecutor

        task = self._build_task()
        executor = ToolExecutor(
            lg=self.lg,
            llm=_LLMTraitBackend(llm_trait),
            registry=tools_trait.registry,
            model=self.agent.config.model,
        )

        try:
            messages = self.conversation.messages()
            exec_result = executor.run(messages=messages, max_iterations=task.max_iterations)
            self._update_conversation_from_execution(messages)
            return self._build_task_result(task, exec_result, llm_trait)
        except RuntimeError as e:
            self.lg.warning("tool loop failed", extra={"agent": self.name, "exception": e})
            return TaskResult(success=False, content="", error=str(e))

    def _update_conversation_from_execution(self, messages: list[Message]) -> None:
        """Update conversation with messages added during tool execution.

        The executor modifies the messages list in-place, adding assistant
        messages with tool calls and tool result messages. We need to sync
        these back to our conversation.
        """
        # Find where our original messages end
        original_count = len(self.conversation.messages())

        # Add any new messages from execution
        for msg in messages[original_count:]:
            self.conversation.add(msg)

    def _build_task_result(self, task: Task, exec_result: Any, llm_trait: LLMTrait) -> TaskResult:
        """Build TaskResult from tool execution, with optional structured extraction."""
        from llm_agent.core.task import TaskResult

        parsed, final_content, extra_tokens = None, exec_result.content, 0

        if task.output_schema is not None:
            extraction = self._try_extract_structured(exec_result, task.output_schema, llm_trait)
            if isinstance(extraction, TaskResult):
                return extraction  # Error result
            parsed, final_content, extra_tokens = extraction

        return TaskResult(
            success=True,
            content=final_content,
            parsed=parsed,
            completion=self._extract_completion(exec_result.terminal_data),
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
            self.lg.warning("invalid terminal status", extra={"status": status, "agent": self.name})
            return None

    def _try_extract_structured(
        self, exec_result: Any, output_schema: type, llm_trait: LLMTrait
    ) -> tuple[Any, str, int] | TaskResult:
        """Try to extract structured output, return tuple or error TaskResult."""
        from llm_agent.core.task import TaskResult

        try:
            extraction = self._extract_structured_output(
                exec_result.content, output_schema, llm_trait
            )
            return extraction["parsed"], extraction["content"], extraction["tokens"]
        except Exception as e:
            return TaskResult(
                success=False,
                content=exec_result.content,
                tool_calls=exec_result.tool_calls,
                iterations=exec_result.iterations,
                tokens_used=exec_result.total_tokens,
                error=f"Structured output error: {e}",
            )

    def _extract_structured_output(
        self,
        tool_loop_content: str,
        output_schema: type,
        llm_trait: LLMTrait,
    ) -> dict[str, Any]:
        """Extract structured output from tool loop result."""
        system_prompt = (
            "You are a data extraction assistant. "
            "Your task is to extract information and return it in the exact JSON format requested."
        )
        user_prompt = (
            "Based on the following information, extract the relevant data.\n\n"
            f"Information:\n{tool_loop_content}"
        )
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        result = llm_trait.complete(
            messages=messages,
            model=self.agent.config.model,
            output_schema=output_schema,
        )

        return {
            "parsed": result.parsed,
            "content": result.content,
            "tokens": result.tokens_used,
        }

    def _build_task(self) -> Task:
        """Build Task from config."""
        output_schema = None
        if self.config.task.output_schema:
            output_schema = _json_schema_to_pydantic(
                self.config.task.output_schema,
                f"{self.name}Output",
            )

        return Task(
            name=f"{self.name}-task",
            description=self.config.task.description,
            output_schema=output_schema,
        )

    def get_recent_results(self, limit: int = 10) -> list[TaskResult]:
        """Get recent task execution results.

        Args:
            limit: Maximum results to return.

        Returns:
            List of recent TaskResults, most recent last.
        """
        return self._results[-limit:]

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
        self.lg.debug("ask received", extra={"agent": self.name, "question_len": len(question)})

        # Build context from recent results
        context = self._build_context_for_ask()

        system_prompt = f"""You are {self.name}, a learning agent.

{self.config.directive.prompt}

{context}

Answer the user's question based on your purpose and what you've learned."""

        result = self.agent.complete(question, system_prompt=system_prompt)

        self.lg.debug(
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

    def record_feedback(self, signal: str, context: dict[str, Any] | None = None) -> None:
        """Record feedback on the agent's behavior.

        Args:
            signal: Feedback signal (e.g., "helpful", "wrong", "good insight").
            context: Additional context about the feedback.
        """
        from llm_agent.core.traits.learn import LearnTrait

        learn_trait = self.agent.get_trait(LearnTrait)
        if learn_trait is None:
            self.lg.warning("feedback ignored - LearnTrait not configured")
            return

        learn_trait.record_feedback(
            content=signal,
            signal="positive"
            if "good" in signal.lower() or "helpful" in signal.lower()
            else "negative",
            context=context or {},
        )


# Tool type aliases map to canonical names
_TOOL_TYPE_ALIASES: dict[str, str] = {
    "file_read": "read_file",
    "file_write": "write_file",
    "fetch": "http_fetch",
}


def _create_tool(
    tool_type: str, tool_config: dict[str, Any], learn_trait: LearnTrait | None
) -> Any:
    """Create a tool instance from type and config."""
    from llm_agent.core.tools.builtin import (
        CompleteTaskTool,
        FileReadTool,
        FileWriteTool,
        HTTPFetchTool,
        RecallTool,
        RememberTool,
        ShellTool,
    )

    canonical_type = _TOOL_TYPE_ALIASES.get(tool_type, tool_type)

    if canonical_type == "shell":
        return ShellTool(**tool_config)
    elif canonical_type == "read_file":
        return FileReadTool(**tool_config)
    elif canonical_type == "write_file":
        return FileWriteTool(**tool_config)
    elif canonical_type == "http_fetch":
        return HTTPFetchTool(**tool_config)
    elif canonical_type == "complete_task":
        return CompleteTaskTool()
    elif canonical_type in ("remember", "recall"):
        if learn_trait is None:
            raise ValueError(f"{canonical_type} tool requires LearnTrait")
        return (
            RememberTool(learn_trait) if canonical_type == "remember" else RecallTool(learn_trait)
        )
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")


def _setup_tools(
    registry: ToolRegistry,
    tools_config: dict[str, dict[str, Any]],
    learn_trait: LearnTrait | None,
) -> None:
    """Set up tools from configuration."""
    for tool_type, tool_config in tools_config.items():
        tool = _create_tool(tool_type, tool_config, learn_trait)
        registry.register(tool)


def _json_schema_to_pydantic(schema: dict[str, Any], name: str) -> type[BaseModel]:
    """Convert JSON Schema dict to a Pydantic model class.

    This is a simplified conversion - supports basic types.

    Args:
        schema: JSON Schema dictionary.
        name: Name for the generated class.

    Returns:
        Dynamically created Pydantic model class.
    """
    from pydantic import create_model

    field_definitions: dict[str, Any] = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        python_type = type_mapping.get(json_type, str)

        if field_name in required:
            field_definitions[field_name] = (python_type, ...)
        else:
            field_definitions[field_name] = (python_type | None, None)

    return create_model(name, **field_definitions)
