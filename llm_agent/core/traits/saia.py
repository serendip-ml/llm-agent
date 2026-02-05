"""SAIA trait for verb-based LLM operations.

Provides optional SAIA integration for agents that want structured
verb vocabulary (complete, verify, confirm, etc.) rather than raw LLM calls.

This trait is optional - agents can use LLMTrait directly for raw access,
or add SAIATrait for structured operations, or use both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from llm_saia import SAIA, RunConfig, SAIABackend, ToolDef

from llm_agent.core.traits.base import Trait


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent
    from llm_agent.core.tools.registry import ToolRegistry


@dataclass
class SAIAConfig:
    """Configuration for SAIATrait."""

    terminal_tool: str = "complete_task"
    """Name of the tool that signals task completion."""

    max_iterations: int = 0
    """Max tool-calling iterations (0 = unlimited)."""

    timeout_secs: float = 0
    """Timeout in seconds (0 = no timeout)."""

    system_prompt: str | None = None
    """Default system prompt for SAIA operations."""


@dataclass
class SAIATrait(Trait):
    """Provides SAIA verb vocabulary to agents.

    SAIA offers structured LLM operations through semantic verbs:
    - complete(): Task execution with tool loop
    - verify(): Check if artifact satisfies predicate
    - confirm(): Yes/no confirmation
    - classify(): Categorize into options
    - critique(): Find counter-arguments
    - extract(): Pull structured data from content
    - And more...

    Example:
        from llm_saia.backends.anthropic import AnthropicBackend

        # Create agent with SAIA
        agent = MyAgent(lg, config)  # Your Agent subclass
        agent.add_trait(SAIATrait(
            lg=lg,
            backend=AnthropicBackend(),
            config=SAIAConfig(terminal_tool="complete_task"),
        ))
        agent.start()

        # Use SAIA verbs
        saia_trait = agent.require_trait(SAIATrait)
        result = await saia_trait.saia.complete("Analyze this code...")
        verified = await agent.saia.verify(output, "is valid JSON")
    """

    _lg: Logger
    backend: SAIABackend
    config: SAIAConfig = field(default_factory=SAIAConfig)

    _agent: Agent | None = field(default=None, repr=False, compare=False)
    _saia: SAIA | None = field(default=None, repr=False, compare=False)

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent."""
        self._agent = agent

    def on_start(self) -> None:
        """Build SAIA instance on agent start."""
        if self._agent is None:
            raise RuntimeError("SAIATrait not attached to agent")

        tools, executor = self._get_tools_and_executor()

        self._lg.debug(
            "SAIA tools configured",
            extra={
                "tool_count": len(tools),
                "tool_names": [t.name for t in tools],
                "has_executor": executor is not None,
            },
        )

        run_config = RunConfig(
            max_iterations=self.config.max_iterations,
            timeout_secs=self.config.timeout_secs,
        )

        self._saia = SAIA(
            backend=self.backend,
            tools=tools,
            executor=executor,
            system=self.config.system_prompt,
            run=run_config,
            terminal_tool=self.config.terminal_tool,
            lg=self._lg,
        )
        self._lg.debug("SAIA trait started")

    def on_stop(self) -> None:
        """Clean up on agent stop."""
        # Backend cleanup handled by caller (they own the backend)
        self._saia = None
        self._lg.debug("SAIA trait stopped")

    @property
    def saia(self) -> SAIA:
        """Access the SAIA instance.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._saia is None:
            raise RuntimeError("SAIATrait not started - ensure agent.start() was called")
        return self._saia

    def _get_tools_and_executor(self) -> tuple[list[ToolDef], Any]:
        """Get tools and executor from ToolsTrait if available."""
        from llm_agent.core.traits.tools import ToolsTrait

        assert self._agent is not None

        tools_trait = self._agent.get_trait(ToolsTrait)
        if tools_trait is None or not tools_trait.has_tools():
            return [], None

        tools = [_tool_to_tooldef(t) for t in tools_trait.registry.list_tools()]
        executor = _create_executor(tools_trait.registry, self._lg)

        return tools, executor


def _tool_to_tooldef(tool: Any) -> ToolDef:
    """Convert llm-agent Tool to SAIA ToolDef."""
    return ToolDef(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def _create_executor(registry: ToolRegistry, lg: Logger) -> Any:
    """Create async tool executor for SAIA.

    Runs sync tool.execute() in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    import inspect

    async def executor(name: str, arguments: dict[str, Any]) -> str:
        tool = registry.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"

        try:
            # Run sync tool.execute in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(tool.execute, **arguments)

            # Handle case where tool.execute might be async (future-proofing)
            if inspect.isawaitable(result):
                result = await result

            if result.success:
                return result.output
            return f"Error: {result.error or 'Tool execution failed'}"
        except Exception as e:
            lg.warning("tool execution failed", extra={"tool": name, "exception": e})
            return f"Error executing {name}: {e}"

    return executor
