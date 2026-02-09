"""Tools trait for agent tool capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...tools.base import Tool
from ...tools.registry import Registry


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent


@dataclass
class ToolsTrait:
    """Tool capability trait.

    Manages a registry of tools available to the agent. Tools can be
    registered before or after attaching to an agent.

    Example:
        from llm_agent.core.traits import ToolsTrait
        from llm_agent.core.tools import ShellTool, FileReadTool

        tools_trait = ToolsTrait()
        tools_trait.register(ShellTool())
        tools_trait.register(FileReadTool())

        agent = Agent(lg, config)
        agent.add_trait(tools_trait)
        agent.start()

        # Agent can now use tools during execute()
    """

    _agent: Agent | None = field(default=None, repr=False, compare=False)
    _registry: Registry = field(default_factory=Registry)

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent.

        Args:
            agent: The agent this trait is attached to.
        """
        self._agent = agent

    def on_start(self) -> None:
        """Called when agent starts. No-op for ToolsTrait."""
        pass

    def on_stop(self) -> None:
        """Called when agent stops. No-op for ToolsTrait."""
        pass

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        self._registry.register(tool)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of the tool to unregister.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        self._registry.unregister(name)

    @property
    def registry(self) -> Registry:
        """Access the tool registry."""
        return self._registry

    def has_tools(self) -> bool:
        """Check if any tools are registered.

        Returns:
            True if at least one tool is registered.
        """
        return len(self._registry) > 0
