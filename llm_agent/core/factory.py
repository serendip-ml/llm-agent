"""Factory classes for creating agents, traits, and tools.

This module provides a factory hierarchy for creating agent components:
- ToolFactory: Creates Tool instances from configuration
- TraitFactory: Creates Trait instances from configuration
- AgentFactory: Creates ConversationalAgent instances from configuration

Example:
    from appinfra.log import Logger
    from llm_agent.core.factory import AgentFactory, ToolFactory, TraitFactory
    from llm_agent.core.traits.llm import LLMConfig

    lg = Logger.create("agent")
    llm_config = LLMConfig(base_url="http://localhost:8000/v1")

    tool_factory = ToolFactory()
    trait_factory = TraitFactory(tool_factory, llm_config)
    agent_factory = AgentFactory(lg, trait_factory)

    config = {
        "name": "explorer",
        "identity": "You are a codebase exploration agent.",
        "task": {"description": "Explore the codebase"},
        "tools": {"shell": {"allowed_commands": ["grep", "find"]}},
    }

    agent = agent_factory.create(config)
    agent.start()
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llm_agent.core.config import AgentConfig
from llm_agent.core.conversation import (
    Compactor,
    Conversation,
    ConversationConfig,
    SlidingWindowCompactor,
)
from llm_agent.core.conversational import ConversationalAgent


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.tools.base import Tool
    from llm_agent.core.traits.identity import IdentityTrait, MethodTrait
    from llm_agent.core.traits.learn import LearnTrait
    from llm_agent.core.traits.llm import LLMConfig, LLMTrait
    from llm_agent.core.traits.tools import ToolsTrait


class ToolFactory:
    """Factory for creating Tool instances from configuration.

    Supports built-in tool types and custom tool registration.

    Example:
        factory = ToolFactory()

        # Create built-in tools
        shell = factory.create(ToolFactory.SHELL, {"allowed_commands": ["ls", "grep"]})
        reader = factory.create(ToolFactory.READ_FILE, {"allowed_paths": ["/home"]})

        # Register custom tool type
        factory.register("my_tool", lambda config: MyTool(**config))
        custom = factory.create("my_tool", {"option": "value"})
    """

    # Canonical tool type constants
    SHELL = "shell"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    HTTP_FETCH = "http_fetch"
    COMPLETE_TASK = "complete_task"
    REMEMBER = "remember"
    RECALL = "recall"

    # Built-in tool type aliases
    _ALIASES: dict[str, str] = {
        "file_read": READ_FILE,
        "file_write": WRITE_FILE,
        "fetch": HTTP_FETCH,
    }

    def __init__(self) -> None:
        """Initialize factory with built-in tool creators."""
        self._creators: dict[str, Callable[[dict[str, Any]], Tool]] = {}
        self._learn_trait: LearnTrait | None = None
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in tool creators."""
        from llm_agent.core.tools.builtin import (
            CompleteTaskTool,
            FileReadTool,
            FileWriteTool,
            HTTPFetchTool,
            ShellTool,
        )

        self._creators[self.SHELL] = lambda c: ShellTool(**c)
        self._creators[self.READ_FILE] = lambda c: FileReadTool(**c)
        self._creators[self.WRITE_FILE] = lambda c: FileWriteTool(**c)
        self._creators[self.HTTP_FETCH] = lambda c: HTTPFetchTool(**c)
        self._creators[self.COMPLETE_TASK] = lambda _: CompleteTaskTool()

    def set_learn_trait(self, learn_trait: LearnTrait | None) -> None:
        """Set LearnTrait for memory tools (remember/recall).

        Args:
            learn_trait: LearnTrait instance or None.
        """
        self._learn_trait = learn_trait

    def register(self, tool_type: str, creator: Callable[[dict[str, Any]], Tool]) -> None:
        """Register a custom tool type.

        Args:
            tool_type: Tool type identifier.
            creator: Callable that takes config dict and returns Tool.

        Example:
            factory.register("my_tool", lambda c: MyTool(**c))
        """
        self._creators[tool_type] = creator

    def create(self, tool_type: str, config: dict[str, Any] | None = None) -> Tool:
        """Create a tool from type and configuration.

        Args:
            tool_type: Tool type (e.g., "shell", "read_file").
            config: Tool-specific configuration.

        Returns:
            Configured Tool instance.

        Raises:
            ValueError: If tool type is unknown or requirements not met.
        """
        config = config or {}
        canonical_type = self._ALIASES.get(tool_type, tool_type)

        # Handle memory tools specially (need LearnTrait)
        if canonical_type in (self.REMEMBER, self.RECALL):
            return self._create_memory_tool(canonical_type)

        creator = self._creators.get(canonical_type)
        if creator is None:
            raise ValueError(f"Unknown tool type: {tool_type}")

        return creator(config)

    def _create_memory_tool(self, tool_type: str) -> Tool:
        """Create remember or recall tool."""
        from llm_agent.core.tools.builtin import RecallTool, RememberTool

        if self._learn_trait is None:
            raise ValueError(f"{tool_type} tool requires LearnTrait - call set_learn_trait() first")

        if tool_type == self.REMEMBER:
            return RememberTool(self._learn_trait)
        return RecallTool(self._learn_trait)


class TraitFactory:
    """Factory for creating Trait instances from configuration.

    Example:
        tool_factory = ToolFactory()
        trait_factory = TraitFactory(tool_factory, llm_config)

        # Create individual traits
        llm_trait = trait_factory.create_llm_trait()
        identity_trait = trait_factory.create_identity_trait("You are helpful.")
        tools_trait = trait_factory.create_tools_trait({"shell": {}})
    """

    def __init__(
        self,
        tool_factory: ToolFactory,
        llm_config: LLMConfig,
        learn_trait: LearnTrait | None = None,
    ) -> None:
        """Initialize factory.

        Args:
            tool_factory: Factory for creating tools.
            llm_config: LLM backend configuration.
            learn_trait: Optional LearnTrait for memory capabilities.
        """
        self._tool_factory = tool_factory
        self._llm_config = llm_config
        self._learn_trait = learn_trait

        # Configure tool factory with learn trait
        self._tool_factory.set_learn_trait(learn_trait)

    @property
    def learn_trait(self) -> LearnTrait | None:
        """LearnTrait instance if configured."""
        return self._learn_trait

    def create_llm_trait(self) -> LLMTrait:
        """Create LLMTrait with configured LLM backend."""
        from llm_agent.core.traits.llm import LLMTrait

        return LLMTrait(self._llm_config)

    def create_identity_trait(self, config: str | dict[str, Any]) -> IdentityTrait:
        """Create IdentityTrait from string or dict config.

        Args:
            config: Identity prompt string or dict with 'prompt' key.

        Returns:
            Configured IdentityTrait.
        """
        from llm_agent.core.traits.identity import Identity, IdentityTrait

        if isinstance(config, str):
            identity = Identity(prompt=config)
        elif isinstance(config, dict):
            identity = Identity(**config)
        else:
            identity = config  # Already an Identity

        return IdentityTrait(identity)

    def create_method_trait(self, method: str) -> MethodTrait:
        """Create MethodTrait from method string.

        Args:
            method: Method/approach description.

        Returns:
            Configured MethodTrait.
        """
        from llm_agent.core.traits.identity import MethodTrait

        return MethodTrait(method)

    def create_tools_trait(self, tools_config: dict[str, dict[str, Any]]) -> ToolsTrait:
        """Create ToolsTrait with tools from configuration.

        Args:
            tools_config: Tool configurations keyed by tool type.

        Returns:
            ToolsTrait with all configured tools registered.
        """
        from llm_agent.core.traits.tools import ToolsTrait

        tools_trait = ToolsTrait()

        for tool_type, tool_config in tools_config.items():
            tool = self._tool_factory.create(tool_type, tool_config)
            tools_trait.register(tool)

        return tools_trait


class AgentFactory:
    """Factory for creating ConversationalAgent instances from configuration.

    Example:
        tool_factory = ToolFactory()
        trait_factory = TraitFactory(tool_factory, llm_config)
        agent_factory = AgentFactory(lg, trait_factory)

        config = {
            "name": "code-reviewer",
            "identity": "You are a senior code reviewer.",
            "method": "- Read the full diff first",
            "task": {"description": "Review pull requests"},
            "tools": {"shell": {"allowed_commands": ["git"]}},
        }

        agent = agent_factory.create(config)
        agent.start()
    """

    def __init__(
        self,
        lg: Logger,
        trait_factory: TraitFactory,
        compactor: Compactor | None = None,
    ) -> None:
        """Initialize factory.

        Args:
            lg: Logger instance for created agents.
            trait_factory: Factory for creating traits.
            compactor: Optional conversation compaction strategy.
        """
        self._lg = lg
        self._trait_factory = trait_factory
        self._compactor = compactor

    def create(
        self,
        config: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> ConversationalAgent:
        """Create a ConversationalAgent from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - name (required): Agent identifier
                - identity: Agent's identity/persona (string or dict)
                - method: How the agent operates (string)
                - directive: Legacy field for identity (dict)
                - task: Task configuration with 'description'
                - tools: Tool configurations keyed by type
                - conversation: Conversation config (max_tokens, etc.)
            variables: Variable substitutions for {{VAR}} patterns.

        Returns:
            Configured ConversationalAgent ready to start.
        """
        variables = variables or {}
        config = _substitute_in_dict(config, variables)

        agent = self._create_base_agent(config)
        self._add_traits(agent, config)

        return agent

    def _create_base_agent(self, config: dict[str, Any]) -> ConversationalAgent:
        """Create the base agent with conversation configured."""
        task_config = config.get("task", {})
        conv_config = config.get("conversation", {})

        conversation_config = ConversationConfig(
            max_tokens=conv_config.get("max_tokens", 32000),
            compact_threshold=conv_config.get("compact_threshold", 0.8),
            min_recent_messages=conv_config.get("min_recent_messages", 4),
        )

        agent_config = AgentConfig(
            name=config["name"],
            default_prompt=task_config.get("description", ""),
        )

        return ConversationalAgent(
            lg=self._lg,
            config=agent_config,
            conversation=Conversation(config=conversation_config),
            compactor=self._compactor or SlidingWindowCompactor(),
        )

    def _add_traits(self, agent: ConversationalAgent, config: dict[str, Any]) -> None:
        """Add all traits to the agent based on configuration."""
        # Always add LLM trait
        agent.add_trait(self._trait_factory.create_llm_trait())

        # Add identity trait (from identity or legacy directive field)
        identity_config = config.get("identity") or config.get("directive")
        if identity_config is not None:
            agent.add_trait(self._trait_factory.create_identity_trait(identity_config))

        # Add method trait if specified
        method = config.get("method")
        if method is not None:
            agent.add_trait(self._trait_factory.create_method_trait(method))

        # Add learn trait if configured
        if self._trait_factory.learn_trait is not None:
            agent.add_trait(self._trait_factory.learn_trait)

        # Add tools trait
        tools_config = config.get("tools", {})
        agent.add_trait(self._trait_factory.create_tools_trait(tools_config))


# =============================================================================
# Helper Functions
# =============================================================================


def _substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Substitute {{VAR}} patterns in text with variable values.

    Falls back to environment variables if not in variables dict.
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


# =============================================================================
# Convenience Function
# =============================================================================


def create_agent_from_config(
    lg: Logger,
    config_dict: dict[str, Any],
    llm_config: LLMConfig,
    learn_trait: LearnTrait | None = None,
    variables: dict[str, str] | None = None,
    compactor: Compactor | None = None,
) -> ConversationalAgent:
    """Create a ConversationalAgent from configuration dictionary.

    Convenience function that wraps the factory hierarchy for simpler usage.

    Args:
        lg: Logger instance for the agent.
        config_dict: Configuration dictionary (see AgentFactory.create for format).
        llm_config: LLM backend configuration.
        learn_trait: Optional LearnTrait for memory capabilities.
        variables: Variable substitutions for {{VAR}} patterns.
        compactor: Optional conversation compaction strategy.

    Returns:
        Configured ConversationalAgent ready to start.

    Example:
        from llm_agent.core.factory import create_agent_from_config
        from llm_agent.core.traits.llm import LLMConfig

        config = {
            "name": "assistant",
            "identity": "You are a helpful assistant.",
            "tools": {"shell": {}},
        }

        agent = create_agent_from_config(
            lg=logger,
            config_dict=config,
            llm_config=LLMConfig(base_url="http://localhost:8000/v1"),
        )
        agent.start()
    """
    tool_factory = ToolFactory()
    trait_factory = TraitFactory(tool_factory, llm_config, learn_trait)
    agent_factory = AgentFactory(lg, trait_factory, compactor)
    return agent_factory.create(config_dict, variables)
