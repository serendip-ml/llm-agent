"""Agent factory base class and configuration utilities."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, ClassVar

from appinfra.log import Logger

from ..errors import ConfigError, TraitNotFoundError
from ..traits import TraitName
from ..traits.directive import Directive
from .agent import Agent
from .identity import Identity


if TYPE_CHECKING:
    from ..platform import PlatformContext


class Factory:
    """Base factory for agents with standard initialization.

    This factory handles common patterns:
    - Parses profile + identity → constructs Identity
    - Extracts config params → passes to agent __init__
    - Creates and attaches traits based on requirements
    - Configures tools from YAML or factory defaults

    Trait Requirements (3 ways, priority order):
        1. YAML config (highest priority):
            traits:
              required: [llm, learn]

        2. Factory class variable:
            class CustomFactory(Factory):
                agent_class = MyAgent
                required_traits = [TraitName.LLM, TraitName.LEARN]

        3. No requirements (agent handles validation in code)

    Tool Configuration (2 ways, priority order):
        1. YAML config (highest priority):
            tools:
              remember: {}
              recall: {}

        2. Factory class variable:
            class CustomFactory(Factory):
                agent_class = MyAgent
                default_tools = {"remember": {}, "recall": {}}

    Usage:
        class CustomFactory(Factory):
            agent_class = MyAgent  # Just specify your agent class
    """

    # Subclasses must set this
    agent_class: ClassVar[type[Agent]] = None  # type: ignore[assignment]

    # Optional: declare required traits at factory level
    required_traits: ClassVar[list[TraitName]] = []

    # Optional: declare default tools at factory level
    default_tools: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self, platform: PlatformContext) -> None:
        """Initialize factory with platform context.

        Args:
            platform: Platform context with all resources.
        """
        self._platform = platform
        self._lg = platform.logger

    def create(
        self,
        config: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> Agent:
        """Create agent instance with standard initialization.

        Args:
            config: Full config dict from manifest.
            variables: Optional environment variable substitutions.

        Returns:
            Configured agent instance.

        Raises:
            ConfigError: If agent_class not set or required config fields missing.
        """
        if self.agent_class is None:
            raise ConfigError(f"{self.__class__.__name__} must set agent_class class variable")

        # Parse profile → construct Identity (required)
        identity = self._build_identity(config)

        # Extract agent-specific config
        agent_config = config.get("config", {})

        # Instantiate agent (only identity is required)
        agent = self.agent_class(  # type: ignore[call-arg]
            lg=self._lg,
            identity=identity,
            **agent_config,
        )

        # Create and attach traits (all handled uniformly)
        self._attach_traits(agent, config)

        return agent

    def _build_identity(self, config: dict[str, Any]) -> Identity:
        """Build Identity from profile fields.

        Args:
            config: Full config dict from manifest.

        Returns:
            Constructed Identity for addressing.

        Raises:
            ConfigError: If required fields missing.
        """
        # Extract profile
        profile = config.get("profile", {})
        name = profile.get("name")

        if not name:
            raise ConfigError("profile.name is required")

        # Use Identity.from_config to properly resolve IDs
        return Identity.from_config(profile, defaults={"name": name})

    def _attach_traits(self, agent: Agent, config: dict[str, Any]) -> None:
        """Create and attach traits for the agent.

        Only creates traits that are explicitly requested via:
        - config['traits']['required']
        - self.required_traits class variable

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.

        Raises:
            TraitNotFoundError: If required traits cannot be created.
        """
        # Determine which traits to create
        traits_to_create = self._determine_required_traits(config)

        # Create each requested trait using platform.trait_factory
        for trait_name in traits_to_create:
            trait = self._create_trait(trait_name, agent, config)
            agent.add_trait(trait)
            self._lg.debug("created trait", extra={"agent": agent.name, "trait": trait_name.value})

        # Validate all required traits were created
        self._validate_trait_requirements(agent, config, traits_to_create)

        # Configure tools from YAML or factory defaults
        self._configure_tools(agent, config)

    def _determine_required_traits(self, config: dict[str, Any]) -> list[TraitName]:
        """Determine which traits to create for this agent.

        Args:
            config: Full config dict from manifest.

        Returns:
            List of trait names to create.
        """
        traits_config = config.get("traits", {})
        if "required" in traits_config:
            # Config takes priority - convert strings to TraitName
            return [TraitName(name) for name in traits_config["required"]]
        elif self.required_traits:
            # Use factory class variable
            return self.required_traits
        else:
            # No traits specified - create none
            return []

    def _create_trait(self, trait_name: TraitName, agent: Agent, config: dict[str, Any]) -> Any:
        """Create a trait instance using platform.trait_factory.

        Delegates to trait_factory.create() which handles validation and routing.

        Args:
            trait_name: Type of trait to create.
            agent: Agent instance (needed for agent-specific config like identity).
            config: Full config dict from manifest.

        Returns:
            Created trait instance.

        Raises:
            ConfigError: If required configuration is missing.
            ValueError: If trait type is unknown.
        """
        return self._platform.trait_factory.create(
            trait_name=trait_name,  # Pass enum directly
            agent_config=config,
            identity=agent.identity,  # For LearnTrait
        )

    def _validate_trait_requirements(
        self, agent: Agent, config: dict[str, Any], required: list[TraitName]
    ) -> None:
        """Validate that required traits were successfully created.

        Args:
            agent: Agent instance with traits attached.
            config: Full config dict from manifest.
            required: List of trait names that should have been created.

        Raises:
            TraitNotFoundError: If required traits are missing.
        """
        if not required:
            return

        # Map trait names to classes for validation
        from ..traits.directive import DirectiveTrait
        from ..traits.learn import LearnTrait
        from ..traits.llm import LLMTrait

        trait_class_map = {
            TraitName.DIRECTIVE: DirectiveTrait,
            TraitName.LLM: LLMTrait,
            TraitName.LEARN: LearnTrait,
        }

        # Validate each required trait is attached
        missing: list[str] = []
        for trait_name in required:
            trait_class = trait_class_map.get(trait_name)
            if trait_class is None or agent.get_trait(trait_class) is None:  # type: ignore[arg-type]
                missing.append(trait_name.value)

        if missing:
            raise TraitNotFoundError(
                f"{agent.name} requires traits {missing} but they were not created. "
                f"Check platform configuration (e.g., 'learn' section for LearnTrait)."
            )

    def _configure_tools(self, agent: Agent, config: dict[str, Any]) -> None:
        """Configure tools for the agent from YAML or factory defaults.

        Tool configuration priority:
        1. config['tools'] (YAML config)
        2. self.default_tools class variable
        3. No tools (empty ToolsTrait)

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.
        """
        from ..traits.tools import ToolsTrait

        # Determine which tools to configure
        tools_config = config.get("tools", self.default_tools)
        if not tools_config:
            # No tools configured - skip
            return

        # Bind LearnTrait to platform.tool_factory if available
        from ..traits.learn import LearnTrait

        learn_trait = agent.get_trait(LearnTrait)  # type: ignore[arg-type]
        if learn_trait:
            self._platform.tool_factory.set_learn_trait(learn_trait)

        # Create ToolsTrait
        tools_trait = ToolsTrait()

        # Create and register each configured tool
        for tool_name, tool_config in tools_config.items():
            tool = self._platform.tool_factory.create(tool_name, tool_config)
            if tool is None:
                self._lg.warning(
                    "tool could not be created",
                    extra={"agent": agent.name, "tool": tool_name},
                )
                continue

            tools_trait.register(tool)
            self._lg.debug(
                "created tool for agent",
                extra={"agent": agent.name, "tool": tool_name},
            )

        # Attach ToolsTrait if any tools were created
        if len(tools_trait._registry) > 0:
            agent.add_trait(tools_trait)


# =============================================================================
# Configuration Helper Functions
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
