"""Base factory for programmatic agents.

This provides default initialization logic so agents don't need to write
boilerplate Factory code. Most agents can simply subclass and set agent_class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import ConfigError, TraitNotFoundError
from ..traits import TraitName
from ..traits.identity import Directive
from .agent import Agent
from .identity import Identity


if TYPE_CHECKING:
    from ..platform import PlatformContext


class ProgAgentFactory:
    """Base factory for programmatic agents with standard initialization.

    This factory handles common patterns:
    - Parses profile + identity → constructs Identity
    - Extracts config params → passes to agent __init__
    - Attaches traits based on agent requirements
    - Configures tools from YAML or factory defaults

    Trait Requirements (3 ways, priority order):
        1. YAML config (highest priority):
            traits:
              required: [llm, learn]

        2. Factory class variable:
            class Factory(ProgAgentFactory):
                agent_class = MyAgent
                required_traits = [TraitName.LLM, TraitName.LEARN]

        3. No requirements (agent handles validation in code)

    Tool Configuration (2 ways, priority order):
        1. YAML config (highest priority):
            tools:
              remember: {}
              recall: {}

        2. Factory class variable:
            class Factory(ProgAgentFactory):
                agent_class = MyAgent
                default_tools = {"remember": {}, "recall": {}}

    Usage:
        class Factory(ProgAgentFactory):
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

        # Parse profile → construct Identity
        identity = self._build_identity(config)

        # Parse identity prompt → construct Directive
        directive = self._build_directive(config)

        # Extract agent-specific config
        agent_config = config.get("config", {})

        # Instantiate agent
        agent = self.agent_class(  # type: ignore[call-arg]
            lg=self._lg,
            identity=identity,
            directive=directive,
            **agent_config,
        )

        # Attach standard traits
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

    def _build_directive(self, config: dict[str, Any]) -> Directive:
        """Build Directive from identity field.

        Args:
            config: Full config dict from manifest.

        Returns:
            Constructed Directive.

        Raises:
            ConfigError: If identity field missing.
        """
        identity_prompt = config.get("identity")
        if not identity_prompt:
            raise ConfigError("identity is required")

        if isinstance(identity_prompt, str):
            return Directive(prompt=identity_prompt)
        else:
            return Directive(**identity_prompt)

    def _attach_traits(self, agent: Agent, config: dict[str, Any]) -> None:
        """Attach all available platform traits and validate requirements.

        Trait requirements are determined in priority order:
        1. config['traits']['required'] (if present)
        2. self.required_traits class variable (if set)
        3. No requirements (validation skipped)

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.

        Raises:
            TraitNotFoundError: If required traits are not available in platform.
        """
        # Attach all available traits from platform registry
        for trait in self._platform.traits.all():
            agent.add_trait(trait)

        trait_names = [type(t).__name__ for t in self._platform.traits.all()]
        self._lg.debug(
            "attached traits to agent",
            extra={"agent": agent.name, "traits": trait_names},
        )

        # Validate trait requirements
        self._validate_trait_requirements(agent, config)

        # Configure tools from YAML or factory defaults
        self._configure_tools(agent, config)

    def _validate_trait_requirements(self, agent: Agent, config: dict[str, Any]) -> None:
        """Validate that required traits are attached.

        Args:
            agent: Agent instance with traits attached.
            config: Full config dict from manifest.

        Raises:
            TraitNotFoundError: If required traits are missing.
        """
        # Determine required traits (priority: config > class variable > none)
        required: list[TraitName] = []

        traits_config = config.get("traits", {})
        if "required" in traits_config:
            # Config takes priority - convert strings to TraitName
            required = [TraitName(name) for name in traits_config["required"]]
        elif self.required_traits:
            # Use factory class variable
            required = self.required_traits

        if not required:
            # No requirements specified - skip validation
            return

        # Validate each required trait
        missing: list[str] = []
        for trait_name in required:
            trait = self._platform.traits.get_by_name(trait_name)
            if trait is None:
                missing.append(trait_name.value)

        if missing:
            raise TraitNotFoundError(
                f"{agent.name} requires traits {missing} but they are not configured. "
                f"Check platform configuration (e.g., 'learn' section for LearnTrait)."
            )

    def _configure_tools(self, agent: Agent, config: dict[str, Any]) -> None:  # cq: max-lines=35
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

        # Create ToolsTrait
        tools_trait = ToolsTrait()

        # Register each configured tool
        for tool_name in tools_config:
            # Get tool from platform registry
            tool = self._platform.tools.get(tool_name)
            if tool is None:
                self._lg.warning(
                    "tool not available in platform",
                    extra={"agent": agent.name, "tool": tool_name},
                )
                continue

            tools_trait.register(tool)
            self._lg.debug(
                "registered tool for agent",
                extra={"agent": agent.name, "tool": tool_name},
            )

        # Attach ToolsTrait if any tools were registered
        if len(tools_trait._registry) > 0:
            agent.add_trait(tools_trait)
