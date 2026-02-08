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
        """Create and attach traits for the agent.

        Creates fresh trait instances from platform configuration.
        Each agent gets its own trait instances (no sharing).

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.

        Raises:
            TraitNotFoundError: If required traits cannot be created.
        """
        # Create and attach LLMTrait if available
        if self._platform.llm_config():
            self._add_llm_trait(agent)

        # Create and attach LearnTrait if available
        if self._platform.learn_config():
            self._add_learn_trait(agent)

        # Validate trait requirements
        self._validate_trait_requirements(agent, config)

        # Configure tools from YAML or factory defaults
        self._configure_tools(agent, config)

    def _add_llm_trait(self, agent: Agent) -> None:
        """Create and attach LLMTrait."""
        from ..traits.llm import LLMTrait

        llm_trait = LLMTrait(self._lg, self._platform.llm_config())
        agent.add_trait(llm_trait)
        self._lg.debug("created LLMTrait", extra={"agent": agent.name})

    def _add_learn_trait(self, agent: Agent) -> None:
        """Create and attach LearnTrait with agent-specific identity."""
        from ..traits.learn import LearnConfig, LearnTrait

        learn_config_dict = self._platform.learn_config()
        if not learn_config_dict:
            return

        learn_config = LearnConfig(
            identity=agent.identity,
            llm=learn_config_dict.get("llm", {}),
            db=learn_config_dict["db"],
            embedder_url=learn_config_dict.get("embedder_url"),
            embedder_model=learn_config_dict.get("embedder_model", "default"),
            embedder_timeout=learn_config_dict.get("embedder_timeout", 30.0),
        )

        learn_trait = LearnTrait(self._lg, learn_config)
        agent.add_trait(learn_trait)
        self._lg.debug("created LearnTrait", extra={"agent": agent.name})

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

        # Map trait names to classes for validation
        from ..traits.learn import LearnTrait
        from ..traits.llm import LLMTrait

        trait_class_map = {
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

        # Create ToolFactory and bind LearnTrait if available
        from ..tools.factory import ToolFactory
        from ..traits.learn import LearnTrait

        tool_factory = ToolFactory()
        learn_trait = agent.get_trait(LearnTrait)  # type: ignore[arg-type]
        if learn_trait:
            tool_factory.set_learn_trait(learn_trait)

        # Create ToolsTrait
        tools_trait = ToolsTrait()

        # Create and register each configured tool
        for tool_name, tool_config in tools_config.items():
            tool = tool_factory.create(tool_name, tool_config)
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
