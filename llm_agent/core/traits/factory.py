"""Trait factory for creating Trait instances from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent, Identity
    from llm_agent.core.platform import PlatformContext
    from llm_agent.core.traits import TraitName
    from llm_agent.core.traits.builtin.directive import DirectiveTrait, MethodTrait
    from llm_agent.core.traits.builtin.learn import LearnTrait
    from llm_agent.core.traits.builtin.llm import LLMConfig, LLMTrait

from .base import Trait


class Factory:
    """Factory for creating Trait instances from configuration.

    Has reference to platform for accessing configs.

    Example:
        trait_factory = Factory(lg, platform)

        # Generic creation
        trait = trait_factory.create("llm", agent_config={}, identity=agent.identity)

        # Or direct
        llm_trait = trait_factory.create_llm_trait(platform.llm_config())
    """

    def __init__(self, platform: PlatformContext) -> None:
        """Initialize factory.

        Args:
            platform: Platform context for accessing configs and logger.
        """
        from . import TraitName

        self._platform = platform
        self._lg = platform.logger

        # Map trait types to creator functions
        self._creators: dict[TraitName, Callable[..., Trait]] = {
            TraitName.DIRECTIVE: self._create_directive,
            TraitName.LLM: self._create_llm,
            TraitName.LEARN: self._create_learn,
            TraitName.METHOD: self._create_method,
        }

    def create(
        self,
        trait_name: TraitName,  # Use enum, not string
        agent: Agent,
        agent_config: dict[str, Any],
        **kwargs: Any,
    ) -> Trait:
        """Create a trait by name - uses mapping to route to specific creators.

        Args:
            trait_name: Trait type (TraitName enum).
            agent: Agent instance that will own this trait.
            agent_config: Agent's config dict (for directive, method fields).
            **kwargs: Additional args (e.g., identity for LearnTrait).

        Returns:
            Created trait instance.

        Raises:
            ConfigError: If required configuration missing or trait type unknown.
        """
        from ..errors import ConfigError

        creator = self._creators.get(trait_name)
        if not creator:
            raise ConfigError(f"Unknown trait type: {trait_name}")

        return creator(agent, agent_config, **kwargs)

    def _create_directive(
        self, agent: Agent, agent_config: dict[str, Any], **kwargs: Any
    ) -> DirectiveTrait:
        """Route to create_directive_trait."""
        return self.create_directive_trait(agent, agent_config.get("directive"))

    def _create_llm(self, agent: Agent, agent_config: dict[str, Any], **kwargs: Any) -> LLMTrait:
        """Route to create_llm_trait."""
        return self.create_llm_trait(agent, self._platform.llm_config())

    def _create_learn(
        self, agent: Agent, agent_config: dict[str, Any], **kwargs: Any
    ) -> LearnTrait:
        """Route to create_learn_trait."""
        return self.create_learn_trait(agent, kwargs.get("identity"), self._platform.learn_config())

    def _create_method(
        self, agent: Agent, agent_config: dict[str, Any], **kwargs: Any
    ) -> MethodTrait:
        """Route to create_method_trait."""
        return self.create_method_trait(agent, agent_config.get("method"))

    def create_llm_trait(self, agent: Agent, llm_config: LLMConfig | None) -> LLMTrait:
        """Create LLMTrait with LLM backend configuration.

        Args:
            agent: Agent instance that will own this trait.
            llm_config: LLM backend configuration dict.

        Returns:
            Configured LLMTrait instance.

        Raises:
            ConfigError: If llm_config is None or invalid.
        """
        from llm_agent.core.traits.builtin.llm import LLMTrait

        from ..errors import ConfigError

        if not llm_config:
            raise ConfigError("LLM configuration required but not provided")

        return LLMTrait(agent, llm_config)

    def create_directive_trait(
        self, agent: Agent, config: str | dict[str, Any] | None
    ) -> DirectiveTrait:
        """Create DirectiveTrait from string or dict config.

        Args:
            agent: Agent instance that will own this trait.
            config: Directive prompt string or dict with 'prompt' key.

        Returns:
            Configured DirectiveTrait instance.

        Raises:
            ConfigError: If config is None or invalid.
        """
        from llm_agent.core.traits.builtin.directive import Directive, DirectiveTrait

        from ..errors import ConfigError

        if not config:
            raise ConfigError("Directive configuration required but not provided")

        if isinstance(config, str):
            directive = Directive(prompt=config)
        elif isinstance(config, dict):
            directive = Directive(**config)
        elif isinstance(config, Directive):
            directive = config
        else:
            raise ConfigError(
                f"Directive config must be str, dict, or Directive, got {type(config).__name__}"
            )

        return DirectiveTrait(agent, directive)

    def create_learn_trait(
        self, agent: Agent, identity: Identity | None, learn_config: dict[str, Any] | None
    ) -> LearnTrait:
        """Create LearnTrait with agent-specific identity.

        Args:
            agent: Agent instance that will own this trait.
            identity: Agent's identity for memory addressing.
            learn_config: Learning configuration dict (db, embedder_url, etc.).

        Returns:
            Configured LearnTrait instance.

        Raises:
            ConfigError: If learn_config is None or missing required fields.
        """
        from llm_agent.core.traits.builtin.learn import LearnConfig, LearnTrait

        from ..errors import ConfigError

        if not learn_config:
            raise ConfigError("Learning configuration required but not provided")

        if "db" not in learn_config:
            raise ConfigError("Learning configuration missing required 'db' field")

        config = LearnConfig(
            identity=identity,
            llm=learn_config.get("llm", {}),
            db=learn_config["db"],
            embedder_url=learn_config.get("embedder_url"),
            embedder_model=learn_config.get("embedder_model", "default"),
            embedder_timeout=learn_config.get("embedder_timeout", 30.0),
        )
        return LearnTrait(agent, config)

    def create_method_trait(self, agent: Agent, method: str | None) -> MethodTrait:
        """Create MethodTrait from method string.

        Args:
            agent: Agent instance that will own this trait.
            method: Method/approach description.

        Returns:
            Configured MethodTrait.

        Raises:
            ConfigError: If method is None or empty.
        """
        from llm_agent.core.traits.builtin.directive import MethodTrait

        from ..errors import ConfigError

        if not method:
            raise ConfigError("Method configuration required but not provided")

        return MethodTrait(agent, method)
