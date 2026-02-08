"""Platform context - central resource management for llm-agent platform.

The PlatformContext is the single source of truth for all platform resources:
- Configuration
- Shared resources (LLM clients, databases)
- Trait registry (shared platform traits)
- Tool registry (standard platform tools)

It provides a clean boundary between the platform and agents (apps).
"""

from __future__ import annotations

from typing import Any

from appinfra.log import Logger

from .tools.factory import ToolFactory
from .tools.registry import ToolRegistry
from .traits.registry import TraitRegistry


class PlatformContext:
    """Central container for all platform resources.

    This encapsulates all shared resources and provides them to agent factories.
    Think of it as the "platform runtime environment" that apps (agents) run within.

    Lifecycle:
        1. Platform creates PlatformContext during startup
        2. Initializes all shared resources (DB, LLM clients, traits)
        3. Passes context to factories when creating agents
        4. Factories access resources from context
        5. Platform calls cleanup() on shutdown

    Example:
        # Platform startup
        platform = PlatformContext.from_config(server_config, logger)

        # Creating agents
        factory = Factory(platform)  # Factory gets full platform access
        agent = factory.create(agent_config)

        # Platform shutdown
        platform.cleanup()
    """

    def __init__(
        self,
        lg: Logger,
        config: dict[str, Any],
        traits: TraitRegistry,
        tools: ToolRegistry,
    ) -> None:
        """Initialize platform context.

        Args:
            lg: Platform logger.
            config: Platform configuration dict.
            traits: Initialized trait registry with platform traits.
            tools: Initialized tool registry with standard tools.
        """
        self._lg = lg
        self._config = config
        self._traits = traits
        self._tools = tools

        self._lg.info(
            "platform context initialized",
            extra={"traits": traits.count(), "tools": len(tools)},
        )

    @classmethod
    def from_config(
        cls,
        lg: Logger,
        llm_config: dict[str, Any],
        learn_config: dict[str, Any] | None = None,
    ) -> PlatformContext:
        """Create platform context from configuration.

        Args:
            lg: Platform logger.
            llm_config: LLM backend configuration.
            learn_config: Optional learning backend configuration.

        Returns:
            Initialized PlatformContext with all resources ready.
        """
        # Store configs
        config = {
            "llm": llm_config,
            "learn": learn_config,
        }

        # Initialize trait registry
        traits = cls._build_trait_registry(lg, llm_config, learn_config)

        # Initialize tool registry with standard tools
        tools = cls._build_tool_registry(lg)

        return cls(lg=lg, config=config, traits=traits, tools=tools)

    @classmethod
    def _build_trait_registry(
        cls,
        lg: Logger,
        llm_config: dict[str, Any],
        learn_config: dict[str, Any] | None,
    ) -> TraitRegistry:
        """Build trait registry with all available platform traits.

        Args:
            lg: Platform logger.
            llm_config: LLM backend configuration.
            learn_config: Optional learning backend configuration.

        Returns:
            TraitRegistry with all configured traits.
        """
        from .traits.learn import LearnConfig, LearnTrait
        from .traits.llm import LLMTrait

        registry = TraitRegistry(lg)

        # Register LLMTrait (always available if llm_config provided)
        if llm_config:
            llm_trait = LLMTrait(lg, llm_config)
            registry.register(llm_trait)  # type: ignore[arg-type]
            lg.debug("registered LLMTrait")

        # Register LearnTrait (if learning configured)
        if learn_config:
            learn_config_obj = LearnConfig(**learn_config)
            learn_trait = LearnTrait(lg, learn_config_obj)
            registry.register(learn_trait)  # type: ignore[arg-type]
            lg.debug("registered LearnTrait")

        return registry

    @classmethod
    def _build_tool_registry(cls, lg: Logger) -> ToolRegistry:
        """Build tool registry with standard platform tools.

        Uses ToolFactory and STANDARD_TOOLS list from tools package.

        Args:
            lg: Platform logger.

        Returns:
            ToolRegistry with standard tools registered.
        """
        from .tools import STANDARD_TOOLS

        factory = ToolFactory()
        registry = ToolRegistry()

        for tool_name in STANDARD_TOOLS:
            tool = factory.create(tool_name.value, {})
            if tool is not None:
                registry.register(tool)
                lg.debug("registered standard tool", extra={"tool": tool.name})

        # Note: REMEMBER/RECALL tools require LearnTrait, so they're not in
        # STANDARD_TOOLS. They'll be added when agents with LearnTrait are created.

        lg.info("tool registry built", extra={"tools": len(registry)})
        return registry

    @property
    def logger(self) -> Logger:
        """Platform logger."""
        return self._lg

    @property
    def config(self) -> dict[str, Any]:
        """Platform configuration."""
        return self._config

    @property
    def traits(self) -> TraitRegistry:
        """Trait registry with all available platform traits."""
        return self._traits

    @property
    def tools(self) -> ToolRegistry:
        """Tool registry with standard platform tools."""
        return self._tools

    def llm_config(self) -> dict[str, Any]:
        """Get LLM configuration.

        Returns:
            LLM backend configuration dict.
        """
        return self._config.get("llm", {})  # type: ignore[no-any-return]

    def learn_config(self) -> dict[str, Any] | None:
        """Get learning configuration.

        Returns:
            Learning backend configuration dict, or None if not configured.
        """
        return self._config.get("learn")

    def start_traits(self) -> None:
        """Start all registered traits.

        Calls on_start() on all traits in the registry.
        """
        for trait in self._traits.all():
            try:
                trait.on_start()
                self._lg.debug("trait started", extra={"trait": type(trait).__name__})
            except Exception as e:
                self._lg.error(
                    "failed to start trait",
                    extra={"trait": type(trait).__name__, "exception": e},
                )
                raise

        self._lg.info("all traits started", extra={"count": self._traits.count()})

    def stop_traits(self) -> None:
        """Stop all registered traits.

        Calls on_stop() on all traits in the registry.
        """
        for trait in self._traits.all():
            try:
                trait.on_stop()
                self._lg.debug("trait stopped", extra={"trait": type(trait).__name__})
            except Exception as e:
                self._lg.warning(
                    "error stopping trait",
                    extra={"trait": type(trait).__name__, "exception": e},
                )

        self._lg.info("all traits stopped", extra={"count": self._traits.count()})

    def cleanup(self) -> None:
        """Cleanup platform resources.

        Called during platform shutdown to release resources.
        """
        self._lg.info("cleaning up platform context")

        # Stop all traits
        self.stop_traits()

        # Future: Close database connections, LLM clients, etc.

        self._lg.info("platform context cleaned up")

    def __repr__(self) -> str:
        """String representation of platform context."""
        trait_names = [type(t).__name__ for t in self._traits.all()]
        return f"PlatformContext(traits=[{', '.join(trait_names)}])"
