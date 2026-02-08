"""Trait factory for creating Trait instances from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger


if TYPE_CHECKING:
    from llm_agent.core.tools.factory import ToolFactory
    from llm_agent.core.traits.identity import IdentityTrait, MethodTrait
    from llm_agent.core.traits.learn import LearnTrait
    from llm_agent.core.traits.llm import LLMConfig, LLMTrait
    from llm_agent.core.traits.tools import ToolsTrait


class TraitFactory:
    """Factory for creating Trait instances from configuration.

    Example:
        tool_factory = ToolFactory()
        trait_factory = TraitFactory(lg, tool_factory, llm_config)

        # Create individual traits
        llm_trait = trait_factory.create_llm_trait()
        identity_trait = trait_factory.create_identity_trait("You are helpful.")
        tools_trait = trait_factory.create_tools_trait({"shell": {}})
    """

    def __init__(
        self,
        lg: Logger,
        tool_factory: ToolFactory,
        llm_config: LLMConfig,
        learn_trait: LearnTrait | None = None,
    ) -> None:
        """Initialize factory.

        Args:
            lg: Logger instance.
            tool_factory: Factory for creating tools.
            llm_config: LLM backend configuration.
            learn_trait: Optional LearnTrait for memory capabilities.
        """
        self._lg = lg
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

        return LLMTrait(self._lg, self._llm_config)

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

    def create_tools_trait(
        self, tools_config: dict[str, dict[str, Any]], lg: Logger | None = None
    ) -> ToolsTrait:
        """Create ToolsTrait with tools from configuration.

        Args:
            tools_config: Tool configurations keyed by tool type.
            lg: Optional logger for warnings about skipped tools.

        Returns:
            ToolsTrait with all configured tools registered.
        """
        from llm_agent.core.traits.tools import ToolsTrait

        tools_trait = ToolsTrait()

        for tool_type, tool_config in tools_config.items():
            tool = self._tool_factory.create(tool_type, tool_config)
            if tool is None:
                # Tool could not be created (e.g., memory tool without LearnTrait)
                if lg is not None:
                    lg.debug(f"skipping {tool_type} tool (LearnTrait not available)")
                continue
            tools_trait.register(tool)

        return tools_trait
