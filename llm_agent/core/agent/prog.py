"""Base factory for programmatic agents.

This provides default initialization logic so agents don't need to write
boilerplate Factory code. Most agents can simply subclass and set agent_class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..traits import TraitName
from ..traits.identity import Directive
from .agent import Agent
from .identity import Identity


if TYPE_CHECKING:
    from ..platform import PlatformContext
    from ..traits import BaseTrait


class ProgAgentFactory:
    """Base factory for programmatic agents with standard initialization.

    This factory handles common patterns:
    - Parses profile + identity → constructs Identity
    - Extracts config params → passes to agent __init__
    - Attaches traits based on agent requirements

    Trait Declaration (two options):
        1. In agent class:
            class MyAgent(Agent):
                required_traits = [TraitName.LLM, TraitName.LEARN]

        2. In YAML config:
            traits:
              required: [llm, learn]

    Usage:
        class Factory(ProgAgentFactory):
            agent_class = MyAgent  # Just specify your agent class
    """

    # Subclasses must set this
    agent_class: ClassVar[type[Agent]] = None  # type: ignore[assignment]

    # Registry mapping trait names to trait classes
    TRAIT_REGISTRY: ClassVar[dict[TraitName, type[BaseTrait]]] = {}

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
            RuntimeError: If agent_class not set.
            ValueError: If required config fields missing.
        """
        if self.agent_class is None:
            raise RuntimeError(f"{self.__class__.__name__} must set agent_class class variable")

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
        self._attach_traits(agent)

        return agent

    def _build_identity(self, config: dict[str, Any]) -> Identity:
        """Build Identity from profile fields.

        Args:
            config: Full config dict from manifest.

        Returns:
            Constructed Identity for addressing.

        Raises:
            ValueError: If required fields missing.
        """
        # Extract profile
        profile = config.get("profile", {})
        name = profile.get("name")

        if not name:
            raise ValueError("profile.name is required")

        # Use Identity.from_config to properly resolve IDs
        return Identity.from_config(profile, defaults={"name": name})

    def _build_directive(self, config: dict[str, Any]) -> Directive:
        """Build Directive from identity field.

        Args:
            config: Full config dict from manifest.

        Returns:
            Constructed Directive.

        Raises:
            ValueError: If identity field missing.
        """
        identity_prompt = config.get("identity")
        if not identity_prompt:
            raise ValueError("identity is required")

        if isinstance(identity_prompt, str):
            return Directive(prompt=identity_prompt)
        else:
            return Directive(**identity_prompt)

    def _attach_traits(self, agent: Agent) -> None:
        """Attach all available platform traits to agent.

        Subclasses can override to customize trait attachment or add validation.

        Args:
            agent: Agent instance.
        """
        # Attach all available traits from platform registry
        for trait in self._platform.traits.all():
            agent.add_trait(trait)

        trait_names = [type(t).__name__ for t in self._platform.traits.all()]
        self._lg.debug(
            "attached traits to agent",
            extra={"agent": agent.name, "traits": trait_names},
        )
