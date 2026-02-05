"""Abstract base class for agents."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from appinfra.log import Logger

from .runnable import ExecutionResult, Runnable
from .traits.base import Trait


TraitT = TypeVar("TraitT", bound=Trait)


class Agent(Runnable):
    """Abstract base class for agents.

    Provides trait composition and lifecycle management. Subclasses
    implement the concrete agent behavior.

    For LLM operations, attach SAIATrait and call agent.get_trait(SAIATrait).saia
    directly. For learning, attach LearnTrait and access via get_trait().

    Example:
        class MyAgent(Agent):
            def __init__(self, lg: Logger, name: str) -> None:
                super().__init__(lg)
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            def start(self) -> None:
                self._start_traits()

            def stop(self) -> None:
                self._stop_traits()
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
        """
        self._lg = lg
        self._traits: dict[type[Trait], Trait] = {}
        self._started = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start the agent.

        Subclasses should call _start_traits() to start attached traits.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the agent.

        Subclasses should call _stop_traits() to stop attached traits.
        """
        ...

    @abstractmethod
    def record_feedback(self, message: str) -> None:
        """Record feedback about execution.

        Args:
            message: The feedback message.
        """
        ...

    @abstractmethod
    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent ExecutionResult objects.
        """
        ...

    # =========================================================================
    # Trait Management
    # =========================================================================

    def add_trait(self, trait: Trait) -> None:
        """Add a trait to this agent.

        Traits are attached immediately upon adding. If the agent is already
        started, the trait's on_start() is called automatically.

        Args:
            trait: The trait instance to add.

        Raises:
            ValueError: If a trait of this type is already added.
        """
        trait_type = type(trait)
        if trait_type in self._traits:
            raise ValueError(f"Trait {trait_type.__name__} already added")
        self._traits[trait_type] = trait
        trait.attach(self)
        if self._started:
            trait.on_start()

    def get_trait(self, trait_type: type[TraitT]) -> TraitT | None:
        """Get an attached trait by its type.

        Args:
            trait_type: The trait class to look up.

        Returns:
            The trait instance, or None if not attached.
        """
        return self._traits.get(trait_type)  # type: ignore[return-value]

    def has_trait(self, trait_type: type[Trait]) -> bool:
        """Check if a trait is attached.

        Args:
            trait_type: The trait class to check.

        Returns:
            True if the trait is attached.
        """
        return trait_type in self._traits

    def require_trait(self, trait_type: type[TraitT]) -> TraitT:
        """Get a required trait, raising if not attached.

        Args:
            trait_type: The trait class to look up.

        Returns:
            The trait instance.

        Raises:
            RuntimeError: If the trait is not attached.
        """
        trait = self.get_trait(trait_type)
        if trait is None:
            raise RuntimeError(
                f"{trait_type.__name__} required but not attached - "
                f"add it with agent.add_trait({trait_type.__name__}(...))"
            )
        return trait

    # =========================================================================
    # Trait Lifecycle Helpers
    # =========================================================================

    def _start_traits(self) -> None:
        """Start all attached traits. Call from start()."""
        self._started = True
        for trait in self._traits.values():
            trait.on_start()

    def _stop_traits(self) -> None:
        """Stop all attached traits. Call from stop()."""
        for trait in self._traits.values():
            trait.on_stop()
        self._started = False
