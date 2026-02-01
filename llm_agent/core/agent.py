"""Abstract base class for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from appinfra.log import Logger

from llm_agent.core.traits.base import Trait


if TYPE_CHECKING:
    from llm_agent.core.task import Task


TraitT = TypeVar("TraitT", bound=Trait)


class Agent(ABC):
    """Abstract base class for agents.

    Defines the runnable interface that all agents must implement:
    - name: Agent identifier
    - start(): Begin running
    - stop(): Stop running
    - submit(task): Submit a task for execution

    Agents support composable traits for capabilities like LLM access,
    tool use, learning, etc.

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
                # ... agent-specific startup

            def stop(self) -> None:
                self._stop_traits()
                # ... agent-specific cleanup

            def submit(self, task: Task) -> None:
                # ... handle task submission
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

        After start(), the agent runs continuously, working on tasks
        or idling. Call stop() to shut down.

        Subclasses should call _start_traits() to start attached traits.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the agent.

        Gracefully shuts down the agent. After stop(), the agent
        no longer processes tasks.

        Subclasses should call _stop_traits() to stop attached traits.
        """
        ...

    @abstractmethod
    def submit(self, task: Task) -> None:
        """Submit a task for the agent to work on.

        Args:
            task: The task to execute.
        """
        ...

    # =========================================================================
    # Trait Management
    # =========================================================================

    def add_trait(self, trait: Trait) -> None:
        """Add a trait to this agent.

        Traits are attached immediately upon adding.

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
        for trait in self._traits.values():
            trait.on_start()

    def _stop_traits(self) -> None:
        """Stop all attached traits. Call from stop()."""
        for trait in self._traits.values():
            trait.on_stop()
