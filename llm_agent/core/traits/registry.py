"""Trait registry for managing available platform traits.

The registry is a container for all configured platform traits (LLM, Learn, HTTP, etc.)
that can be attached to agents. It provides type-safe access and discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from appinfra.log import Logger

from ..errors import TraitAlreadyRegisteredError, TraitNotFoundError
from .base import BaseTrait


if TYPE_CHECKING:
    from . import TraitName


T = TypeVar("T", bound=BaseTrait)


class Registry:
    """Registry of available platform traits.

    Manages the collection of configured traits that can be attached to agents.
    Provides type-safe access and trait discovery.

    Example:
        # Platform creates registry with configured traits
        registry = Registry(lg)
        registry.register(LLMTrait(lg, llm_config))
        registry.register(LearnTrait(lg, learn_config))

        # Factory accesses traits by type
        llm_trait = registry.get(LLMTrait)
        if registry.has(LearnTrait):
            learn_trait = registry.get(LearnTrait)

        # Or by name (if trait declares trait_name)
        llm_trait = registry.get_by_name(TraitName.LLM)

        # Attach all available traits to agent
        for trait in registry.all():
            agent.add_trait(trait)
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize empty trait registry.

        Args:
            lg: Logger instance for registry operations.
        """
        self._lg = lg
        self._traits: dict[type[BaseTrait], BaseTrait] = {}
        self._by_name: dict[TraitName, BaseTrait] = {}

    def register(self, trait: BaseTrait) -> None:
        """Register a trait in the registry.

        Automatically registers by name if trait has trait_name attribute.

        Args:
            trait: Trait instance to register.

        Raises:
            TraitAlreadyRegisteredError: If a trait of this type is already registered.
        """
        trait_type = type(trait)
        if trait_type in self._traits:
            raise TraitAlreadyRegisteredError(
                f"Trait {trait_type.__name__} is already registered. "
                "Use replace() to update an existing trait."
            )

        # Check for name collision before registering
        if hasattr(trait, "trait_name"):
            name = trait.trait_name
            existing = self._by_name.get(name)
            if existing is not None and type(existing) is not trait_type:
                raise TraitAlreadyRegisteredError(
                    f"Trait name '{name}' is already registered for {type(existing).__name__}."
                )

        self._traits[trait_type] = trait

        # Auto-register by name if trait declares one
        if hasattr(trait, "trait_name"):
            self._by_name[trait.trait_name] = trait

        self._lg.debug(
            "trait registered",
            extra={"trait": trait_type.__name__, "total": len(self._traits)},
        )

    def replace(self, trait: BaseTrait) -> None:
        """Replace an existing trait or register if not present.

        Args:
            trait: Trait instance to register/replace.
        """
        trait_type = type(trait)
        old = self._traits.get(trait_type)
        self._traits[trait_type] = trait

        # Update name mapping if trait has one, or remove stale name if replacing with unnamed
        if hasattr(trait, "trait_name"):
            self._by_name[trait.trait_name] = trait
        elif old is not None and hasattr(old, "trait_name"):
            self._by_name.pop(old.trait_name, None)

        self._lg.debug("trait replaced", extra={"trait": trait_type.__name__})

    def get(self, trait_type: type[T]) -> T | None:
        """Get a trait by type.

        Args:
            trait_type: The trait class to retrieve.

        Returns:
            Trait instance if registered, None otherwise.
        """
        return self._traits.get(trait_type)  # type: ignore[return-value]

    def get_by_name(self, name: TraitName) -> BaseTrait | None:
        """Get a trait by name.

        Args:
            name: The trait name to retrieve.

        Returns:
            Trait instance if registered with this name, None otherwise.
        """
        return self._by_name.get(name)

    def require(self, trait_type: type[T]) -> T:
        """Get a required trait by type.

        Args:
            trait_type: The trait class to retrieve.

        Returns:
            Trait instance.

        Raises:
            TraitNotFoundError: If trait is not registered.
        """
        trait = self.get(trait_type)
        if trait is None:
            raise TraitNotFoundError(
                f"Trait {trait_type.__name__} is required but not registered. "
                "Check platform configuration."
            )
        return trait

    def require_by_name(self, name: TraitName) -> BaseTrait:
        """Get a required trait by name.

        Args:
            name: The trait name to retrieve.

        Returns:
            Trait instance.

        Raises:
            TraitNotFoundError: If trait with this name is not registered.
        """
        trait = self.get_by_name(name)
        if trait is None:
            raise TraitNotFoundError(
                f"Trait '{name.value}' is required but not registered. "
                "Check platform configuration."
            )
        return trait

    def has(self, trait_type: type[BaseTrait]) -> bool:
        """Check if a trait type is registered.

        Args:
            trait_type: The trait class to check.

        Returns:
            True if trait is registered, False otherwise.
        """
        return trait_type in self._traits

    def all(self) -> list[BaseTrait]:
        """Get all registered traits.

        Returns:
            List of all registered trait instances.
        """
        return list(self._traits.values())

    def types(self) -> list[type[BaseTrait]]:
        """Get all registered trait types.

        Returns:
            List of registered trait classes.
        """
        return list(self._traits.keys())

    def count(self) -> int:
        """Get number of registered traits.

        Returns:
            Count of registered traits.
        """
        return len(self._traits)

    def clear(self) -> None:
        """Remove all traits from registry."""
        self._traits.clear()
        self._by_name.clear()
        self._lg.debug("trait registry cleared")

    def __repr__(self) -> str:
        """String representation of registry."""
        trait_names = [t.__name__ for t in self._traits]
        return f"Registry({', '.join(trait_names)})"
