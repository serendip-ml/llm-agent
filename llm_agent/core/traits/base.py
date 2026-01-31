"""Base trait protocol and convenience base class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent


@runtime_checkable
class Trait(Protocol):
    """Protocol for agent traits.

    Traits extend agent capabilities through composition. Each trait
    encapsulates a specific capability and integrates with the agent
    lifecycle:

    - attach(agent): Called during agent setup to get agent reference
    - on_start(): Called when agent starts
    - on_stop(): Called when agent stops

    Traits can depend on other traits. Use agent.get_trait() in attach()
    to get references to required traits.

    For convenience, inherit from BaseTrait instead of implementing this
    protocol directly.
    """

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent during setup.

        Called before on_start(). Use to get agent reference and
        resolve dependencies on other traits via agent.get_trait().

        Args:
            agent: The agent this trait is attached to.
        """
        ...

    def on_start(self) -> None:
        """Called when agent starts."""
        ...

    def on_stop(self) -> None:
        """Called when agent stops."""
        ...


class BaseTrait:
    """Convenience base class for custom traits.

    Provides common boilerplate: agent storage, property access, lifecycle stubs.
    Users can also implement the Trait protocol directly if they need different
    patterns (e.g., dataclass-based traits like HTTPTrait).

    Example:
        class MyTrait(BaseTrait):
            def __init__(self, some_config: str) -> None:
                super().__init__()
                self._config = some_config

            def do_something(self) -> str:
                # Access agent via self.agent property
                result = self.agent.complete("query")
                return result.content

        agent.add_trait(MyTrait("config"))
        my_trait = agent.get_trait(MyTrait)
        my_trait.do_something()
    """

    def __init__(self) -> None:
        """Initialize base trait."""
        self._agent: Agent | None = None

    @property
    def agent(self) -> Agent:
        """Access the attached agent.

        Raises:
            RuntimeError: If trait is not attached to an agent.
        """
        if self._agent is None:
            raise RuntimeError(
                f"{type(self).__name__} not attached to agent - "
                "ensure add_trait() was called before accessing agent"
            )
        return self._agent

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent.

        Args:
            agent: The agent this trait is attached to.
        """
        self._agent = agent

    def on_start(self) -> None:
        """Called when agent starts. Override to add startup logic."""
        pass

    def on_stop(self) -> None:
        """Called when agent stops. Override to add cleanup logic."""
        pass
