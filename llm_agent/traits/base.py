"""Base trait protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from llm_agent.agent import Agent


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
