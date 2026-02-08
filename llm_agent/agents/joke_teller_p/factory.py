"""Factory for creating JokeTellerAgent instances."""

from __future__ import annotations

from ...core.agent import Agent
from ...core.agent.prog import ProgAgentFactory
from .agent import JokeTellerAgent


class Factory(ProgAgentFactory):
    """Factory for JokeTellerAgent.

    Uses base factory with standard initialization.
    Only needs to specify the agent class - everything else is handled by base.
    """

    agent_class = JokeTellerAgent

    def _attach_traits(self, agent: Agent) -> None:
        """Attach traits and ensure LearnTrait is available.

        JokeTellerAgent requires LearnTrait for novelty checking.

        Raises:
            RuntimeError: If LearnTrait is not configured.
        """
        # Use base factory's trait attachment
        super()._attach_traits(agent)

        # Verify LearnTrait was attached (required for this agent)
        from ...core.traits.learn import LearnTrait

        if agent.get_trait(LearnTrait) is None:
            raise RuntimeError(
                "LearnTrait is required for JokeTellerAgent but learning is not configured. "
                "Add 'learn' section to server config."
            )
