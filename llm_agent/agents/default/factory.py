"""Factory for creating Agent from configuration.

Creates agents from YAML/dict configuration, wiring up traits automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from llm_agent.agents.default.agent import Agent
from llm_agent.core.factory import AgentFactory, ToolFactory
from llm_agent.core.traits.saia import SAIAConfig, SAIATrait


if TYPE_CHECKING:
    from llm_saia import SAIABackend

    from llm_agent.core.traits.learn import LearnTrait


class Factory(AgentFactory):
    """Factory for creating Agent instances from configuration.

    Creates SAIA backend from llm_config internally.

    Example:
        factory = Factory(lg, llm_config={"default": "local", "backends": {...}})

        config = {
            "name": "explorer",
            "identity": "You are a codebase exploration agent.",
            "tools": {"shell": {}, "read_file": {}},
        }

        agent = factory.create(config)
        agent.start()
    """

    def __init__(
        self,
        lg: Logger,
        llm_config: dict[str, Any],
        learn_trait: LearnTrait | None = None,
    ) -> None:
        """Initialize factory.

        Args:
            lg: Logger instance for created agents.
            llm_config: LLM configuration dict for backend creation.
            learn_trait: Optional LearnTrait for memory capabilities.
        """
        super().__init__(lg, llm_config)
        self._backend: SAIABackend | None = None
        self._tool_factory = ToolFactory()
        self._learn_trait = learn_trait
        self._tool_factory.set_learn_trait(learn_trait)

    def _get_backend(self) -> SAIABackend:
        """Get or create SAIA backend from llm_config."""
        if self._backend is None:
            from llm_infer.client import Factory as LLMClientFactory
            from llm_infer.client import SAIAAdapter

            llm_client = LLMClientFactory(self._lg).from_config(self._llm_config)
            self._backend = SAIAAdapter(client=llm_client)
        return self._backend

    def create(
        self,
        config: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> Agent:
        """Create a Agent from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - name (required): Agent identifier
                - identity: Agent's identity/persona (string or dict)
                - method: How the agent operates (string)
                - tools: Tool configurations keyed by type
            variables: Variable substitutions for {{VAR}} patterns.

        Returns:
            Configured Agent ready to start.
        """
        from llm_agent.core.factory import _substitute_in_dict

        variables = variables or {}
        config = _substitute_in_dict(config, variables)

        # Extract default prompt from task config
        task_config = config.get("task", {})
        default_prompt = task_config.get("description", "")

        agent = Agent(self._lg, name=config["name"], default_prompt=default_prompt)
        self._add_traits(agent, config)

        return agent

    def _add_traits(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add traits to the agent based on configuration."""
        self._add_tools_trait(agent, config.get("tools", {}))
        # Add identity traits first so we can build system prompt from them
        self._add_identity_traits(agent, config)
        # SAIA needs system prompt from identity/method traits
        self._add_saia_trait(agent, config)
        if self._learn_trait is not None:
            agent.add_trait(self._learn_trait)

    def _add_tools_trait(self, agent: Agent, tools_config: dict[str, Any]) -> None:
        """Add ToolsTrait with configured tools."""
        if not tools_config:
            return
        from llm_agent.core.traits.tools import ToolsTrait

        tools_trait = ToolsTrait()
        for tool_type, tool_config in tools_config.items():
            tool = self._tool_factory.create(tool_type, tool_config)
            if tool is not None:
                tools_trait.register(tool)
        agent.add_trait(tools_trait)

    def _add_saia_trait(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add SAIATrait for LLM operations."""
        # Build system prompt from identity and method traits
        system_prompt = self._build_system_prompt(agent)

        saia_config = SAIAConfig(
            terminal_tool=config.get("terminal_tool", "complete_task"),
            max_iterations=config.get("max_iterations", 0),
            timeout_secs=config.get("timeout_secs", 0),
            system_prompt=system_prompt,
        )
        agent.add_trait(SAIATrait(_lg=self._lg, backend=self._get_backend(), config=saia_config))

    def _build_system_prompt(self, agent: Agent) -> str | None:
        """Build system prompt from identity and method traits."""
        from llm_agent.core.traits.identity import IdentityTrait, MethodTrait

        parts: list[str] = []

        identity_trait = agent.get_trait(IdentityTrait)
        if identity_trait is not None:
            parts.append(identity_trait.identity.prompt)

        method_trait = agent.get_trait(MethodTrait)
        if method_trait is not None:
            parts.append(f"## Method\n{method_trait.method}")

        if not parts:
            return None

        return "\n\n".join(parts)

    def _add_identity_traits(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add identity and method traits if configured."""
        from llm_agent.core.traits.identity import Identity, IdentityTrait, MethodTrait

        identity_config = config.get("identity")
        if identity_config is not None:
            if isinstance(identity_config, str):
                identity = Identity(prompt=identity_config)
            else:
                identity = Identity(**identity_config)
            agent.add_trait(IdentityTrait(identity))

        method = config.get("method")
        if method is not None:
            agent.add_trait(MethodTrait(method))
