"""Factory for creating Agent from configuration.

Creates agents from YAML/dict configuration, wiring up traits automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from llm_agent.agents.default.agent import Agent
from llm_agent.core.agent import Factory as AgentFactory
from llm_agent.core.tools.factory import ToolFactory
from llm_agent.core.traits.saia import SAIAConfig, SAIATrait


if TYPE_CHECKING:
    from llm_saia import Backend

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
        self._backend: Backend | None = None
        self._tool_factory = ToolFactory()
        self._learn_trait = learn_trait
        self._tool_factory.set_learn_trait(learn_trait)

    def _get_backend(self) -> Backend:
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
                - profile: Profile identity config (domain/workspace/name)
                - identity: Agent's identity/persona (string or dict)
                - method: How the agent operates (string)
                - tools: Tool configurations keyed by type
            variables: Variable substitutions for {{VAR}} patterns.

        Returns:
            Configured Agent ready to start.
        """
        from llm_agent.core.agent import Identity, _substitute_in_dict

        variables = variables or {}
        config = _substitute_in_dict(config, variables)

        # Resolve agent identity from config
        profile_config = config.get("profile", {})
        identity = Identity.from_config(profile_config, defaults={"name": config["name"]})

        # Extract default prompt from task config
        task_config = config.get("task", {})
        default_prompt = task_config.get("description", "")

        agent = Agent(self._lg, identity=identity, default_prompt=default_prompt)
        self._add_traits(agent, config)

        return agent

    def _add_traits(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add traits to the agent based on configuration."""
        # LearnTrait must be created before tools so remember/recall bind to the
        # instance that will actually be started (not the factory's template).
        self._add_learn_trait(agent, config)
        self._add_tools_trait(agent, config.get("tools", {}))
        # Add identity traits so we can build system prompt from them
        self._add_identity_traits(agent, config)
        # SAIA needs system prompt from identity/method traits
        self._add_saia_trait(agent, config)
        # Configure event handlers from YAML (if specified)
        self._configure_event_handlers(agent, config)

    def _add_learn_trait(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add LearnTrait if configured.

        Creates a new LearnTrait instance for each agent using the agent's
        resolved ProfileIdentity.

        Args:
            agent: The agent to add the trait to.
            config: Agent configuration dict.
        """
        if self._learn_trait is None:
            return

        from llm_agent.core.traits.learn import LearnConfig, LearnTrait

        # Use the agent's already-resolved ProfileIdentity
        agent_learn_config = LearnConfig(
            identity=agent.identity,
            llm=self._learn_trait.config.llm,
            db=self._learn_trait.config.db,
            embedder_url=self._learn_trait.config.embedder_url,
            embedder_model=self._learn_trait.config.embedder_model,
            embedder_timeout=self._learn_trait.config.embedder_timeout,
        )

        learn_trait = LearnTrait(_lg=self._lg, config=agent_learn_config)
        self._tool_factory.set_learn_trait(learn_trait)
        agent.add_trait(learn_trait)

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
        from llm_agent.core.traits.directive import DirectiveTrait, MethodTrait

        parts: list[str] = []

        directive_trait = agent.get_trait(DirectiveTrait)
        if directive_trait is not None:
            parts.append(directive_trait.directive.prompt)

        method_trait = agent.get_trait(MethodTrait)
        if method_trait is not None:
            parts.append(f"## Method\n{method_trait.method}")

        if not parts:
            return None

        return "\n\n".join(parts)

    def _add_identity_traits(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add identity and method traits if configured."""
        from llm_agent.core.traits.directive import Directive, DirectiveTrait, MethodTrait

        identity_config = config.get("identity")
        if identity_config is not None:
            if isinstance(identity_config, str):
                directive = Directive(prompt=identity_config)
            else:
                directive = Directive(**identity_config)
            agent.add_trait(DirectiveTrait(directive))

        method = config.get("method")
        if method is not None:
            agent.add_trait(MethodTrait(method))

    def _configure_event_handlers(self, agent: Agent, config: dict[str, Any]) -> None:
        """Configure event handlers from YAML config.

        If events are specified in config, create custom handlers that override
        the default handlers. This allows declarative configuration of memory
        strategies and prompt composition per event.

        Args:
            agent: The agent to configure handlers for.
            config: Agent configuration dict (may contain 'events' section).
        """
        events_config = config.get("events", {})
        if not events_config:
            # No custom event config - agent will use default handlers
            return

        # For each configured event, create a custom handler
        for event_name, event_config in events_config.items():
            handler = self._create_event_handler(agent, event_name, event_config)
            # Register handler - this will override default handlers after agent.start()
            # We need to register after traits are added but before start() is called
            agent._dispatcher.on(event_name, handler)

    def _create_event_handler(
        self, agent: Agent, event_name: str, event_config: dict[str, Any]
    ) -> Any:
        """Create an event handler from configuration.

        Args:
            agent: The agent this handler is for.
            event_name: Name of the event (schedule, question).
            event_config: Event configuration dict with recall_strategy, etc.

        Returns:
            Async handler function.
        """
        recall_strategy = event_config.get("recall_strategy", "chronological")
        recall_limit = event_config.get("recall_limit", 5)

        async def handler(**kwargs: Any) -> dict[str, Any]:
            """Custom event handler from YAML config."""
            return await self._execute_event_handler(
                agent_name=kwargs.get("agent_name", agent.name),
                saia_trait=kwargs.get("saia_trait"),
                learn_trait=kwargs.get("learn_trait"),
                task=kwargs.get("task") or kwargs.get("question", ""),
                recall_strategy=recall_strategy,
                recall_limit=recall_limit,
            )

        return handler

    async def _execute_event_handler(
        self,
        agent_name: str,
        saia_trait: Any,
        learn_trait: Any | None,
        task: str,
        recall_strategy: str,
        recall_limit: int,
    ) -> dict[str, Any]:
        """Execute event handler with configured memory strategy."""
        if saia_trait is None:
            return {
                "success": False,
                "content": "SAIATrait not attached",
                "iterations": 0,
                "tokens_used": 0,
            }

        context = self._recall_context(learn_trait, task, agent_name, recall_strategy, recall_limit)
        prompt = saia_trait.saia.compose(context, task)
        saia_result = await saia_trait.saia.complete(prompt)

        return {
            "success": saia_result.completed,
            "content": saia_result.output,
            "iterations": saia_result.iterations,
            "tokens_used": saia_result.score.total_tokens if saia_result.score else 0,
            "trace_id": saia_result.trace_id,
        }

    def _recall_context(
        self,
        learn_trait: Any | None,
        task: str,
        agent_name: str,
        recall_strategy: str,
        recall_limit: int,
    ) -> str:
        """Recall past solutions and format as context string."""
        from llm_agent.core.memory import (
            format_solutions_context,
            recall_chronological,
            recall_semantic,
        )

        if learn_trait is None:
            return ""

        if recall_strategy == "semantic":
            past = recall_semantic(
                learn_trait, query=task, limit=recall_limit, agent_name=agent_name
            )
        else:  # chronological
            past = recall_chronological(learn_trait, agent_name, limit=recall_limit)

        return format_solutions_context(past)
