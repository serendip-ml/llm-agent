"""Factory for creating Agent from configuration.

Creates agents from YAML/dict configuration, wiring up traits automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.agent import Factory as BaseFactory
from ...core.traits.builtin.saia import SAIAConfig, SAIATrait
from .agent import Agent


if TYPE_CHECKING:
    from llm_saia import Backend


class Factory(BaseFactory):
    """Factory for creating prompt-based Agent instances.

    Extends base Factory with SAIA integration for LLM-driven agents.
    Uses base Factory infrastructure for identity, traits, and tools.

    Example:
        platform = PlatformContext.from_config(lg, llm_config, learn_config)
        factory = Factory(platform)

        config = {
            "profile": {"name": "explorer"},
            "directive": "You are a codebase exploration agent.",
            "method": "Think step by step...",
            "tools": {"shell": {}, "read_file": {}},
        }

        agent = factory.create(config)
        agent.start()
    """

    agent_class = Agent  # Use default prompt-based agent

    # Prompt agents typically need these traits (can override in YAML)
    required_traits = []  # Let YAML config specify (directive, method, tools optional)
    default_tools = {}  # Tools specified in YAML config

    def __init__(self, platform: Any) -> None:
        """Initialize factory.

        Args:
            platform: Platform context with all resources.
        """
        super().__init__(platform)
        self._backend: Backend | None = None

    def _get_backend(self) -> Backend:
        """Get or create SAIA backend from llm_config."""
        if self._backend is None:
            from llm_infer.client import Factory as LLMClientFactory
            from llm_infer.client import SAIAAdapter

            llm_client = LLMClientFactory(self._lg).from_config(self._platform.llm_config())
            self._backend = SAIAAdapter(client=llm_client)
        return self._backend

    def create(
        self,
        config: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> Agent:
        """Create prompt-based Agent from configuration.

        Extends base create() to extract default_prompt from task config.

        Args:
            config: Configuration dictionary.
            variables: Variable substitutions for {{VAR}} patterns.

        Returns:
            Configured Agent ready to start.
        """
        # Absolute import to avoid circular dependency
        from llm_agent.core.agent import _substitute_in_dict

        # Apply variable substitutions
        if variables:
            config = _substitute_in_dict(config, variables)

        # Extract default_prompt and inject into config.config for base Factory
        task_config = config.get("task", {})
        default_prompt = task_config.get("description", "")

        if "config" not in config:
            config["config"] = {}
        config["config"]["default_prompt"] = default_prompt

        # Use base Factory.create() - handles identity, traits, tools
        agent = super().create(config, variables=None)  # Already substituted

        # Add SAIA trait and event handlers (prompt-agent specific)
        self._add_saia_trait(agent, config)  # type: ignore[arg-type]
        self._configure_event_handlers(agent, config)  # type: ignore[arg-type]

        return agent  # type: ignore[return-value]

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
        agent.add_trait(SAIATrait(agent, self._get_backend(), saia_config))

    def _build_system_prompt(self, agent: Agent) -> str | None:
        """Build system prompt from identity and method traits."""
        # Absolute import to avoid circular dependency
        from llm_agent.core.traits.builtin.directive import DirectiveTrait, MethodTrait

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
        # Absolute import to avoid circular dependency
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
