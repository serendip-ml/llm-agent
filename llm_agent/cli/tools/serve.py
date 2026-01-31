"""Serve tool - starts the agent gateway server."""

from __future__ import annotations

import argparse
import signal
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.app.fastapi import ServerBuilder
from appinfra.app.tools import Tool, ToolConfig


if TYPE_CHECKING:
    from llm_agent.core.traits.learn import LearnTrait
    from llm_agent.runtime import AgentRegistry, Core
    from llm_agent.runtime.server import AgentServerConfig


class ServeTool(Tool):
    """Start the agent gateway server."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="serve", help_text="Start the agent gateway server")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host",
            help="Host to bind to (overrides config)",
        )
        parser.add_argument(
            "--port",
            type=int,
            help="Port to bind to (overrides config)",
        )
        parser.add_argument(
            "-e",
            "--env",
            action="append",
            metavar="KEY=VALUE",
            dest="env_vars",
            help="Set environment variable for agent configs (e.g., -e CODEBASE_PATH=/path/to/code)",
        )

    def _parse_env_vars(self) -> dict[str, str]:
        """Parse -e KEY=VALUE arguments into a dict."""
        env_vars = {}
        if self.args.env_vars:
            for item in self.args.env_vars:
                if "=" in item:
                    key, value = item.split("=", 1)
                    env_vars[key] = value
        return env_vars

    def run(self, **kwargs: Any) -> int:
        from llm_agent.runtime.server import AgentServerConfig

        raw_config = dict(self.app.config) if self.app.config else {}
        config = AgentServerConfig.from_dict(raw_config)
        self._apply_cli_overrides(config)

        learn_trait = self._create_learn_trait(config)
        registry = self._create_registry()
        core = self._create_core(registry, config, learn_trait)
        self._register_agents(registry, core, config)

        self._log_startup(config)
        self._run_server(config, core, learn_trait)
        return 0

    def _create_learn_trait(self, config: AgentServerConfig) -> LearnTrait | None:
        """Create shared learn trait if configured."""
        from llm_agent.core.traits.learn import LearnConfig, LearnTrait

        if config.learn is None:
            return None

        return LearnTrait(
            LearnConfig(
                profile_id=config.learn.profile_id,
                llm=config.llm,
                db_config_path=config.learn.db_config_path,
                db_key=config.learn.db_key,
                embedder_url=config.learn.embedder_url,
                embedder_model=config.learn.embedder_model,
            )
        )

    def _create_registry(self) -> AgentRegistry:
        """Create agent registry."""
        from llm_agent.runtime import AgentRegistry

        return AgentRegistry(lg=self.lg)

    def _create_core(
        self,
        registry: AgentRegistry,
        config: AgentServerConfig,
        learn_trait: LearnTrait | None,
    ) -> Core:
        """Create runtime core."""
        from llm_agent.runtime import Core

        return Core(
            lg=self.lg,
            registry=registry,
            llm_config=config.llm,
            learn_trait=learn_trait,
            variables=self._parse_env_vars(),
        )

    def _log_startup(self, config: AgentServerConfig) -> None:
        """Log server startup information."""
        self.lg.info(
            "starting agent gateway",
            extra={
                "host": config.server.host,
                "port": config.server.port,
                "agents": list(config.agents.keys()),
            },
        )

    def _run_server(
        self,
        config: AgentServerConfig,
        core: Core,
        learn_trait: LearnTrait | None,
    ) -> None:
        """Run the server with signal handling."""
        if learn_trait is not None:
            learn_trait.on_start()

        shutdown_done = False

        def do_shutdown() -> None:
            nonlocal shutdown_done
            if shutdown_done:
                return
            shutdown_done = True
            core.shutdown()
            if learn_trait is not None:
                learn_trait.on_stop()

        self._install_signal_handlers(do_shutdown)
        server = self._build_server(config, core)

        try:
            server.start()
        finally:
            do_shutdown()

    def _install_signal_handlers(self, do_shutdown: Callable[[], None]) -> None:
        """Install signal handlers for graceful shutdown.

        Note: do_shutdown() calls core.shutdown() which uses bounded timeouts
        (5s graceful + 2s terminate + kill) per agent, so this won't hang indefinitely.
        """

        def shutdown_handler(signum: int, frame: Any) -> None:
            self.lg.info("shutdown signal received")
            do_shutdown()

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

    def _build_server(self, config: AgentServerConfig, core: Core) -> Any:
        """Build the FastAPI server with routes."""
        from appinfra.app.fastapi.runtime.server import Server

        from llm_agent.runtime.server.management import create_management_routes

        server: Server = (
            ServerBuilder("agent-gateway")
            .with_config(config.server)
            .routes.with_router(create_management_routes())
            .done()
            .uvicorn.with_config(config.server.uvicorn)
            .done()
            .build()
        )
        server.app.state.core = core
        return server

    def _register_agents(
        self, registry: AgentRegistry, core: Core, config: AgentServerConfig
    ) -> None:
        """Register and auto-start agents from configuration."""
        for name, agent_config in config.agents.items():
            if agent_config.class_ != "prompt":
                self.lg.warning(
                    "skipping non-prompt agent",
                    extra={"agent": name, "class": agent_config.class_},
                )
                continue

            config_dict = self._build_agent_config_dict(name, agent_config)
            try:
                registry.register(name, config_dict)
                # Auto-start agents with a schedule
                if agent_config.schedule is not None:
                    core.start(name)
            except Exception as e:
                self.lg.error("failed to register agent", extra={"agent": name, "exception": e})

    def _build_agent_config_dict(self, name: str, agent_config: Any) -> dict[str, Any]:
        """Build config dict for agent registration."""
        return {
            "name": name,
            "directive": agent_config.directive.model_dump(),
            "task": agent_config.task.model_dump(),
            "tools": agent_config.tools,
            "schedule": agent_config.schedule.model_dump() if agent_config.schedule else None,
        }

    def _apply_cli_overrides(self, config: Any) -> None:
        """Apply command-line overrides to config."""
        if self.args.host:
            config.server.host = self.args.host
        if self.args.port:
            config.server.port = self.args.port
