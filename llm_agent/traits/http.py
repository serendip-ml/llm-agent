"""HTTP server trait for agents."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from fastapi import APIRouter


if TYPE_CHECKING:
    from llm_agent.agent import Agent
    from llm_agent.server.http import HTTPServer


@dataclass
class HTTPConfig:
    """Configuration for HTTP trait.

    Attributes:
        host: Host to bind to.
        port: Port to bind to. Use 0 for ephemeral port.
        title: API title for OpenAPI docs (defaults to "{agent_name} API").
        description: API description for OpenAPI docs.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    title: str | None = None
    description: str | None = None


@dataclass
class HTTPTrait:
    """HTTP server trait.

    Runs a FastAPI server in a subprocess and communicates via IPC queues.
    The agent provides route factory and request handler methods.

    Architecture:
        HTTP Request -> FastAPI subprocess -> request_queue -> IPC thread -> agent.handle_request()
        HTTP Response <- FastAPI subprocess <- response_queue <- agent.handle_request()

    The agent should implement:
        - create_routes() -> APIRouter: Creates the API routes
        - handle_request(request) -> response: Handles IPC requests

    Example:
        from appinfra.log import Logger

        lg = Logger.create("agent")
        agent = Agent(config)
        agent.add_trait(HTTPTrait(lg, HTTPConfig(port=8080)))

    Lifecycle:
        - attach(): Creates HTTPServer instance
        - on_start(): Starts server subprocess and IPC thread
        - on_stop(): Stops IPC thread and server
    """

    lg: Logger
    config: HTTPConfig = field(default_factory=HTTPConfig)

    _agent: Agent | None = field(default=None, repr=False, compare=False)
    _server: HTTPServer | None = field(default=None, repr=False, compare=False)
    _ipc_thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    _ipc_shutdown: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent and create HTTP server.

        Args:
            agent: The agent this trait is attached to.
        """
        self._agent = agent

        from llm_agent.server.http import HTTPServer, HTTPServerConfig

        router_factory = self._get_router_factory(agent)
        title = self.config.title or f"{agent.name} API"
        description = self.config.description or f"HTTP API for {agent.name}"

        http_config = HTTPServerConfig(
            host=self.config.host,
            port=self.config.port,
            title=title,
            description=description,
        )
        self._server = HTTPServer(http_config, router_factory=router_factory)
        self.lg.info(
            "HTTP server configured",
            extra={"host": self.config.host, "port": self.config.port},
        )

    def _get_router_factory(self, agent: Agent) -> Callable[[], APIRouter]:
        """Get router factory from agent or use default."""
        from llm_agent.server.routes import create_agent_routes

        create_routes: Callable[[], APIRouter] | None = getattr(agent, "create_routes", None)
        if create_routes is not None:
            return create_routes
        return create_agent_routes

    def on_start(self) -> None:
        """Start HTTP server and IPC thread."""
        if self._server is None:
            raise RuntimeError("HTTPTrait not attached - call attach() first")

        self._server.start()
        self._start_ipc_thread()
        self.lg.info("HTTP server started")

    def on_stop(self) -> None:
        """Stop IPC thread and HTTP server."""
        self._stop_ipc_thread()
        if self._server:
            self._server.stop()
            self.lg.info("HTTP server stopped")

    @property
    def server(self) -> HTTPServer | None:
        """HTTP server instance (available after attach)."""
        return self._server

    @property
    def url(self) -> str | None:
        """Base URL for the HTTP server."""
        if self.config.port == 0:
            return None
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def is_running(self) -> bool:
        """Check if server subprocess is running."""
        return self._server is not None and self._server.is_running

    # -------------------------------------------------------------------------
    # IPC Thread Management
    # -------------------------------------------------------------------------

    def _start_ipc_thread(self) -> None:
        """Start background thread for IPC processing."""
        assert self._agent is not None
        self._ipc_shutdown.clear()
        self._ipc_thread = threading.Thread(
            target=self._ipc_loop,
            daemon=True,
            name=f"{self._agent.name}-http-ipc",
        )
        self._ipc_thread.start()
        self.lg.debug("IPC processing thread started")

    def _stop_ipc_thread(self) -> None:
        """Stop IPC processing thread."""
        if self._ipc_thread is None:
            return
        self._ipc_shutdown.set()
        self._ipc_thread.join(timeout=2.0)
        self._ipc_thread = None
        self.lg.debug("IPC processing thread stopped")

    def _ipc_loop(self, poll_timeout: float = 0.1) -> None:
        """Background loop processing IPC requests.

        Args:
            poll_timeout: Queue polling timeout in seconds.
        """
        assert self._server is not None
        handle_request: Callable[[Any], Any] = getattr(
            self._agent, "handle_request", self._default_handler
        )

        while not self._ipc_shutdown.is_set():
            try:
                request = self._server.request_queue.get(timeout=poll_timeout)
                response = handle_request(request)
                if response is not None:
                    self._server.response_queue.put(response)
            except Empty:
                pass
            except Exception:
                self.lg.exception("IPC processing error")

    def _default_handler(self, request: Any) -> Any:
        """Default request handler when agent doesn't implement handle_request."""
        self.lg.warning(
            "No handle_request method on agent, request ignored",
            extra={"request_type": type(request).__name__},
        )
        return None
