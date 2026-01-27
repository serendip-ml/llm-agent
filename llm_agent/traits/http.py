"""HTTP trait for exposing agent via REST API."""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.queues import Queue

    from llm_agent.agent import Agent
    from llm_agent.http.models import AgentRequest


@dataclass
class HTTPConfig:
    """Configuration for HTTP trait.

    Args:
        host: Bind address (default: "127.0.0.1")
        port: Bind port (default: 8080)
        title: API title for OpenAPI docs (default: "Agent API")
        response_timeout: Request timeout in seconds (default: 60.0)
    """

    host: str = "127.0.0.1"
    port: int = 8080
    title: str = "Agent API"
    response_timeout: float = 60.0


@dataclass
class HTTPTrait:
    """Trait that exposes agent capabilities via HTTP REST API.

    Runs a FastAPI server in a subprocess, communicating with the agent
    via multiprocessing queues. This keeps the main process free for
    agent work (tick loop, background tasks).

    Example:
        agent = Agent(config=config, llm=llm, learn=learn)
        agent.add_trait(HTTPTrait(HTTPConfig(port=8080)))

        http = agent.get_trait(HTTPTrait)
        http.on_start()  # Start server subprocess

        try:
            while True:
                agent.tick()  # Main process work
                time.sleep(1)
        finally:
            http.on_stop()
    """

    config: HTTPConfig = field(default_factory=HTTPConfig)

    _agent: Agent | None = field(default=None, repr=False, compare=False)
    _request_queue: Queue[Any] | None = field(default=None, repr=False, compare=False)
    _response_queue: Queue[Any] | None = field(default=None, repr=False, compare=False)
    _process: Process | None = field(default=None, repr=False, compare=False)
    _ipc_thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent during setup."""
        self._agent = agent

    def on_start(self) -> None:
        """Start HTTP server subprocess and IPC handler thread."""
        if self._agent is None:
            raise RuntimeError("HTTPTrait not attached to agent")

        server = self._build_server()
        self._process = server.start_subprocess()
        self._start_ipc_thread()

    def _build_server(self) -> Any:
        """Build the FastAPI server with IPC configuration."""
        from appinfra.app.fastapi import ServerBuilder

        from llm_agent.http.routes import router

        self._request_queue = mp.Queue()
        self._response_queue = mp.Queue()

        return (
            ServerBuilder(self.config.title)
            .with_host(self.config.host)
            .with_port(self.config.port)
            .with_title(self.config.title)
            .subprocess.with_ipc(self._request_queue, self._response_queue)
            .with_response_timeout(self.config.response_timeout)
            .with_auto_restart(enabled=True)
            .done()
            .routes.with_router(router)
            .done()
            .build()
        )

    def _start_ipc_thread(self) -> None:
        """Start the IPC handler thread."""
        self._stop_event.clear()
        self._ipc_thread = threading.Thread(
            target=self._handle_ipc,
            name="http-trait-ipc",
            daemon=True,
        )
        self._ipc_thread.start()

    def on_stop(self) -> None:
        """Stop IPC handler thread and server subprocess."""
        self._stop_event.set()

        if self._ipc_thread is not None:
            self._ipc_thread.join(timeout=5.0)
            self._ipc_thread = None

        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=5.0)
            self._process = None

        if self._request_queue is not None:
            self._request_queue.close()
            self._request_queue = None
        if self._response_queue is not None:
            self._response_queue.close()
            self._response_queue = None

    @property
    def is_running(self) -> bool:
        """Check if server subprocess is running."""
        return self._process is not None and self._process.is_alive()

    def _handle_ipc(self) -> None:
        """IPC handler thread that dispatches requests to agent methods."""
        while not self._stop_event.is_set():
            try:
                request: AgentRequest = self._request_queue.get(timeout=0.1)  # type: ignore[union-attr]
            except Empty:
                continue

            response = self._dispatch(request)
            self._response_queue.put(response)  # type: ignore[union-attr]

    def _dispatch(self, request: Any) -> Any:
        """Dispatch an IPC request to the appropriate agent method."""
        from llm_agent.http.models import AgentRequest, AgentResponse

        req = AgentRequest(**request) if isinstance(request, dict) else request
        agent = self._agent
        assert agent is not None

        handlers = {
            "health": self._handle_health,
            "complete": self._handle_complete,
            "remember": self._handle_remember,
            "forget": self._handle_forget,
            "recall": self._handle_recall,
            "feedback": self._handle_feedback,
        }

        handler = handlers.get(req.method)
        if handler is None:
            return AgentResponse(
                id=req.id, success=False, result=None, error=f"Unknown method: {req.method}"
            )

        try:
            result = handler(agent, req.params)
            return AgentResponse(id=req.id, success=True, result=result)
        except Exception as e:
            return AgentResponse(id=req.id, success=False, result=None, error=str(e))

    def _handle_health(self, agent: Agent, params: dict[str, Any]) -> dict[str, Any]:
        """Handle health check request."""
        return {"status": "ok", "agent_name": agent.name}

    def _handle_complete(self, agent: Agent, params: dict[str, Any]) -> dict[str, Any]:
        """Handle completion request."""
        completion = agent.complete(
            query=params["query"],
            system_prompt=params.get("system_prompt"),
        )
        return {
            "id": completion.id,
            "content": completion.content,
            "model": completion.model,
            "tokens_used": completion.tokens_used,
        }

    def _handle_remember(self, agent: Agent, params: dict[str, Any]) -> dict[str, Any]:
        """Handle remember request."""
        fact_id = agent.remember(
            fact=params["fact"],
            category=params.get("category", "general"),
        )
        return {"fact_id": fact_id}

    def _handle_forget(self, agent: Agent, params: dict[str, Any]) -> dict[str, Any]:
        """Handle forget request."""
        agent.forget(fact_id=params["fact_id"])
        return {"success": True}

    def _handle_recall(self, agent: Agent, params: dict[str, Any]) -> dict[str, Any]:
        """Handle recall request."""
        scored_facts = agent.recall(
            query=params["query"],
            top_k=params.get("top_k"),
            min_similarity=params.get("min_similarity"),
            categories=params.get("categories"),
        )
        return {
            "facts": [
                {
                    "id": sf.fact.id,
                    "content": sf.fact.content,
                    "category": sf.fact.category,
                    "similarity": sf.similarity,
                }
                for sf in scored_facts
            ]
        }

    def _handle_feedback(self, agent: Agent, params: dict[str, Any]) -> dict[str, Any]:
        """Handle feedback request."""
        agent.feedback(
            response_id=params["response_id"],
            signal=params["signal"],
            correction=params.get("correction"),
        )
        return {"success": True}
