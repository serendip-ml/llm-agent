"""HTTP server trait for agents."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from fastapi import APIRouter

from llm_agent.runtime.server.protocol.base import Request, Response


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent
    from llm_agent.runtime.server.http import HTTPServer


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
    """HTTP server trait with request handling.

    Runs a FastAPI server in a subprocess and communicates via IPC queues.
    Handles standard agent protocol requests (complete, remember, recall, etc.).

    Architecture:
        HTTP Request -> FastAPI subprocess -> request_queue -> IPC thread -> handle_request()
        HTTP Response <- FastAPI subprocess <- response_queue <- handle_request()

    The trait handles protocol requests by calling methods on the attached agent:
    - /health -> agent.name
    - /complete -> agent.complete()
    - /remember -> agent.remember()
    - /forget -> agent.forget()
    - /recall -> agent.recall()
    - /feedback -> agent.feedback()

    Example:
        from appinfra.log import Logger

        lg = Logger.create("agent")
        agent = ConversationalAgent(lg, config)
        agent.add_trait(HTTPTrait(lg, HTTPConfig(port=8080)))
        agent.start()

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

        from llm_agent.runtime.server.http import HTTPServer, HTTPServerConfig

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
        from llm_agent.runtime.server.routes import create_agent_routes

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
        if self._ipc_thread.is_alive():
            self.lg.warning("IPC thread did not stop within timeout, may be orphaned")
        self._ipc_thread = None
        self.lg.debug("IPC processing thread stopped")

    def _ipc_loop(self, poll_timeout: float = 0.1) -> None:
        """Background loop processing IPC requests."""
        assert self._server is not None

        while not self._ipc_shutdown.is_set():
            request = None
            try:
                request = self._server.request_queue.get(timeout=poll_timeout)
                response = self.handle_request(request)
                if response is not None:
                    self._server.response_queue.put(response)
            except Empty:
                pass
            except Exception:
                self.lg.exception("IPC processing error")
                if request is not None:
                    error_resp = Response(
                        id=getattr(request, "id", "unknown"),
                        success=False,
                        error="Internal server error",
                    )
                    self._server.response_queue.put(error_resp)

    # -------------------------------------------------------------------------
    # Request Handling
    # -------------------------------------------------------------------------

    def handle_request(self, request: Request) -> Response:
        """Handle an HTTP protocol request.

        Dispatches to the appropriate handler based on message type.

        Args:
            request: Protocol request message.

        Returns:
            Protocol response message.
        """
        from llm_agent.runtime.server.protocol.v1 import (
            CompleteRequest,
            FeedbackRequest,
            ForgetRequest,
            HealthRequest,
            RecallRequest,
            RememberRequest,
        )

        handlers: dict[str, Callable[[Request], Response]] = {
            HealthRequest.message_type: self._handle_health,
            CompleteRequest.message_type: self._handle_complete,
            RememberRequest.message_type: self._handle_remember,
            ForgetRequest.message_type: self._handle_forget,
            RecallRequest.message_type: self._handle_recall,
            FeedbackRequest.message_type: self._handle_feedback,
        }

        handler = handlers.get(request.message_type)
        if handler is None:
            return Response(
                id=request.id,
                success=False,
                error=f"Unknown message type: {request.message_type}",
            )

        return handler(request)

    def _handle_health(self, request: Request) -> Response:
        """Handle health check request."""
        from llm_agent.runtime.server.protocol.v1 import HealthResponse

        assert self._agent is not None
        return HealthResponse(id=request.id, status="ok", agent_name=self._agent.name)

    def _handle_complete(self, request: Request) -> Response:
        """Handle completion request."""
        from llm_agent.runtime.server.protocol.v1 import CompleteRequest, CompleteResponse

        assert self._agent is not None
        complete_fn: Callable[..., Any] | None = getattr(self._agent, "complete", None)
        if complete_fn is None:
            return self._complete_error(request.id, "Agent does not support complete()")

        req = (
            request
            if isinstance(request, CompleteRequest)
            else CompleteRequest(**request.model_dump())
        )
        try:
            result = complete_fn(query=req.query, system_prompt=req.system_prompt)
            return CompleteResponse(
                id=req.id,
                response_id=result.id,
                content=result.content,
                model=result.model,
                tokens_used=result.tokens_used,
            )
        except Exception as e:
            self.lg.warning("complete request failed", extra={"exception": e})
            return self._complete_error(req.id, str(e))

    def _complete_error(self, request_id: str, error: str) -> Response:
        """Build a CompleteResponse error."""
        from llm_agent.runtime.server.protocol.v1 import CompleteResponse

        return CompleteResponse(
            id=request_id,
            success=False,
            error=error,
            response_id="",
            content="",
            model="",
            tokens_used=0,
        )

    def _handle_remember(self, request: Request) -> Response:
        """Handle remember request."""
        from llm_agent.runtime.server.protocol.v1 import RememberRequest, RememberResponse

        assert self._agent is not None
        remember_fn: Callable[..., int] | None = getattr(self._agent, "remember", None)
        if remember_fn is None:
            return RememberResponse(
                id=request.id, success=False, error="Agent does not support remember()", fact_id=-1
            )

        req = (
            request
            if isinstance(request, RememberRequest)
            else RememberRequest(**request.model_dump())
        )
        try:
            fact_id = remember_fn(fact=req.fact, category=req.category)
            return RememberResponse(id=req.id, fact_id=fact_id)
        except Exception as e:
            self.lg.warning("remember request failed", extra={"exception": e})
            return RememberResponse(id=req.id, success=False, error=str(e), fact_id=-1)

    def _handle_forget(self, request: Request) -> Response:
        """Handle forget request."""
        from llm_agent.runtime.server.protocol.v1 import ForgetRequest, ForgetResponse

        assert self._agent is not None
        forget_fn: Callable[..., None] | None = getattr(self._agent, "forget", None)
        if forget_fn is None:
            return ForgetResponse(
                id=request.id, success=False, error="Agent does not support forget()"
            )

        req = (
            request if isinstance(request, ForgetRequest) else ForgetRequest(**request.model_dump())
        )
        try:
            forget_fn(fact_id=req.fact_id)
            return ForgetResponse(id=req.id)
        except Exception as e:
            self.lg.warning("forget request failed", extra={"exception": e})
            return ForgetResponse(id=req.id, success=False, error=str(e))

    def _handle_recall(self, request: Request) -> Response:
        """Handle recall request."""
        from llm_agent.runtime.server.protocol.v1 import RecallRequest, RecallResponse

        assert self._agent is not None
        recall_fn: Callable[..., Any] | None = getattr(self._agent, "recall", None)
        if recall_fn is None:
            return RecallResponse(
                id=request.id, success=False, error="Agent does not support recall()"
            )

        req = (
            request if isinstance(request, RecallRequest) else RecallRequest(**request.model_dump())
        )
        try:
            scored_facts = recall_fn(
                query=req.query,
                top_k=req.top_k,
                min_similarity=req.min_similarity,
                categories=req.categories,
            )
            return RecallResponse(id=req.id, facts=self._serialize_facts(scored_facts))
        except Exception as e:
            self.lg.warning("recall request failed", extra={"exception": e})
            return RecallResponse(id=req.id, success=False, error=str(e))

    def _serialize_facts(self, scored_facts: Any) -> list[dict[str, Any]]:
        """Serialize scored facts for recall response."""
        return [
            {
                "fact_id": sf.fact.id,
                "content": sf.fact.content,
                "category": sf.fact.category,
                "similarity": sf.similarity,
            }
            for sf in scored_facts
        ]

    def _handle_feedback(self, request: Request) -> Response:
        """Handle feedback request."""
        from llm_agent.runtime.server.protocol.v1 import FeedbackRequest, FeedbackResponse

        assert self._agent is not None
        feedback_fn: Callable[..., None] | None = getattr(self._agent, "feedback", None)
        if feedback_fn is None:
            return FeedbackResponse(
                id=request.id, success=False, error="Agent does not support feedback()"
            )

        req = (
            request
            if isinstance(request, FeedbackRequest)
            else FeedbackRequest(**request.model_dump())
        )
        try:
            feedback_fn(
                response_id=req.response_id,
                signal=req.signal,
                correction=req.correction,
            )
            return FeedbackResponse(id=req.id)
        except Exception as e:
            self.lg.warning("feedback request failed", extra={"exception": e})
            return FeedbackResponse(id=req.id, success=False, error=str(e))
