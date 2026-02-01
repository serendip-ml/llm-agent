"""Agent management routes.

REST API for managing agent lifecycle via Core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


if TYPE_CHECKING:
    from llm_agent.runtime import AgentInfo, Core


# ============================================================================
# Request/Response Models
# ============================================================================


class AgentInfoResponse(BaseModel):
    """Agent information response."""

    name: str
    status: str
    cycle_count: int
    last_run: str | None
    error: str | None
    schedule_interval: int | None


class AgentListResponse(BaseModel):
    """List of agents response."""

    agents: list[AgentInfoResponse]


class AskRequest(BaseModel):
    """Ask question request."""

    question: str


class AskResponse(BaseModel):
    """Ask question response."""

    response: str


class FeedbackRequest(BaseModel):
    """Feedback request."""

    message: str


class FeedbackResponse(BaseModel):
    """Feedback response."""

    success: bool


class InsightResponse(BaseModel):
    """Single insight from agent."""

    success: bool
    content: str
    parsed: dict[str, Any] | None
    iterations: int


class InsightsResponse(BaseModel):
    """List of insights response."""

    insights: list[InsightResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_count: int


# ============================================================================
# Helpers
# ============================================================================


def _to_response(info: AgentInfo) -> AgentInfoResponse:
    """Convert AgentInfo to response model."""
    return AgentInfoResponse(
        name=info.name,
        status=info.status,
        cycle_count=info.cycle_count,
        last_run=info.last_run.isoformat() if info.last_run else None,
        error=info.error,
        schedule_interval=info.schedule_interval,
    )


def _get_core(request: Request) -> Core:
    """Get Core from request app state."""
    return cast("Core", request.app.state.core)


def _agent_not_found(name: str) -> HTTPException:
    """Create 404 exception for agent not found."""
    return HTTPException(status_code=404, detail=f"Agent not found: {name}")


# ============================================================================
# Route Handlers
# ============================================================================


def _health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    core = _get_core(request)
    return HealthResponse(status="ok", agent_count=len(core.registry.list_agents()))


def _list_agents(request: Request) -> AgentListResponse:
    """List all registered agents."""
    core = _get_core(request)
    agents = core.registry.list_agents()
    return AgentListResponse(agents=[_to_response(a) for a in agents])


def _get_agent(name: str, request: Request) -> AgentInfoResponse:
    """Get agent information by name."""
    core = _get_core(request)
    handle = core.registry.get(name)
    if handle is None:
        raise _agent_not_found(name)
    from llm_agent.runtime.handle import AgentInfo

    return _to_response(AgentInfo.from_handle(handle))


def _start_agent(name: str, request: Request) -> AgentInfoResponse:
    """Start an agent process."""
    core = _get_core(request)
    try:
        info = core.start(name)
        return _to_response(info)
    except KeyError:
        raise _agent_not_found(name) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def _stop_agent(name: str, request: Request) -> AgentInfoResponse:
    """Stop an agent process."""
    core = _get_core(request)
    try:
        info = core.stop(name)
        return _to_response(info)
    except KeyError:
        raise _agent_not_found(name) from None


def _ask_agent(name: str, body: AskRequest, request: Request) -> AskResponse:
    """Ask an agent a question."""
    core = _get_core(request)
    try:
        response = core.ask(name, body.question)
        return AskResponse(response=response)
    except KeyError:
        raise _agent_not_found(name) from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def _feedback_agent(name: str, body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Send feedback to an agent."""
    core = _get_core(request)
    try:
        core.feedback(name, body.message)
        return FeedbackResponse(success=True)
    except KeyError:
        raise _agent_not_found(name) from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def _get_insights(name: str, request: Request, limit: int = 10) -> InsightsResponse:
    """Get recent insights from an agent."""
    core = _get_core(request)
    try:
        insights = core.get_insights(name, limit)
        return InsightsResponse(insights=[InsightResponse(**i) for i in insights])
    except KeyError:
        raise _agent_not_found(name) from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


# ============================================================================
# Route Factory
# ============================================================================


def create_management_routes() -> APIRouter:
    """Create a fresh API router for agent management endpoints.

    Returns a new router instance each time, allowing safe mounting
    to multiple apps or test isolation.

    Routes delegate to Core stored in app.state.
    """
    router = APIRouter(tags=["Agent Management"])

    router.add_api_route("/health", _health, methods=["GET"], response_model=HealthResponse)
    router.add_api_route("/agents", _list_agents, methods=["GET"], response_model=AgentListResponse)
    router.add_api_route(
        "/agents/{name}", _get_agent, methods=["GET"], response_model=AgentInfoResponse
    )
    router.add_api_route(
        "/agents/{name}/start", _start_agent, methods=["POST"], response_model=AgentInfoResponse
    )
    router.add_api_route(
        "/agents/{name}/stop", _stop_agent, methods=["POST"], response_model=AgentInfoResponse
    )
    router.add_api_route(
        "/agents/{name}/ask", _ask_agent, methods=["POST"], response_model=AskResponse
    )
    router.add_api_route(
        "/agents/{name}/feedback",
        _feedback_agent,
        methods=["POST"],
        response_model=FeedbackResponse,
    )
    router.add_api_route(
        "/agents/{name}/insights", _get_insights, methods=["GET"], response_model=InsightsResponse
    )

    return router
