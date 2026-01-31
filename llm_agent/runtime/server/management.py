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
# Router - defined at module level for idiomatic FastAPI
# ============================================================================

_router = APIRouter(tags=["Agent Management"])


@_router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    core = _get_core(request)
    return HealthResponse(status="ok", agent_count=len(core.registry.list_agents()))


@_router.get("/agents", response_model=AgentListResponse)
def list_agents(request: Request) -> AgentListResponse:
    """List all registered agents."""
    core = _get_core(request)
    agents = core.registry.list_agents()
    return AgentListResponse(agents=[_to_response(a) for a in agents])


@_router.get("/agents/{name}", response_model=AgentInfoResponse)
def get_agent(name: str, request: Request) -> AgentInfoResponse:
    """Get agent information by name."""
    core = _get_core(request)
    handle = core.registry.get(name)
    if handle is None:
        raise _agent_not_found(name)
    from llm_agent.runtime.handle import AgentInfo

    return _to_response(AgentInfo.from_handle(handle))


@_router.post("/agents/{name}/start", response_model=AgentInfoResponse)
def start_agent(name: str, request: Request) -> AgentInfoResponse:
    """Start an agent process."""
    core = _get_core(request)
    try:
        info = core.start(name)
        return _to_response(info)
    except KeyError:
        raise _agent_not_found(name) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@_router.post("/agents/{name}/stop", response_model=AgentInfoResponse)
def stop_agent(name: str, request: Request) -> AgentInfoResponse:
    """Stop an agent process."""
    core = _get_core(request)
    try:
        info = core.stop(name)
        return _to_response(info)
    except KeyError:
        raise _agent_not_found(name) from None


@_router.post("/agents/{name}/ask", response_model=AskResponse)
def ask_agent(name: str, body: AskRequest, request: Request) -> AskResponse:
    """Ask an agent a question."""
    core = _get_core(request)
    try:
        response = core.ask(name, body.question)
        return AskResponse(response=response)
    except KeyError:
        raise _agent_not_found(name) from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@_router.post("/agents/{name}/feedback", response_model=FeedbackResponse)
def feedback_agent(name: str, body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Send feedback to an agent."""
    core = _get_core(request)
    try:
        core.feedback(name, body.message)
        return FeedbackResponse(success=True)
    except KeyError:
        raise _agent_not_found(name) from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@_router.get("/agents/{name}/insights", response_model=InsightsResponse)
def get_insights(name: str, request: Request, limit: int = 10) -> InsightsResponse:
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
    """Create API router for agent management endpoints.

    Routes delegate to Core stored in app.state.
    """
    return _router
