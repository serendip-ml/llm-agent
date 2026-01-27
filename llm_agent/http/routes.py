"""FastAPI routes for agent HTTP API."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path, Request

from llm_agent.http.models import (
    AgentRequest,
    AgentResponse,
    CompleteRequest,
    CompleteResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForgetResponse,
    HealthResponse,
    RecallRequest,
    RecallResponse,
    RememberRequest,
    RememberResponse,
    ScoredFactResponse,
)


if TYPE_CHECKING:
    from appinfra.app.fastapi import IPCChannel


router = APIRouter(prefix="/v1", tags=["agent"])


async def _submit_request(
    ipc: IPCChannel,
    method: str,
    params: dict[str, object],
    timeout: float | None = None,
) -> AgentResponse:
    """Submit a request to the main process and wait for response."""
    request_id = str(uuid4())
    request = AgentRequest(id=request_id, method=method, params=params)
    response: AgentResponse = await ipc.submit(request_id, request, timeout=timeout)
    return response


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check server status and get agent name",
)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    ipc: IPCChannel = request.app.state.ipc_channel
    response = await _submit_request(ipc, "health", {})

    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)

    return HealthResponse(
        status=response.result["status"],  # type: ignore[index]
        agent_name=response.result["agent_name"],  # type: ignore[index]
    )


@router.post(
    "/complete",
    response_model=CompleteResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Generate completion",
    description="Generate a response from the agent with optional system prompt",
)
async def complete(body: CompleteRequest, request: Request) -> CompleteResponse:
    """Generate a completion from the agent."""
    ipc: IPCChannel = request.app.state.ipc_channel

    params: dict[str, object] = {"query": body.query}
    if body.system_prompt is not None:
        params["system_prompt"] = body.system_prompt

    response = await _submit_request(ipc, "complete", params)

    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)

    result = response.result
    assert result is not None
    return CompleteResponse(
        id=result["id"],
        content=result["content"],
        model=result["model"],
        tokens_used=result["tokens_used"],
    )


@router.post(
    "/remember",
    response_model=RememberResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Store a fact",
    description="Store a fact about the user for future context",
)
async def remember(body: RememberRequest, request: Request) -> RememberResponse:
    """Store a fact in agent memory."""
    ipc: IPCChannel = request.app.state.ipc_channel

    response = await _submit_request(
        ipc, "remember", {"fact": body.fact, "category": body.category}
    )

    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)

    result = response.result
    assert result is not None
    return RememberResponse(fact_id=result["fact_id"])


@router.delete(
    "/facts/{fact_id}",
    response_model=ForgetResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Delete a fact",
    description="Remove a stored fact by its ID",
)
async def forget(
    request: Request,
    fact_id: int = Path(..., description="ID of the fact to delete"),
) -> ForgetResponse:
    """Delete a fact from agent memory."""
    ipc: IPCChannel = request.app.state.ipc_channel

    response = await _submit_request(ipc, "forget", {"fact_id": fact_id})

    if not response.success:
        if response.error and "not found" in response.error.lower():
            raise HTTPException(status_code=404, detail=response.error)
        raise HTTPException(status_code=500, detail=response.error)

    return ForgetResponse(success=True)


@router.post(
    "/recall",
    response_model=RecallResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Search facts",
    description="Search stored facts by semantic similarity",
)
async def recall(body: RecallRequest, request: Request) -> RecallResponse:
    """Search facts by semantic similarity."""
    ipc: IPCChannel = request.app.state.ipc_channel

    params: dict[str, object] = {"query": body.query}
    if body.top_k is not None:
        params["top_k"] = body.top_k
    if body.min_similarity is not None:
        params["min_similarity"] = body.min_similarity
    if body.categories is not None:
        params["categories"] = body.categories

    response = await _submit_request(ipc, "recall", params)

    if not response.success:
        if response.error and "embedder" in response.error.lower():
            raise HTTPException(status_code=400, detail=response.error)
        raise HTTPException(status_code=500, detail=response.error)

    result = response.result
    assert result is not None
    facts = [
        ScoredFactResponse(
            id=f["id"],
            content=f["content"],
            category=f["category"],
            similarity=f["similarity"],
        )
        for f in result["facts"]
    ]
    return RecallResponse(facts=facts)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Provide feedback",
    description="Provide feedback on a previous response",
)
async def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Record feedback on a response."""
    ipc: IPCChannel = request.app.state.ipc_channel

    if body.signal not in ("positive", "negative"):
        raise HTTPException(
            status_code=400,
            detail=f"signal must be 'positive' or 'negative', got '{body.signal}'",
        )

    params: dict[str, object] = {"response_id": body.response_id, "signal": body.signal}
    if body.correction is not None:
        params["correction"] = body.correction

    response = await _submit_request(ipc, "feedback", params)

    if not response.success:
        if response.error and "unknown response_id" in response.error.lower():
            raise HTTPException(status_code=404, detail=response.error)
        raise HTTPException(status_code=400, detail=response.error)

    return FeedbackResponse(success=True)
