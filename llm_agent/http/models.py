"""HTTP endpoint models and IPC message types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


# === HTTP Request/Response Models ===


class CompleteRequest(BaseModel):
    """Request body for /v1/complete endpoint."""

    query: str = Field(..., description="User query to complete")
    system_prompt: str | None = Field(None, description="Optional system prompt override")


class CompleteResponse(BaseModel):
    """Response body for /v1/complete endpoint."""

    id: str = Field(..., description="Response ID (use for feedback)")
    content: str = Field(..., description="LLM response content")
    model: str = Field(..., description="Model used for completion")
    tokens_used: int = Field(..., description="Total tokens used")


class RememberRequest(BaseModel):
    """Request body for /v1/remember endpoint."""

    fact: str = Field(..., description="Fact to store")
    category: str = Field("general", description="Category for organization")


class RememberResponse(BaseModel):
    """Response body for /v1/remember endpoint."""

    fact_id: int = Field(..., description="ID of stored fact")


class ForgetResponse(BaseModel):
    """Response body for /v1/facts/{id} DELETE endpoint."""

    success: bool = Field(..., description="Whether deletion succeeded")


class RecallRequest(BaseModel):
    """Request body for /v1/recall endpoint."""

    query: str = Field(..., description="Query for semantic search")
    top_k: int | None = Field(None, description="Max results to return")
    min_similarity: float | None = Field(None, description="Minimum similarity threshold")
    categories: list[str] | None = Field(None, description="Filter to these categories")


class ScoredFactResponse(BaseModel):
    """A fact with its similarity score."""

    id: int = Field(..., description="Fact ID")
    content: str = Field(..., description="Fact content")
    category: str = Field(..., description="Fact category")
    similarity: float = Field(..., description="Similarity score")


class RecallResponse(BaseModel):
    """Response body for /v1/recall endpoint."""

    facts: list[ScoredFactResponse] = Field(..., description="Matching facts")


class FeedbackRequest(BaseModel):
    """Request body for /v1/feedback endpoint."""

    response_id: str = Field(..., description="ID from CompleteResponse")
    signal: str = Field(..., description="'positive' or 'negative'")
    correction: str | None = Field(None, description="Corrected response if negative")


class FeedbackResponse(BaseModel):
    """Response body for /v1/feedback endpoint."""

    success: bool = Field(..., description="Whether feedback was recorded")


class HealthResponse(BaseModel):
    """Response body for /v1/health endpoint."""

    status: str = Field(..., description="Server status")
    agent_name: str = Field(..., description="Name of the agent")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional detail")


# === IPC Message Types ===


@dataclass
class AgentRequest:
    """Request message sent from HTTP handler to main process.

    The `id` field is required for response routing.
    """

    id: str
    method: str  # "complete", "remember", "forget", "recall", "feedback"
    params: dict[str, Any]


@dataclass
class AgentResponse:
    """Response message sent from main process to HTTP handler.

    The `id` field must match the request's id for routing.
    """

    id: str
    success: bool
    result: dict[str, Any] | None
    error: str | None = None
