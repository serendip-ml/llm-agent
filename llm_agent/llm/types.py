"""LLM message and result types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionResult(BaseModel):
    """Result from LLM completion."""

    id: str = Field(description="Unique response ID for feedback tracking")
    content: str = Field(description="Generated text")
    model: str = Field(description="Model used")
    tokens_used: int = Field(description="Total tokens (prompt + completion)")
    latency_ms: int = Field(description="Response time in milliseconds")
