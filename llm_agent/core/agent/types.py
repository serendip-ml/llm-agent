"""Agent execution result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result from agent execution."""

    success: bool
    content: str
    iterations: int = 1
    tokens_used: int = 0
    latency_ms: int = 0
    trace_id: str = ""
