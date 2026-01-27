"""HTTP API models and routes for llm-agent."""

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
from llm_agent.http.routes import router


__all__ = [
    # IPC types
    "AgentRequest",
    "AgentResponse",
    # HTTP models
    "CompleteRequest",
    "CompleteResponse",
    "ErrorResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "ForgetResponse",
    "HealthResponse",
    "RecallRequest",
    "RecallResponse",
    "RememberRequest",
    "RememberResponse",
    "ScoredFactResponse",
    # Router
    "router",
]
