"""FastAPI application factory.

Creates the FastAPI app with routes configured for the agent registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from llm_agent.runtime.server.management import create_management_routes


if TYPE_CHECKING:
    from llm_agent.runtime.registry import AgentRegistry


def create_app(registry: AgentRegistry, title: str = "Agent Gateway") -> FastAPI:
    """Create FastAPI application with routes.

    Args:
        registry: Agent registry for managing agents.
        title: API title for OpenAPI docs.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description="Gateway for managing multiple LLM agents",
        version="1.0.0",
    )

    # Store registry in app state for route handlers
    app.state.registry = registry

    # Include management routes
    app.include_router(create_management_routes())

    return app
