"""Tests for management routes."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_agent.runtime import AgentHandle, AgentInfo, AgentState
from llm_agent.runtime.server.management import create_management_routes


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_core():
    """Create mock core with mock registry."""
    core = MagicMock()
    core.registry = MagicMock()
    return core


@pytest.fixture
def app(mock_core):
    """Create FastAPI app with routes."""
    app = FastAPI()
    app.state.core = mock_core
    app.include_router(create_management_routes())
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthRoute:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client, mock_core):
        """Health endpoint returns ok status."""
        mock_core.registry.list_agents.return_value = []

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent_count": 0}

    def test_health_includes_agent_count(self, client, mock_core):
        """Health endpoint includes agent count."""
        mock_core.registry.list_agents.return_value = [MagicMock(), MagicMock(), MagicMock()]

        response = client.get("/health")

        assert response.json()["agent_count"] == 3


class TestListAgentsRoute:
    """Tests for GET /agents endpoint."""

    def test_list_empty(self, client, mock_core):
        """List returns empty when no agents."""
        mock_core.registry.list_agents.return_value = []

        response = client.get("/agents")

        assert response.status_code == 200
        assert response.json() == {"agents": []}

    def test_list_returns_agents(self, client, mock_core):
        """List returns all agents."""
        mock_core.registry.list_agents.return_value = [
            AgentInfo(name="agent1", status="idle"),
            AgentInfo(name="agent2", status="running", cycle_count=5),
        ]

        response = client.get("/agents")

        data = response.json()
        assert len(data["agents"]) == 2
        assert data["agents"][0]["name"] == "agent1"
        assert data["agents"][1]["name"] == "agent2"


class TestGetAgentRoute:
    """Tests for GET /agents/{name} endpoint."""

    def test_get_existing_agent(self, client, mock_core):
        """Get returns agent info."""
        mock_core.registry.get.return_value = AgentHandle(
            name="test-agent",
            config={},
            state=AgentState.RUNNING,
            cycle_count=10,
        )

        response = client.get("/agents/test-agent")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-agent"
        assert data["status"] == "running"

    def test_get_nonexistent_agent(self, client, mock_core):
        """Get returns 404 for nonexistent agent."""
        mock_core.registry.get.return_value = None

        response = client.get("/agents/nonexistent")

        assert response.status_code == 404


class TestStartAgentRoute:
    """Tests for POST /agents/{name}/start endpoint."""

    def test_start_agent(self, client, mock_core):
        """Start transitions agent to running."""
        mock_core.start.return_value = AgentInfo(name="test", status="running")

        response = client.post("/agents/test/start")

        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_start_nonexistent(self, client, mock_core):
        """Start returns 404 for nonexistent agent."""
        mock_core.start.side_effect = KeyError("Agent not found")

        response = client.post("/agents/nonexistent/start")

        assert response.status_code == 404

    def test_start_invalid_state(self, client, mock_core):
        """Start returns 400 for invalid state transition."""
        mock_core.start.side_effect = ValueError("Invalid transition")

        response = client.post("/agents/test/start")

        assert response.status_code == 400


class TestStopAgentRoute:
    """Tests for POST /agents/{name}/stop endpoint."""

    def test_stop_agent(self, client, mock_core):
        """Stop transitions agent to stopped."""
        mock_core.stop.return_value = AgentInfo(name="test", status="stopped")

        response = client.post("/agents/test/stop")

        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

    def test_stop_nonexistent(self, client, mock_core):
        """Stop returns 404 for nonexistent agent."""
        mock_core.stop.side_effect = KeyError("Agent not found")

        response = client.post("/agents/nonexistent/stop")

        assert response.status_code == 404


class TestAskAgentRoute:
    """Tests for POST /agents/{name}/ask endpoint."""

    def test_ask_agent(self, client, mock_core):
        """Ask returns agent response."""
        mock_core.ask.return_value = "Agent response"

        response = client.post("/agents/test/ask", json={"question": "What time is it?"})

        assert response.status_code == 200
        assert response.json()["response"] == "Agent response"
        mock_core.ask.assert_called_once_with("test", "What time is it?")

    def test_ask_nonexistent(self, client, mock_core):
        """Ask returns 404 for nonexistent agent."""
        mock_core.ask.side_effect = KeyError("Agent not found")

        response = client.post("/agents/nonexistent/ask", json={"question": "test"})

        assert response.status_code == 404

    def test_ask_not_running(self, client, mock_core):
        """Ask returns 400 for non-running agent."""
        mock_core.ask.side_effect = RuntimeError("Agent not running")

        response = client.post("/agents/test/ask", json={"question": "test"})

        assert response.status_code == 400


class TestFeedbackRoute:
    """Tests for POST /agents/{name}/feedback endpoint."""

    def test_feedback(self, client, mock_core):
        """Feedback returns success."""
        mock_core.feedback.return_value = None

        response = client.post("/agents/test/feedback", json={"message": "Good job!"})

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_feedback_nonexistent(self, client, mock_core):
        """Feedback returns 404 for nonexistent agent."""
        mock_core.feedback.side_effect = KeyError("Agent not found")

        response = client.post("/agents/nonexistent/feedback", json={"message": "test"})

        assert response.status_code == 404


class TestGetInsightsRoute:
    """Tests for GET /agents/{name}/insights endpoint."""

    def test_get_insights(self, client, mock_core):
        """Get insights returns list."""
        mock_core.get_insights.return_value = [
            {"success": True, "content": "Insight 1", "parsed": None, "iterations": 1},
            {"success": True, "content": "Insight 2", "parsed": {"key": "value"}, "iterations": 2},
        ]

        response = client.get("/agents/test/insights")

        assert response.status_code == 200
        data = response.json()
        assert len(data["insights"]) == 2

    def test_get_insights_with_limit(self, client, mock_core):
        """Get insights respects limit parameter."""
        mock_core.get_insights.return_value = []

        client.get("/agents/test/insights?limit=5")

        mock_core.get_insights.assert_called_once_with("test", 5)

    def test_get_insights_nonexistent(self, client, mock_core):
        """Get insights returns 404 for nonexistent agent."""
        mock_core.get_insights.side_effect = KeyError("Agent not found")

        response = client.get("/agents/nonexistent/insights")

        assert response.status_code == 404
