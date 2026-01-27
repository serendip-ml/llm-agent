"""Tests for HTTP trait."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agent import CompletionResult, HTTPConfig, HTTPTrait
from llm_agent.http.models import AgentRequest, AgentResponse


pytestmark = pytest.mark.unit


class TestHTTPConfig:
    """Tests for HTTPConfig."""

    def test_defaults(self):
        config = HTTPConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.title == "Agent API"
        assert config.response_timeout == 60.0

    def test_custom_values(self):
        config = HTTPConfig(
            host="0.0.0.0",
            port=9000,
            title="My Agent",
            response_timeout=120.0,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.title == "My Agent"
        assert config.response_timeout == 120.0


class TestHTTPTrait:
    """Tests for HTTPTrait."""

    def test_init_with_default_config(self):
        trait = HTTPTrait()

        assert trait.config.host == "127.0.0.1"
        assert trait.config.port == 8080

    def test_init_with_custom_config(self):
        config = HTTPConfig(port=9000)
        trait = HTTPTrait(config=config)

        assert trait.config.port == 9000

    def test_attach(self):
        trait = HTTPTrait()
        mock_agent = MagicMock()

        trait.attach(mock_agent)

        assert trait._agent == mock_agent

    def test_is_running_false_when_not_started(self):
        trait = HTTPTrait()

        assert not trait.is_running

    def test_on_start_without_attach_raises(self):
        trait = HTTPTrait()

        with pytest.raises(RuntimeError, match="not attached"):
            trait.on_start()


class TestHTTPTraitDispatch:
    """Tests for HTTPTrait._dispatch method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with all required methods."""
        agent = MagicMock()
        agent.name = "test-agent"
        agent.complete.return_value = CompletionResult(
            id="resp-123",
            content="Hello!",
            model="gpt-4",
            tokens_used=25,
            latency_ms=100,
        )
        agent.remember.return_value = 42
        agent.recall.return_value = []
        return agent

    @pytest.fixture
    def trait(self, mock_agent):
        """Create an attached HTTPTrait."""
        trait = HTTPTrait()
        trait.attach(mock_agent)
        return trait

    def test_dispatch_health(self, trait, mock_agent):
        request = AgentRequest(id="req-1", method="health", params={})

        response = trait._dispatch(request)

        assert response.id == "req-1"
        assert response.success is True
        assert response.result == {"status": "ok", "agent_name": "test-agent"}
        assert response.error is None

    def test_dispatch_complete(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="complete",
            params={"query": "Hello"},
        )

        response = trait._dispatch(request)

        assert response.id == "req-1"
        assert response.success is True
        assert response.result["id"] == "resp-123"
        assert response.result["content"] == "Hello!"
        assert response.result["model"] == "gpt-4"
        assert response.result["tokens_used"] == 25
        mock_agent.complete.assert_called_once_with(query="Hello", system_prompt=None)

    def test_dispatch_complete_with_system_prompt(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="complete",
            params={"query": "Hello", "system_prompt": "Be brief."},
        )

        trait._dispatch(request)

        mock_agent.complete.assert_called_once_with(query="Hello", system_prompt="Be brief.")

    def test_dispatch_remember(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="remember",
            params={"fact": "User likes Python", "category": "preferences"},
        )

        response = trait._dispatch(request)

        assert response.success is True
        assert response.result == {"fact_id": 42}
        mock_agent.remember.assert_called_once_with(
            fact="User likes Python", category="preferences"
        )

    def test_dispatch_remember_default_category(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="remember",
            params={"fact": "Some fact"},
        )

        trait._dispatch(request)

        mock_agent.remember.assert_called_once_with(fact="Some fact", category="general")

    def test_dispatch_forget(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="forget",
            params={"fact_id": 42},
        )

        response = trait._dispatch(request)

        assert response.success is True
        assert response.result == {"success": True}
        mock_agent.forget.assert_called_once_with(fact_id=42)

    def test_dispatch_recall(self, trait, mock_agent):
        # Setup mock scored facts
        mock_fact = MagicMock()
        mock_fact.id = 1
        mock_fact.content = "User likes Python"
        mock_fact.category = "preferences"
        mock_scored = MagicMock()
        mock_scored.fact = mock_fact
        mock_scored.similarity = 0.85
        mock_agent.recall.return_value = [mock_scored]

        request = AgentRequest(
            id="req-1",
            method="recall",
            params={"query": "programming"},
        )

        response = trait._dispatch(request)

        assert response.success is True
        assert len(response.result["facts"]) == 1
        assert response.result["facts"][0]["content"] == "User likes Python"
        assert response.result["facts"][0]["similarity"] == 0.85
        mock_agent.recall.assert_called_once_with(
            query="programming",
            top_k=None,
            min_similarity=None,
            categories=None,
        )

    def test_dispatch_recall_with_params(self, trait, mock_agent):
        mock_agent.recall.return_value = []

        request = AgentRequest(
            id="req-1",
            method="recall",
            params={
                "query": "programming",
                "top_k": 10,
                "min_similarity": 0.5,
                "categories": ["preferences"],
            },
        )

        trait._dispatch(request)

        mock_agent.recall.assert_called_once_with(
            query="programming",
            top_k=10,
            min_similarity=0.5,
            categories=["preferences"],
        )

    def test_dispatch_feedback(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="feedback",
            params={"response_id": "resp-123", "signal": "positive"},
        )

        response = trait._dispatch(request)

        assert response.success is True
        assert response.result == {"success": True}
        mock_agent.feedback.assert_called_once_with(
            response_id="resp-123",
            signal="positive",
            correction=None,
        )

    def test_dispatch_feedback_with_correction(self, trait, mock_agent):
        request = AgentRequest(
            id="req-1",
            method="feedback",
            params={
                "response_id": "resp-123",
                "signal": "negative",
                "correction": "Better answer",
            },
        )

        trait._dispatch(request)

        mock_agent.feedback.assert_called_once_with(
            response_id="resp-123",
            signal="negative",
            correction="Better answer",
        )

    def test_dispatch_unknown_method(self, trait):
        request = AgentRequest(
            id="req-1",
            method="unknown",
            params={},
        )

        response = trait._dispatch(request)

        assert response.id == "req-1"
        assert response.success is False
        assert response.result is None
        assert "Unknown method" in response.error

    def test_dispatch_handles_exception(self, trait, mock_agent):
        mock_agent.complete.side_effect = ValueError("Something went wrong")

        request = AgentRequest(
            id="req-1",
            method="complete",
            params={"query": "Hello"},
        )

        response = trait._dispatch(request)

        assert response.id == "req-1"
        assert response.success is False
        assert response.result is None
        assert "Something went wrong" in response.error

    def test_dispatch_handles_dict_request(self, trait, mock_agent):
        """Verify dispatch handles dict form (from queue serialization)."""
        request = {
            "id": "req-1",
            "method": "health",
            "params": {},
        }

        response = trait._dispatch(request)

        assert response.success is True
        assert response.result["status"] == "ok"


class TestHTTPTraitLifecycle:
    """Tests for HTTPTrait on_start/on_stop lifecycle."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test-agent"
        return agent

    @patch("llm_agent.traits.http.mp.Queue")
    @patch("appinfra.app.fastapi.ServerBuilder")
    def test_on_start_creates_server(self, mock_builder_cls, mock_queue, mock_agent):
        """Verify on_start creates server with correct configuration."""
        # Setup mocks
        mock_queue.return_value = MagicMock()
        mock_server = MagicMock()
        mock_server.start_subprocess.return_value = MagicMock()

        # Chain the builder methods
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.with_host.return_value = mock_builder
        mock_builder.with_port.return_value = mock_builder
        mock_builder.with_title.return_value = mock_builder
        mock_builder.subprocess = MagicMock()
        mock_builder.subprocess.with_ipc.return_value = mock_builder.subprocess
        mock_builder.subprocess.with_response_timeout.return_value = mock_builder.subprocess
        mock_builder.subprocess.with_auto_restart.return_value = mock_builder.subprocess
        mock_builder.subprocess.done.return_value = mock_builder
        mock_builder.routes = MagicMock()
        mock_builder.routes.with_router.return_value = mock_builder.routes
        mock_builder.routes.done.return_value = mock_builder
        mock_builder.build.return_value = mock_server

        config = HTTPConfig(host="0.0.0.0", port=9000, title="Test API")
        trait = HTTPTrait(config=config)
        trait.attach(mock_agent)

        trait.on_start()

        # Verify builder was called with correct config
        mock_builder_cls.assert_called_once_with("Test API")
        mock_builder.with_host.assert_called_once_with("0.0.0.0")
        mock_builder.with_port.assert_called_once_with(9000)

        # Verify server was started
        mock_server.start_subprocess.assert_called_once()

        # Cleanup
        trait.on_stop()

    @patch("llm_agent.traits.http.mp.Queue")
    @patch("appinfra.app.fastapi.ServerBuilder")
    def test_on_stop_terminates_process(self, mock_builder_cls, mock_queue, mock_agent):
        """Verify on_stop terminates subprocess and cleans up."""
        mock_queue.return_value = MagicMock()
        mock_process = MagicMock()
        mock_server = MagicMock()
        mock_server.start_subprocess.return_value = mock_process

        # Setup builder chain
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.with_host.return_value = mock_builder
        mock_builder.with_port.return_value = mock_builder
        mock_builder.with_title.return_value = mock_builder
        mock_builder.subprocess = MagicMock()
        mock_builder.subprocess.with_ipc.return_value = mock_builder.subprocess
        mock_builder.subprocess.with_response_timeout.return_value = mock_builder.subprocess
        mock_builder.subprocess.with_auto_restart.return_value = mock_builder.subprocess
        mock_builder.subprocess.done.return_value = mock_builder
        mock_builder.routes = MagicMock()
        mock_builder.routes.with_router.return_value = mock_builder.routes
        mock_builder.routes.done.return_value = mock_builder
        mock_builder.build.return_value = mock_server

        trait = HTTPTrait()
        trait.attach(mock_agent)
        trait.on_start()

        trait.on_stop()

        mock_process.terminate.assert_called_once()
        mock_process.join.assert_called_once()


class TestHTTPModels:
    """Tests for HTTP request/response models."""

    def test_agent_request_dataclass(self):
        request = AgentRequest(id="req-1", method="complete", params={"query": "Hello"})

        assert request.id == "req-1"
        assert request.method == "complete"
        assert request.params == {"query": "Hello"}

    def test_agent_response_success(self):
        response = AgentResponse(id="req-1", success=True, result={"content": "Hi"})

        assert response.id == "req-1"
        assert response.success is True
        assert response.result == {"content": "Hi"}
        assert response.error is None

    def test_agent_response_error(self):
        response = AgentResponse(id="req-1", success=False, result=None, error="Failed")

        assert response.success is False
        assert response.result is None
        assert response.error == "Failed"
