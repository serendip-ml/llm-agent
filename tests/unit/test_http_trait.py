"""Tests for HTTP trait and server components."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agent import CompletionResult, HTTPConfig, HTTPTrait
from llm_agent.protocol.v1 import (
    CompleteRequest,
    CompleteResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForgetRequest,
    ForgetResponse,
    HealthRequest,
    HealthResponse,
    RecallRequest,
    RememberRequest,
    RememberResponse,
)
from llm_agent.server.http import HTTPServer, HTTPServerConfig


pytestmark = pytest.mark.unit


class TestHTTPConfig:
    """Tests for HTTPConfig."""

    def test_defaults(self):
        config = HTTPConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.title is None
        assert config.description is None

    def test_custom_values(self):
        config = HTTPConfig(
            host="0.0.0.0",
            port=9000,
            title="My Agent",
            description="My API",
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.title == "My Agent"
        assert config.description == "My API"


class TestHTTPServerConfig:
    """Tests for HTTPServerConfig."""

    def test_defaults(self):
        config = HTTPServerConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.title == "Agent API"
        assert config.auto_restart is True
        assert config.response_timeout == 60.0

    def test_custom_values(self):
        config = HTTPServerConfig(
            host="0.0.0.0",
            port=9000,
            title="Custom API",
            auto_restart=False,
            response_timeout=120.0,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.title == "Custom API"
        assert config.auto_restart is False
        assert config.response_timeout == 120.0


class TestHTTPServer:
    """Tests for HTTPServer."""

    def test_init(self):
        config = HTTPServerConfig(port=9000)
        router_factory = MagicMock()

        server = HTTPServer(config, router_factory)

        assert server.config == config
        assert server.request_queue is not None
        assert server.response_queue is not None
        assert not server.is_running


class TestHTTPTrait:
    """Tests for HTTPTrait."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_init_with_default_config(self, mock_logger):
        trait = HTTPTrait(lg=mock_logger)

        assert trait.config.host == "127.0.0.1"
        assert trait.config.port == 8080

    def test_init_with_custom_config(self, mock_logger):
        config = HTTPConfig(port=9000)
        trait = HTTPTrait(lg=mock_logger, config=config)

        assert trait.config.port == 9000

    def test_attach(self, mock_logger):
        trait = HTTPTrait(lg=mock_logger)
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"

        with patch("llm_agent.server.http.HTTPServer"):
            trait.attach(mock_agent)

        assert trait._agent == mock_agent
        assert trait._server is not None

    def test_url_property(self, mock_logger):
        trait = HTTPTrait(lg=mock_logger, config=HTTPConfig(host="localhost", port=8080))

        assert trait.url == "http://localhost:8080"

    def test_url_property_ephemeral_port(self, mock_logger):
        trait = HTTPTrait(lg=mock_logger, config=HTTPConfig(port=0))

        assert trait.url is None

    def test_is_running_false_when_not_started(self, mock_logger):
        trait = HTTPTrait(lg=mock_logger)

        assert not trait.is_running

    def test_on_start_without_attach_raises(self, mock_logger):
        trait = HTTPTrait(lg=mock_logger)

        with pytest.raises(RuntimeError, match="not attached"):
            trait.on_start()


class TestProtocolMessages:
    """Tests for protocol v1 messages."""

    def test_health_request(self):
        req = HealthRequest(id="req-1")
        assert req.id == "req-1"
        assert req.message_type == "health_request"

    def test_health_response(self):
        resp = HealthResponse(id="req-1", status="ok", agent_name="test")
        assert resp.id == "req-1"
        assert resp.status == "ok"
        assert resp.agent_name == "test"
        assert resp.success is True

    def test_complete_request(self):
        req = CompleteRequest(id="req-1", query="Hello")
        assert req.query == "Hello"
        assert req.system_prompt is None

    def test_complete_response(self):
        resp = CompleteResponse(
            id="req-1",
            response_id="resp-1",
            content="Hi there",
            model="gpt-4",
            tokens_used=10,
        )
        assert resp.response_id == "resp-1"
        assert resp.content == "Hi there"

    def test_remember_request(self):
        req = RememberRequest(id="req-1", fact="User likes Python")
        assert req.fact == "User likes Python"
        assert req.category == "general"

    def test_forget_request(self):
        req = ForgetRequest(id="req-1", fact_id=42)
        assert req.fact_id == 42

    def test_recall_request(self):
        req = RecallRequest(id="req-1", query="programming")
        assert req.query == "programming"
        assert req.top_k is None
        assert req.categories is None

    def test_feedback_request(self):
        req = FeedbackRequest(id="req-1", response_id="resp-1", signal="positive")
        assert req.response_id == "resp-1"
        assert req.signal == "positive"
        assert req.correction is None


class TestAgentHandleRequest:
    """Tests for Agent.handle_request method."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_learn(self):
        """Create mock LearnClient."""
        learn = MagicMock()
        learn.facts = MagicMock()
        learn.facts.list_active.return_value = []
        learn.feedback = MagicMock()
        learn.preferences = MagicMock()
        return learn

    @pytest.fixture
    def mock_context_builder(self):
        """Patch ContextBuilder to avoid database calls."""
        with patch("llm_agent.agent.ContextBuilder") as mock_cls:
            mock_builder = MagicMock()
            mock_builder.build_system_prompt.side_effect = lambda base_prompt, **_: base_prompt
            mock_cls.return_value = mock_builder
            yield mock_cls

    @pytest.fixture
    def agent(self, mock_logger, mock_learn, mock_context_builder):
        """Create a test agent."""
        from llm_agent import Agent, AgentConfig

        config = AgentConfig(name="test-agent", fact_injection="none")
        llm = MagicMock()
        llm.complete.return_value = CompletionResult(
            id="resp-123",
            content="Hello!",
            model="gpt-4",
            tokens_used=25,
            latency_ms=100,
        )
        return Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn)

    def test_handle_health_request(self, agent):
        req = HealthRequest(id="req-1")

        resp = agent.handle_request(req)

        assert isinstance(resp, HealthResponse)
        assert resp.id == "req-1"
        assert resp.status == "ok"
        assert resp.agent_name == "test-agent"

    def test_handle_complete_request(self, agent):
        req = CompleteRequest(id="req-1", query="Hello")

        resp = agent.handle_request(req)

        assert isinstance(resp, CompleteResponse)
        assert resp.id == "req-1"
        assert resp.success is True
        assert resp.response_id == "resp-123"
        assert resp.content == "Hello!"

    def test_handle_remember_request(self, agent, mock_learn):
        mock_learn.facts.add.return_value = 42
        req = RememberRequest(id="req-1", fact="User likes Python", category="preferences")

        resp = agent.handle_request(req)

        assert isinstance(resp, RememberResponse)
        assert resp.success is True
        assert resp.fact_id == 42
        mock_learn.facts.add.assert_called_once_with("User likes Python", category="preferences")

    def test_handle_forget_request(self, agent, mock_learn):
        req = ForgetRequest(id="req-1", fact_id=42)

        resp = agent.handle_request(req)

        assert isinstance(resp, ForgetResponse)
        assert resp.success is True
        mock_learn.facts.delete.assert_called_once_with(42)

    def test_handle_feedback_request(self, agent, mock_learn):
        # First complete a request to track the response
        complete_req = CompleteRequest(id="req-1", query="Hello")
        agent.handle_request(complete_req)

        # Then provide feedback
        feedback_req = FeedbackRequest(id="req-2", response_id="resp-123", signal="positive")
        resp = agent.handle_request(feedback_req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True
        mock_learn.feedback.record.assert_called_once()

    def test_handle_unknown_message_type(self, agent):
        # Create a request with unknown message type
        from llm_agent.protocol.base import Request

        class UnknownRequest(Request):
            message_type = "unknown_type"

        req = UnknownRequest(id="req-1")
        resp = agent.handle_request(req)

        assert resp.success is False
        assert "Unknown message type" in resp.error

    def test_handle_complete_request_error(self, agent):
        # Make LLM raise an exception
        agent._llm.complete.side_effect = ValueError("LLM error")
        req = CompleteRequest(id="req-1", query="Hello")

        resp = agent.handle_request(req)

        assert isinstance(resp, CompleteResponse)
        assert resp.success is False
        assert "LLM error" in resp.error

    def test_handle_remember_request_error(self, agent, mock_learn):
        mock_learn.facts.add.side_effect = ValueError("Storage error")
        req = RememberRequest(id="req-1", fact="test fact")

        resp = agent.handle_request(req)

        assert isinstance(resp, RememberResponse)
        assert resp.success is False
        assert "Storage error" in resp.error
        assert resp.fact_id == -1

    def test_handle_forget_request_error(self, agent, mock_learn):
        mock_learn.facts.delete.side_effect = ValueError("Fact not found")
        req = ForgetRequest(id="req-1", fact_id=999)

        resp = agent.handle_request(req)

        assert isinstance(resp, ForgetResponse)
        assert resp.success is False
        assert "Fact not found" in resp.error

    def test_handle_recall_request_error(self, agent, mock_learn):
        # recall() raises when embedder not configured
        req = RecallRequest(id="req-1", query="test query")

        resp = agent.handle_request(req)

        from llm_agent.protocol.v1 import RecallResponse

        assert isinstance(resp, RecallResponse)
        assert resp.success is False
        assert "embedder" in resp.error.lower()

    def test_handle_recall_request_success(self, mock_logger, mock_learn, mock_context_builder):
        """Verify successful recall returns facts serialized correctly."""
        from llm_agent import Agent, AgentConfig
        from llm_agent.protocol.v1 import RecallResponse

        # Create embedder mock
        embedder = MagicMock()
        embedder.model = "test-embed-model"
        embedding_result = MagicMock()
        embedding_result.embedding = [0.1, 0.2, 0.3]
        embedder.embed.return_value = embedding_result

        # Setup mock scored facts
        mock_fact = MagicMock()
        mock_fact.id = 42
        mock_fact.content = "User prefers Python"
        mock_fact.category = "preferences"
        mock_scored_fact = MagicMock()
        mock_scored_fact.fact = mock_fact
        mock_scored_fact.similarity = 0.85
        mock_learn.facts.search_similar.return_value = [mock_scored_fact]

        # Create agent with embedder
        config = AgentConfig(name="test-agent", fact_injection="none")
        llm = MagicMock()
        agent = Agent(lg=mock_logger, config=config, llm=llm, learn=mock_learn, embedder=embedder)

        req = RecallRequest(id="req-1", query="programming")
        resp = agent.handle_request(req)

        assert isinstance(resp, RecallResponse)
        assert resp.success is True
        assert resp.id == "req-1"
        assert len(resp.facts) == 1
        assert resp.facts[0]["fact_id"] == 42
        assert resp.facts[0]["content"] == "User prefers Python"
        assert resp.facts[0]["category"] == "preferences"
        assert resp.facts[0]["similarity"] == 0.85

    def test_handle_feedback_request_error(self, agent):
        # Feedback on unknown response_id
        req = FeedbackRequest(id="req-1", response_id="nonexistent", signal="positive")

        resp = agent.handle_request(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is False
        assert "Unknown response_id" in resp.error


class TestHTTPTraitLifecycle:
    """Tests for HTTPTrait on_start/on_stop lifecycle."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test-agent"
        return agent

    @patch("llm_agent.server.http.mp.Queue")
    @patch("appinfra.app.fastapi.ServerBuilder")
    def test_on_start_creates_server(self, mock_builder_cls, mock_queue, mock_logger, mock_agent):
        """Verify on_start creates server with correct configuration."""
        mock_queue.return_value = MagicMock()
        mock_server = MagicMock()
        mock_server.start_subprocess.return_value = MagicMock()

        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.with_host.return_value = mock_builder
        mock_builder.with_port.return_value = mock_builder
        mock_builder.with_title.return_value = mock_builder
        mock_builder.with_description.return_value = mock_builder
        mock_builder.with_version.return_value = mock_builder
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
        trait = HTTPTrait(lg=mock_logger, config=config)
        trait.attach(mock_agent)
        trait.on_start()

        mock_server.start_subprocess.assert_called_once()
        trait.on_stop()

    @patch("llm_agent.server.http.mp.Queue")
    @patch("appinfra.app.fastapi.ServerBuilder")
    def test_on_stop_stops_server(self, mock_builder_cls, mock_queue, mock_logger, mock_agent):
        """Verify on_stop stops server and cleans up."""
        mock_queue.return_value = MagicMock()
        mock_process = MagicMock()
        mock_server = MagicMock()
        mock_server.start_subprocess.return_value = mock_process

        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.with_host.return_value = mock_builder
        mock_builder.with_port.return_value = mock_builder
        mock_builder.with_title.return_value = mock_builder
        mock_builder.with_description.return_value = mock_builder
        mock_builder.with_version.return_value = mock_builder
        mock_builder.subprocess = MagicMock()
        mock_builder.subprocess.with_ipc.return_value = mock_builder.subprocess
        mock_builder.subprocess.with_response_timeout.return_value = mock_builder.subprocess
        mock_builder.subprocess.with_auto_restart.return_value = mock_builder.subprocess
        mock_builder.subprocess.done.return_value = mock_builder
        mock_builder.routes = MagicMock()
        mock_builder.routes.with_router.return_value = mock_builder.routes
        mock_builder.routes.done.return_value = mock_builder
        mock_builder.build.return_value = mock_server

        trait = HTTPTrait(lg=mock_logger)
        trait.attach(mock_agent)
        trait.on_start()
        trait.on_stop()

        mock_server.stop.assert_called_once()


class TestHTTPTraitIPCLoop:
    """Tests for IPC loop behavior."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_default_handler_returns_error_response(self, mock_logger):
        """Verify default handler returns error instead of None."""
        trait = HTTPTrait(lg=mock_logger)

        # Create a mock request
        mock_request = MagicMock()
        mock_request.id = "req-123"

        response = trait._default_handler(mock_request)

        assert response is not None
        assert response.success is False
        assert response.id == "req-123"
        assert "does not implement handle_request" in response.error

    def test_default_handler_handles_missing_id(self, mock_logger):
        """Verify default handler handles request without id attribute."""
        trait = HTTPTrait(lg=mock_logger)

        # Create a mock request without id
        mock_request = MagicMock(spec=[])  # Empty spec means no attributes

        response = trait._default_handler(mock_request)

        assert response is not None
        assert response.success is False
        assert response.id == "unknown"

    def test_ipc_loop_handles_handler_exception(self, mock_logger):
        """Verify IPC loop sends error response when handler raises."""
        from queue import Queue

        from llm_agent.protocol.v1 import HealthRequest

        trait = HTTPTrait(lg=mock_logger)

        # Create mock server with real queues for testing
        mock_server = MagicMock()
        mock_server.request_queue = Queue()
        mock_server.response_queue = Queue()
        trait._server = mock_server
        trait._agent = MagicMock()

        # Make handle_request raise an exception
        def failing_handler(request):
            raise RuntimeError("Handler crashed")

        trait._agent.handle_request = failing_handler

        # Put a request on the queue
        req = HealthRequest(id="req-1")
        mock_server.request_queue.put(req)

        # Signal shutdown after processing one request
        import threading

        def shutdown_after_delay():
            import time

            time.sleep(0.2)
            trait._ipc_shutdown.set()

        shutdown_thread = threading.Thread(target=shutdown_after_delay)
        shutdown_thread.start()

        # Run the IPC loop (will process one request then exit)
        trait._ipc_loop(poll_timeout=0.05)
        shutdown_thread.join()

        # Verify error response was sent
        assert not mock_server.response_queue.empty()
        response = mock_server.response_queue.get()
        assert response.success is False
        assert response.error == "Internal server error"
        assert response.id == "req-1"
