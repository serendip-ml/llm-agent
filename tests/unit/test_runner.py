"""Tests for Agent Runner."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agent.core.traits.llm import LLMConfig
from llm_agent.runtime import AgentRunner
from llm_agent.runtime.transport import Message, MessageType, Response


pytestmark = pytest.mark.unit


class TestAgentRunner:
    """Tests for AgentRunner."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_channel(self):
        channel = MagicMock()
        channel.recv.return_value = None
        return channel

    @pytest.fixture
    def llm_config(self):
        return LLMConfig(base_url="http://localhost:8000/v1")

    @pytest.fixture
    def config(self):
        return {
            "name": "test-agent",
            "directive": {"prompt": "Test prompt"},
            "task": {"prompt": "Test task"},
        }

    @pytest.fixture
    def runner(self, mock_logger, mock_channel, llm_config, config):
        return AgentRunner(
            name="test-agent",
            config=config,
            channel=mock_channel,
            lg=mock_logger,
            llm_config=llm_config,
        )

    def test_runner_init(self, runner):
        """Runner initializes with correct attributes."""
        assert runner._name == "test-agent"
        assert runner._running is False
        assert runner._agent is None
        assert runner._schedule_interval is None

    def test_runner_init_with_schedule(self, mock_logger, mock_channel, llm_config):
        """Runner extracts schedule interval from config."""
        config = {
            "name": "scheduled",
            "schedule": {"interval": 300},
        }
        runner = AgentRunner(
            name="scheduled",
            config=config,
            channel=mock_channel,
            lg=mock_logger,
            llm_config=llm_config,
        )

        assert runner._schedule_interval == 300.0

    def test_calculate_timeout_no_schedule(self, runner):
        """Calculate timeout returns 60s when no schedule."""
        timeout = runner._calculate_timeout(0)
        assert timeout == 60.0

    def test_calculate_timeout_with_schedule(self, mock_logger, mock_channel, llm_config):
        """Calculate timeout respects schedule interval."""
        import time

        config = {"schedule": {"interval": 30}}
        runner = AgentRunner(
            name="test",
            config=config,
            channel=mock_channel,
            lg=mock_logger,
            llm_config=llm_config,
        )

        last_cycle = time.time()
        timeout = runner._calculate_timeout(last_cycle)

        # Should be close to 30s (schedule interval)
        assert 29.0 <= timeout <= 30.0

    def test_should_run_cycle_no_schedule(self, runner):
        """Should not run cycle when no schedule."""
        assert runner._should_run_cycle(0) is False

    def test_should_run_cycle_with_schedule(self, mock_logger, mock_channel, llm_config):
        """Should run cycle when interval elapsed."""
        import time

        config = {"schedule": {"interval": 1}}
        runner = AgentRunner(
            name="test",
            config=config,
            channel=mock_channel,
            lg=mock_logger,
            llm_config=llm_config,
        )

        # Last cycle was 2 seconds ago
        last_cycle = time.time() - 2
        assert runner._should_run_cycle(last_cycle) is True

    def test_handle_shutdown(self, runner, mock_channel):
        """Handle shutdown message stops the runner."""
        runner._running = True
        msg = Message(type=MessageType.SHUTDOWN)

        runner._handle_message(msg)

        assert runner._running is False

    def test_handle_ask_without_agent(self, runner, mock_channel):
        """Handle ask sends error when agent not initialized."""
        runner._agent = None
        msg = Message(id="test-id", type=MessageType.ASK, payload={"question": "test"})

        runner._handle_ask(msg)

        # Should send error response
        mock_channel.send.assert_called_once()
        call = mock_channel.send.call_args[0][0]
        assert isinstance(call, Response)
        assert call.success is False
        assert call.request_id == "test-id"

    def test_handle_ask_with_agent(self, runner, mock_channel):
        """Handle ask calls agent.ask and returns response."""
        mock_agent = MagicMock()
        mock_agent.ask.return_value = "Agent response"
        runner._agent = mock_agent

        msg = Message(id="test-id", type=MessageType.ASK, payload={"question": "test question"})

        runner._handle_ask(msg)

        mock_agent.ask.assert_called_once_with("test question")
        mock_channel.send.assert_called_once()
        call = mock_channel.send.call_args[0][0]
        assert isinstance(call, Response)
        assert call.success is True
        assert call.payload["response"] == "Agent response"

    def test_handle_feedback_without_agent(self, runner, mock_channel):
        """Handle feedback sends error when agent not initialized."""
        runner._agent = None
        msg = Message(id="test-id", type=MessageType.FEEDBACK, payload={"message": "test"})

        runner._handle_feedback(msg)

        call = mock_channel.send.call_args[0][0]
        assert call.success is False

    def test_handle_feedback_with_agent(self, runner, mock_channel):
        """Handle feedback calls agent.record_feedback."""
        mock_agent = MagicMock()
        runner._agent = mock_agent

        msg = Message(id="test-id", type=MessageType.FEEDBACK, payload={"message": "Good job!"})

        runner._handle_feedback(msg)

        mock_agent.record_feedback.assert_called_once_with("Good job!")
        call = mock_channel.send.call_args[0][0]
        assert call.success is True

    def test_handle_get_insights_without_agent(self, runner, mock_channel):
        """Handle get_insights sends error when agent not initialized."""
        runner._agent = None
        msg = Message(id="test-id", type=MessageType.GET_INSIGHTS, payload={"limit": 10})

        runner._handle_get_insights(msg)

        call = mock_channel.send.call_args[0][0]
        assert call.success is False

    def test_handle_get_insights_with_agent(self, runner, mock_channel):
        """Handle get_insights returns agent results."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "Test content"
        mock_result.parsed = None
        mock_result.iterations = 1
        mock_agent.get_recent_results.return_value = [mock_result]
        runner._agent = mock_agent

        msg = Message(id="test-id", type=MessageType.GET_INSIGHTS, payload={"limit": 5})

        runner._handle_get_insights(msg)

        mock_agent.get_recent_results.assert_called_once_with(5)
        call = mock_channel.send.call_args[0][0]
        assert call.success is True
        assert len(call.payload["insights"]) == 1

    def test_handle_run_cycle(self, runner, mock_channel):
        """Handle run_cycle calls _run_cycle."""
        runner._agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.iterations = 1
        runner._agent.run_once.return_value = mock_result

        msg = Message(type=MessageType.RUN_CYCLE)

        runner._handle_message(msg)

        runner._agent.run_once.assert_called_once()

    def test_run_cycle_sends_complete(self, runner, mock_channel):
        """Run cycle sends CYCLE_COMPLETE on success."""
        runner._agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.iterations = 3
        runner._agent.run_once.return_value = mock_result

        runner._run_cycle()

        call = mock_channel.send.call_args[0][0]
        assert call.type == MessageType.CYCLE_COMPLETE
        assert call.payload["success"] is True
        assert call.payload["iterations"] == 3

    def test_run_cycle_sends_error_on_failure(self, runner, mock_channel):
        """Run cycle sends CYCLE_ERROR on exception."""
        runner._agent = MagicMock()
        runner._agent.run_once.side_effect = RuntimeError("Cycle failed")

        runner._run_cycle()

        call = mock_channel.send.call_args[0][0]
        assert call.type == MessageType.CYCLE_ERROR
        assert "Cycle failed" in call.payload["error"]

    def test_send_error_response_maps_types(self, runner, mock_channel):
        """Send error response maps request types to response types."""
        msg = Message(id="test-id", type=MessageType.ASK)

        runner._send_error_response(msg, "Error message")

        call = mock_channel.send.call_args[0][0]
        assert call.type == MessageType.ASK_RESPONSE
        assert call.request_id == "test-id"
        assert call.success is False
        assert call.error == "Error message"

    def test_cleanup_stops_agent(self, runner, mock_channel):
        """Cleanup stops agent and closes channel."""
        mock_agent = MagicMock()
        runner._agent = mock_agent

        runner._cleanup()

        mock_agent.stop.assert_called_once()
        mock_channel.close.assert_called_once()

    def test_cleanup_handles_agent_stop_error(self, runner, mock_channel, mock_logger):
        """Cleanup handles errors from agent.stop()."""
        mock_agent = MagicMock()
        mock_agent.stop.side_effect = RuntimeError("Stop failed")
        runner._agent = mock_agent

        runner._cleanup()  # Should not raise

        mock_channel.close.assert_called_once()

    def test_handle_unknown_message_type(self, runner, mock_channel, mock_logger):
        """Handle unknown message type logs warning."""
        msg = Message(type="unknown_type")

        runner._handle_message(msg)

        mock_logger.warning.assert_called()


class TestAgentRunnerIntegration:
    """Integration tests for AgentRunner.run()."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def llm_config(self):
        return LLMConfig(base_url="http://localhost:8000/v1")

    def test_run_starts_agent_and_sends_started(self, mock_logger, llm_config):
        """Run starts agent and notifies registry."""
        mock_channel = MagicMock()

        # Simulate shutdown after startup
        def recv_side_effect(timeout=None):
            return Message(type=MessageType.SHUTDOWN)

        mock_channel.recv.side_effect = recv_side_effect

        config = {
            "name": "test",
            "directive": {"prompt": "Test"},
            "task": {"prompt": "Task"},
        }

        runner = AgentRunner(
            name="test",
            config=config,
            channel=mock_channel,
            lg=mock_logger,
            llm_config=llm_config,
        )

        with patch.object(runner, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            runner.run()

            mock_agent.start.assert_called_once()
            mock_agent.stop.assert_called_once()

            # Check that STARTED message was sent
            calls = mock_channel.send.call_args_list
            assert any(call[0][0].type == MessageType.STARTED for call in calls)
