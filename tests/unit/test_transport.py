"""Tests for runtime transport layer."""

from unittest.mock import MagicMock

import pytest

from llm_gent.runtime.transport import (
    Channel,
    Message,
    QueueChannel,
    Request,
    Response,
    create_channel_pair,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger():
    """Create mock Logger for tests."""
    return MagicMock()


class TestMessage:
    """Tests for Message dataclass."""

    def test_default_id(self):
        """Message gets auto-generated ID."""
        msg = Message(type="test")
        assert msg.id  # Non-empty
        assert len(msg.id) == 32  # UUID hex

    def test_explicit_id(self):
        """Message accepts explicit ID."""
        msg = Message(id="my-id", type="test")
        assert msg.id == "my-id"

    def test_default_payload(self):
        """Message has empty payload by default."""
        msg = Message(type="test")
        assert msg.payload == {}

    def test_with_payload(self):
        """Message stores payload."""
        msg = Message(type="test", payload={"key": "value"})
        assert msg.payload == {"key": "value"}


class TestRequest:
    """Tests for Request dataclass."""

    def test_inherits_message(self):
        """Request inherits from Message."""
        req = Request(type="ask", payload={"question": "test"})
        assert isinstance(req, Message)
        assert req.type == "ask"


class TestResponse:
    """Tests for Response dataclass."""

    def test_inherits_message(self):
        """Response inherits from Message."""
        resp = Response(type="ask_response", request_id="req-123", success=True)
        assert isinstance(resp, Message)

    def test_success_response(self):
        """Response can be successful."""
        resp = Response(request_id="req-123", success=True)
        assert resp.success is True
        assert resp.error is None

    def test_error_response(self):
        """Response can contain error."""
        resp = Response(request_id="req-123", success=False, error="Something went wrong")
        assert resp.success is False
        assert resp.error == "Something went wrong"


class TestQueueChannel:
    """Tests for QueueChannel."""

    def test_send_recv_roundtrip(self, mock_logger):
        """Messages can be sent and received."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        # Main sends, subprocess receives
        msg = Message(type="test", payload={"data": 123})
        main_ch.send(msg)

        received = sub_ch.recv(timeout=1.0)
        assert received is not None
        assert received.type == "test"
        assert received.payload == {"data": 123}

    def test_bidirectional(self, mock_logger):
        """Both channels can send and receive."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        # Main -> Sub
        main_ch.send(Message(type="from_main"))
        assert sub_ch.recv(timeout=1.0).type == "from_main"

        # Sub -> Main
        sub_ch.send(Message(type="from_sub"))
        assert main_ch.recv(timeout=1.0).type == "from_sub"

    def test_recv_timeout(self, mock_logger):
        """recv returns None on timeout."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        result = main_ch.recv(timeout=0.1)
        assert result is None

    def test_close_prevents_send(self, mock_logger):
        """Closed channel raises on send."""
        main_ch, sub_ch = create_channel_pair(mock_logger)
        main_ch.close()

        with pytest.raises(RuntimeError, match="closed"):
            main_ch.send(Message(type="test"))

    def test_close_prevents_recv(self, mock_logger):
        """Closed channel raises on recv."""
        main_ch, sub_ch = create_channel_pair(mock_logger)
        main_ch.close()

        with pytest.raises(RuntimeError, match="closed"):
            main_ch.recv(timeout=0.1)

    def test_is_closed(self, mock_logger):
        """is_closed property reflects state."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        assert main_ch.is_closed is False
        main_ch.close()
        assert main_ch.is_closed is True

    def test_request_response(self, mock_logger):
        """request() waits for matching response."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        # Simulate subprocess: receive request, send response
        import threading

        def responder():
            req = sub_ch.recv(timeout=5.0)
            sub_ch.send(
                Response(
                    type="ask_response",
                    request_id=req.id,
                    success=True,
                    payload={"answer": 42},
                )
            )

        thread = threading.Thread(target=responder)
        thread.start()

        # Main process: send request, wait for response
        req = Request(type="ask", payload={"question": "What is 6*7?"})
        resp = main_ch.request(req, timeout=5.0)

        thread.join()

        assert resp.request_id == req.id
        assert resp.success is True
        assert resp.payload["answer"] == 42

    def test_request_timeout(self, mock_logger):
        """request() raises TimeoutError if no response."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        req = Request(type="ask", payload={"question": "test"})

        with pytest.raises(TimeoutError):
            main_ch.request(req, timeout=0.1)

    def test_close_queues(self, mock_logger):
        """close_queues() closes underlying queues."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        main_ch.close_queues()
        assert main_ch.is_closed is True

    def test_set_logger(self, mock_logger):
        """set_logger() updates the channel's logger."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        new_logger = MagicMock()
        main_ch.set_logger(new_logger)
        assert main_ch._lg is new_logger


class TestChannelProtocol:
    """Tests for Channel protocol."""

    def test_queue_channel_implements_protocol(self, mock_logger):
        """QueueChannel implements Channel protocol."""
        main_ch, _ = create_channel_pair(mock_logger)
        assert isinstance(main_ch, Channel)


class TestCreateChannelPair:
    """Tests for create_channel_pair."""

    def test_returns_two_channels(self, mock_logger):
        """create_channel_pair returns tuple of two channels."""
        result = create_channel_pair(mock_logger)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], QueueChannel)
        assert isinstance(result[1], QueueChannel)

    def test_channels_are_connected(self, mock_logger):
        """Created channels are connected to each other."""
        main_ch, sub_ch = create_channel_pair(mock_logger)

        main_ch.send(Message(type="test1"))
        assert sub_ch.recv(timeout=1.0).type == "test1"

        sub_ch.send(Message(type="test2"))
        assert main_ch.recv(timeout=1.0).type == "test2"
