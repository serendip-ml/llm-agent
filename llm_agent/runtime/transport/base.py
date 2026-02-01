"""Abstract protocol for agent communication.

Defines the Channel protocol and message types used for communication
between the main process and agent subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4


class MessageType(StrEnum):
    """Message type constants for agent communication.

    Using StrEnum ensures:
    - Typos are caught at development time
    - IDE autocomplete works
    - Values are still strings for serialization
    """

    # Lifecycle messages
    SHUTDOWN = "shutdown"
    STARTED = "started"
    ERROR = "error"

    # Request/response types
    ASK = "ask"
    ASK_RESPONSE = "ask_response"
    FEEDBACK = "feedback"
    FEEDBACK_RESPONSE = "feedback_response"
    GET_INSIGHTS = "get_insights"
    INSIGHTS_RESPONSE = "insights_response"

    # Scheduled execution
    RUN_CYCLE = "run_cycle"
    CYCLE_COMPLETE = "cycle_complete"
    CYCLE_ERROR = "cycle_error"


@dataclass
class Message:
    """Base message for agent communication.

    Attributes:
        id: Unique message identifier.
        type: Message type (e.g., 'ask', 'shutdown', 'cycle').
        payload: Message-specific data.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Request(Message):
    """Request message expecting a response.

    Inherits from Message but semantically indicates the sender expects
    a Response with matching request_id.
    """

    pass


@dataclass
class Response(Message):
    """Response to a request.

    Attributes:
        request_id: ID of the request this responds to.
        success: Whether the request was handled successfully.
        error: Error message if success is False.
    """

    request_id: str = ""
    success: bool = True
    error: str | None = None


@runtime_checkable
class Channel(Protocol):
    """Abstract communication channel.

    Implementations:
    - QueueChannel: Uses multiprocessing.Queue (default)
    - [Future] ZMQChannel: Uses ZeroMQ for distributed deployment

    The Channel protocol is intentionally minimal and sync-first.
    No threading, no async - just simple send/recv operations.
    """

    def send(self, msg: Message) -> None:
        """Send message (non-blocking).

        Args:
            msg: Message to send.
        """
        ...

    def recv(self, timeout: float | None = None) -> Message | None:
        """Receive message (blocking with optional timeout).

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Received message, or None if timeout expired.
        """
        ...

    def request(self, req: Request, timeout: float = 60.0) -> Response:
        """Send request and wait for matching response.

        Args:
            req: Request to send.
            timeout: Maximum seconds to wait for response.

        Returns:
            Response matching the request.

        Raises:
            TimeoutError: If no response within timeout.
        """
        ...

    def close(self) -> None:
        """Close the channel.

        After close(), send() and recv() should raise or return None.
        """
        ...
