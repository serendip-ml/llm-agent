"""Queue-based channel implementation using multiprocessing.Queue.

Simple, reliable, built-in. Good for single-machine deployment.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty
from typing import TYPE_CHECKING

from llm_agent.runtime.transport.base import Message, Request, Response


if TYPE_CHECKING:
    from multiprocessing.queues import Queue


class QueueChannel:
    """Channel using multiprocessing.Queue.

    Uses two queues for bidirectional communication:
    - _send_q: Queue for outgoing messages
    - _recv_q: Queue for incoming messages

    Thread-safe and process-safe via multiprocessing.Queue.
    """

    def __init__(self, send_q: Queue[Message], recv_q: Queue[Message]) -> None:
        """Initialize channel with send and receive queues.

        Args:
            send_q: Queue for sending messages.
            recv_q: Queue for receiving messages.
        """
        self._send_q = send_q
        self._recv_q = recv_q
        self._closed = False

    def send(self, msg: Message) -> None:
        """Send message via queue (non-blocking).

        Args:
            msg: Message to send.

        Raises:
            RuntimeError: If channel is closed.
        """
        if self._closed:
            raise RuntimeError("Channel is closed")
        self._send_q.put(msg)

    def recv(self, timeout: float | None = None) -> Message | None:
        """Receive message from queue (blocking with optional timeout).

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Received message, or None if timeout expired.

        Raises:
            RuntimeError: If channel is closed.
        """
        if self._closed:
            raise RuntimeError("Channel is closed")
        try:
            return self._recv_q.get(timeout=timeout)
        except Empty:
            return None

    def request(self, req: Request, timeout: float = 60.0) -> Response:
        """Send request and wait for matching response.

        This is a convenience method that sends a request and waits for
        a Response with matching request_id.

        Args:
            req: Request to send.
            timeout: Maximum seconds to wait for response.

        Returns:
            Response matching the request.

        Raises:
            TimeoutError: If no matching response received within timeout.
            RuntimeError: If channel is closed.
        """
        self.send(req)
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = max(0.001, deadline - time.time())
            msg = self.recv(timeout=min(0.1, remaining))

            if msg is None:
                continue

            if isinstance(msg, Response) and msg.request_id == req.id:
                return msg

            # Non-matching message - this shouldn't happen in normal use
            # but we'll keep polling until we get our response or timeout

        raise TimeoutError(f"No response to request {req.id} within {timeout}s")

    def close(self) -> None:
        """Mark channel as closed.

        Note: Does not close the underlying queues as they may be shared
        and should be closed by their owner (typically the registry).
        """
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Check if channel is closed."""
        return self._closed


def create_channel_pair() -> tuple[QueueChannel, QueueChannel]:
    """Create a pair of connected channels for main<->subprocess communication.

    Returns a tuple of (main_channel, subprocess_channel) where:
    - main_channel.send() -> subprocess_channel.recv()
    - subprocess_channel.send() -> main_channel.recv()

    Returns:
        Tuple of (main_channel, subprocess_channel).
    """
    # Create two queues
    q1: Queue[Message] = mp.Queue()
    q2: Queue[Message] = mp.Queue()

    # Main sends on q1, receives on q2
    main_channel = QueueChannel(send_q=q1, recv_q=q2)

    # Subprocess sends on q2, receives on q1
    subprocess_channel = QueueChannel(send_q=q2, recv_q=q1)

    return main_channel, subprocess_channel
