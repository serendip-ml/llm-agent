"""Communication abstractions for agent processes.

Provides pluggable transport layer:
- QueueChannel: multiprocessing.Queue-based (default, single-machine)
- [Future] ZMQChannel: ZeroMQ-based (distributed deployment)
"""

from llm_agent.runtime.transport.base import Channel, Message, MessageType, Request, Response
from llm_agent.runtime.transport.queue import QueueChannel, create_channel_pair


__all__ = [
    "Channel",
    "Message",
    "MessageType",
    "Request",
    "Response",
    "QueueChannel",
    "create_channel_pair",
]
