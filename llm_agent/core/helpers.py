"""Internal helper classes for agent implementation."""

from __future__ import annotations


class ResponseContext:
    """Tracks response context for feedback.

    Used internally by ConversationalAgent to store the context needed
    to record feedback on a previous completion.
    """

    __slots__ = ("system_prompt", "query", "response", "model")

    def __init__(
        self,
        system_prompt: str,
        query: str,
        response: str,
        model: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.query = query
        self.response = response
        self.model = model
