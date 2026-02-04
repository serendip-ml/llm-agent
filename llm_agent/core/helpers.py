"""Internal helper classes for agent implementation."""

from __future__ import annotations


class ResponseContext:
    """Tracks response context for feedback.

    TODO: Review after ConversationalAgent removal - may not be needed,
    or could be useful for other feedback tracking patterns.
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
