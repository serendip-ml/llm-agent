"""LLM trait for agent completion capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_infer.client import ChatResponse, LLMClient

from llm_agent.llm.types import CompletionResult, Message


if TYPE_CHECKING:
    from llm_agent.agent import Agent


@dataclass
class LLMConfig:
    """Configuration for LLM trait.

    Attributes:
        base_url: Base URL for OpenAI-compatible API.
        model: Model name to use.
        api_key: Optional API key for authenticated APIs.
        timeout: Request timeout in seconds.
        temperature: Default sampling temperature.
        max_tokens: Default max tokens (None = model default).
    """

    base_url: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str | None = None
    timeout: float = 120.0
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass
class LLMTrait:
    """LLM capability trait.

    Wraps llm_infer.client.LLMClient to provide completion capability
    to agents. Uses sync API for compatibility with current agent architecture.

    Example:
        from llm_agent.traits import LLMTrait, LLMConfig

        agent = Agent(lg, config)
        agent.add_trait(LLMTrait(LLMConfig(
            base_url="http://localhost:8000/v1",
            model="qwen2.5-72b",
        )))

        # Agent can now use complete()
        result = agent.complete("What is 2+2?")

    Lifecycle:
        - attach(): Stores agent reference
        - on_start(): Creates LLMClient
        - on_stop(): Closes LLMClient
    """

    config: LLMConfig = field(default_factory=LLMConfig)

    _agent: Agent | None = field(default=None, repr=False, compare=False)
    _client: LLMClient | None = field(default=None, repr=False, compare=False)

    def attach(self, agent: Agent) -> None:
        """Attach trait to agent.

        Args:
            agent: The agent this trait is attached to.
        """
        self._agent = agent

    def on_start(self) -> None:
        """Create LLM client on agent start."""
        self._client = LLMClient.openai(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    def on_stop(self) -> None:
        """Close LLM client on agent stop."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> LLMClient:
        """Access the LLM client.

        Raises:
            RuntimeError: If trait not started (on_start not called).
        """
        if self._client is None:
            raise RuntimeError("LLMTrait not started - ensure agent.start() was called")
        return self._client

    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Generate a completion.

        Args:
            messages: Conversation messages.
            model: Model override (uses config default if None).
            temperature: Temperature override (uses config default if None).
            max_tokens: Max tokens override (uses config default if None).
            tools: Tool definitions for function calling.

        Returns:
            CompletionResult with content and metadata.
        """
        # Convert Message objects to dicts for LLMClient
        api_messages = self._messages_to_dicts(messages)

        response = self.client.chat_full(
            messages=api_messages,
            model=model or self.config.model,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            tools=tools,
        )

        return self._response_to_result(response)

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to API format."""
        result = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            result.append(msg)
        return result

    def _response_to_result(self, response: ChatResponse) -> CompletionResult:
        """Convert ChatResponse to CompletionResult."""
        import uuid

        tokens_used = self._extract_tokens(response)
        tool_calls = self._extract_tool_calls(response)

        return CompletionResult(
            id=str(uuid.uuid4()),
            content=response.content,
            model=response.model or self.config.model,
            tokens_used=tokens_used,
            latency_ms=0,
            tool_calls=tool_calls,
        )

    def _extract_tokens(self, response: ChatResponse) -> int:
        """Extract token count from response usage."""
        if not response.usage:
            return 0
        return response.usage.total_tokens or (
            (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
        )

    def _extract_tool_calls(self, response: ChatResponse) -> list[dict[str, Any]] | None:
        """Extract and convert tool calls from response."""
        if not response.tool_calls:
            return None
        return [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in response.tool_calls
        ]
