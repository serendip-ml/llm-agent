"""LLM trait for agent completion capability."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llm_infer.client import ChatResponse, LLMClient
from llm_infer.client import Factory as LLMClientFactory
from pydantic import BaseModel, ValidationError

from ...llm.backend import StructuredOutputError
from ...llm.types import CompletionResult, Message
from ..base import BaseTrait


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent


# Type alias for LLM configuration dict
LLMConfig = dict[str, Any]
"""LLM configuration dictionary.

Supports multi-backend format (see llm-infer LLMClient.from_config):

    default: local
    backends:
      local:
        type: openai_compatible
        base_url: http://localhost:8000/v1
        model: qwen2.5-72b
      anthropic:
        type: anthropic
        model: claude-sonnet-4-20250514
"""


def _resolve_llm_defaults(config: LLMConfig) -> dict[str, Any]:
    """Extract default values from LLM config.

    Returns dict with model, temperature, max_tokens from the selected backend.
    """
    backends = config.get("backends", {})
    default_name = config.get("default")

    if not backends:
        # Single backend config (no "backends" wrapper)
        return {
            "model": config.get("model", "default"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens"),
        }

    if not default_name:
        default_name = next(iter(backends.keys()))

    backend_config = backends.get(default_name, {})
    return {
        "model": backend_config.get("model", "default"),
        "temperature": backend_config.get("temperature", 0.7),
        "max_tokens": backend_config.get("max_tokens"),
    }


class LLMTrait(BaseTrait):
    """LLM capability trait.

    Wraps llm_infer.client.LLMClient to provide completion capability
    to agents. Uses sync API for compatibility with current agent architecture.

    Example:
        from llm_agent.core.traits import LLMTrait

        llm_config = {
            "default": "local",
            "backends": {
                "local": {"type": "openai_compatible", "base_url": "...", "model": "..."}
            }
        }
        agent = Agent(lg, config)
        agent.add_trait(LLMTrait(agent, llm_config))

        # Agent can now use complete()
        result = agent.complete("What is 2+2?")

    Lifecycle:
        - on_start(): Creates LLMClient
        - on_stop(): Closes LLMClient
    """

    def __init__(self, agent: Agent, config: LLMConfig | None = None) -> None:
        """Initialize LLM trait.

        Args:
            agent: The agent this trait belongs to.
            config: LLM configuration dict.
        """
        super().__init__(agent)
        self.config = config or {}
        self._client: LLMClient | None = None
        self._defaults: dict[str, Any] = {}

    def on_start(self) -> None:
        """Create LLM client on agent start."""
        self._client = LLMClientFactory(self.agent.lg).from_config(self.config)
        self._defaults = _resolve_llm_defaults(self.config)

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
        output_schema: type[BaseModel] | None = None,
    ) -> CompletionResult:
        """Generate a completion.

        Args:
            messages: Conversation messages.
            model: Model override (uses config default if None).
            temperature: Temperature override (uses config default if None).
            max_tokens: Max tokens override (uses config default if None).
            tools: Tool definitions for function calling.
            output_schema: Pydantic model class for structured output. When provided,
                the LLM is instructed to return JSON matching the schema, and the
                response is validated. Result.parsed will contain the validated object.

        Returns:
            CompletionResult with content and metadata. If output_schema was provided,
            result.parsed contains the validated Pydantic object.

        Raises:
            ValueError: If both tools and output_schema are provided.
            StructuredOutputError: If JSON parsing or schema validation fails.
        """
        if tools and output_schema:
            raise ValueError("Cannot use both tools and output_schema")

        api_messages = self._messages_to_dicts(messages)
        extra_body = None

        if output_schema:
            api_messages = self._inject_schema_prompt(api_messages, output_schema)
            extra_body = {"response_format": {"type": "json_object"}}

        response = self.client.chat_full(
            messages=api_messages,
            model=model or self._defaults.get("model"),
            temperature=temperature
            if temperature is not None
            else self._defaults.get("temperature", 0.7),
            max_tokens=max_tokens if max_tokens is not None else self._defaults.get("max_tokens"),
            tools=tools,
            extra_body=extra_body,
        )

        result = self._response_to_result(response)

        # Parse and validate structured output
        if output_schema:
            result.parsed = self._parse_structured_output(result.content, output_schema)

        return result

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
            model=response.model or self._defaults.get("model", "unknown"),
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

    def _build_schema_prompt(self, schema: type[BaseModel]) -> str:
        """Generate prompt instructing LLM to output JSON matching schema."""
        json_schema = schema.model_json_schema()
        return (
            "You must respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            "Respond ONLY with the JSON object, no other text."
        )

    def _inject_schema_prompt(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Inject schema instruction into messages.

        Appends schema prompt to existing system message or creates one.
        """
        schema_prompt = self._build_schema_prompt(schema)
        result = list(messages)  # shallow copy

        if result and result[0].get("role") == "system":
            # Append to existing system message
            result[0] = {
                **result[0],
                "content": f"{result[0]['content']}\n\n{schema_prompt}",
            }
        else:
            # Insert new system message at the start
            result.insert(0, {"role": "system", "content": schema_prompt})

        return result

    def _parse_structured_output(
        self,
        content: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Parse JSON content and validate against schema.

        Args:
            content: Raw JSON string from LLM response.
            schema: Pydantic model class to validate against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            StructuredOutputError: If JSON is invalid or doesn't match schema.
        """
        from ...llm.json_cleaner import JSONCleaner

        try:
            cleaner = JSONCleaner()
            cleaned = cleaner.clean(content)
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            self._agent._lg.warning(
                "failed to parse LLM JSON output",
                extra={"raw_content": content, "error": str(e)},
            )
            raise StructuredOutputError(f"Invalid JSON in response: {e}") from e

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise StructuredOutputError(f"Response doesn't match schema: {e}") from e


class LLMTraitBackend:
    """Adapter that wraps LLMTrait to satisfy LLMBackend protocol.

    Provides an LLMBackend-compatible interface for code that needs it.

    Example:
        llm_trait = agent.require_trait(LLMTrait)
        backend = LLMTraitBackend(llm_trait)
    """

    def __init__(self, llm_trait: LLMTrait) -> None:
        self._trait = llm_trait

    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Delegate to LLMTrait.complete()."""
        return self._trait.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

    def load_adapter(self, adapter_path: str) -> None:
        """Not supported through trait adapter."""
        raise NotImplementedError("Adapter loading not supported via trait")

    def unload_adapter(self) -> None:
        """Not supported through trait adapter."""
        raise NotImplementedError("Adapter unloading not supported via trait")
