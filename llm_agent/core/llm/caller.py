"""Framework-level LLM wrapper with logging and dry-run support.

Sits between llm-infer client and the rest of the framework:
    llm-infer/LLMRouter -> llm-agent/LLMCaller -> llm-agent/LLMTrait (agents)
                                               -> RatingService (CLI)

Provides:
- Trace logging of full request/response
- Dry-run mode (log without sending)
- Single place for framework-level LLM concerns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from appinfra.log import Logger
    from llm_infer.client import ChatResponse, LLMRouter


@dataclass
class CallResult:
    """Result of an LLM call."""

    content: str
    model: str
    usage: dict[str, int] | None
    tool_calls: list[dict[str, Any]] | None
    adapter_fallback: bool
    dry_run: bool  # True if this was a dry-run (no actual LLM call)
    raw_response: ChatResponse | None  # None in dry-run mode


class LLMCaller:
    """Framework wrapper for LLM calls with logging and dry-run.

    Wraps llm-infer's LLMRouter to add framework-level concerns:
    - Trace logging of full request/response (for debugging)
    - Dry-run mode (log what would be sent without calling LLM)
    - Consistent interface for both agent and CLI contexts

    Example:
        from llm_infer.client import Factory as LLMClientFactory

        router = LLMClientFactory(lg).from_config(config)
        caller = LLMCaller(lg, router)

        # Normal call
        result = caller.chat(messages=[{"role": "user", "content": "Hello"}])

        # Dry-run (logs but doesn't call LLM)
        caller_dry = LLMCaller(lg, router, dry_run=True)
        result = caller_dry.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        lg: Logger,
        router: LLMRouter,
        dry_run: bool = False,
    ) -> None:
        """Initialize LLM caller.

        Args:
            lg: Logger instance.
            router: LLM router from llm-infer.
            dry_run: If True, log requests but don't send to LLM.
        """
        self._lg = lg
        self._router = router
        self._dry_run = dry_run

    @property
    def router(self) -> LLMRouter:
        """Access underlying router (for compatibility)."""
        return self._router

    @property
    def dry_run(self) -> bool:
        """Whether dry-run mode is enabled."""
        return self._dry_run

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        backend: str | None = None,
        adapter_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CallResult:
        """Call LLM with logging and optional dry-run.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model to use (None = router default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions in OpenAI format.
            backend: Specific backend to use.
            adapter_id: LoRA adapter ID.
            extra_body: Extra request body parameters.

        Returns:
            CallResult with response content and metadata.
        """
        resolved = self._router.resolve(model=model, backend=backend)
        self._log_request(messages, resolved, temperature, max_tokens, adapter_id, tools)

        if self._dry_run:
            return self._dry_run_result(model)

        response = self._router.chat_full(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            backend=backend,
            adapter_id=adapter_id,
            extra_body=extra_body,
        )

        self._log_response(response)
        return self._to_result(response)

    def _dry_run_result(self, model: str | None) -> CallResult:
        """Return stub result for dry-run mode."""
        self._lg.info("dry-run: skipping LLM call")
        return CallResult(
            content="[dry-run: no response]",
            model=model or "dry-run",
            usage=None,
            tool_calls=None,
            adapter_fallback=False,
            dry_run=True,
            raw_response=None,
        )

    def _to_result(self, response: ChatResponse) -> CallResult:
        """Convert ChatResponse to CallResult."""
        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response.tool_calls
            ]

        return CallResult(
            content=response.content or "",
            model=getattr(response, "model", None) or "unknown",
            usage=getattr(response, "usage", None),
            tool_calls=tool_calls,
            adapter_fallback=getattr(response, "adapter_fallback", False),
            dry_run=False,
            raw_response=response,
        )

    def _log_request(
        self,
        messages: list[dict[str, Any]],
        resolved: Any,
        temperature: float,
        max_tokens: int | None,
        adapter_id: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Log full LLM request at trace level."""
        self._lg.trace(
            "llm request",
            extra={
                "backend": resolved.backend,
                "model": resolved.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "adapter_id": adapter_id,
                "messages": messages,
                "tools": [t.get("function", {}).get("name") for t in (tools or [])],
                "dry_run": self._dry_run,
            },
        )

    def _log_response(self, response: ChatResponse) -> None:
        """Log full LLM response at trace level."""
        self._lg.trace(
            "llm response",
            extra={
                "content": response.content,
                "model": getattr(response, "model", None),
                "usage": getattr(response, "usage", None),
                "tool_calls": [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (response.tool_calls or [])
                ],
                "adapter_fallback": getattr(response, "adapter_fallback", False),
            },
        )

    def close(self) -> None:
        """Close the underlying router."""
        self._router.close()
