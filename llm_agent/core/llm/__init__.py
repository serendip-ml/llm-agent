"""LLM backend abstraction."""

from llm_agent.core.llm.backend import HTTPBackend, LLMBackend, LLMError, StructuredOutputError
from llm_agent.core.llm.types import CompletionResult, Message


__all__ = [
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMError",
    "Message",
    "StructuredOutputError",
]
