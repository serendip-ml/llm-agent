"""LLM backend abstraction."""

from llm_agent.llm.backend import HTTPBackend, LLMBackend, LLMError, StructuredOutputError
from llm_agent.llm.types import CompletionResult, Message


__all__ = [
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMError",
    "Message",
    "StructuredOutputError",
]
