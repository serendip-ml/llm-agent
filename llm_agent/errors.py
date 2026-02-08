"""Custom errors for llm-agent."""


class AgentError(Exception):
    """Base exception for all llm-agent errors."""

    pass


class ConfigError(AgentError):
    """Raised when configuration is missing or invalid."""

    pass
