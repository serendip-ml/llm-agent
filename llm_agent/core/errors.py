"""Core exceptions for llm-agent."""


class AgentError(Exception):
    """Base exception for all llm-agent errors."""


class ConfigError(AgentError):
    """Configuration is missing or invalid."""


class TraitError(AgentError):
    """Base for trait-related errors."""


class TraitNotFoundError(TraitError):
    """Required trait not found in registry."""


class TraitAlreadyRegisteredError(TraitError):
    """Trait already registered in registry."""
