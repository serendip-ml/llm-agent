"""Agent factory base class and configuration utilities."""

from __future__ import annotations

import os
import re
from typing import Any

from appinfra.log import Logger


class Factory:
    """Abstract base class for agent factories.

    Agent factories create Agent instances from configuration. Concrete
    implementations handle their own backend creation from llm_config.

    This class must be subclassed - it cannot be instantiated directly.
    Subclasses must implement the create() method.

    Example:
        class MyFactory(Factory):
            def create(self, config, variables=None):
                # Create backend from self._llm_config
                # Create and return agent
                ...
    """

    def __init__(self, lg: Logger, llm_config: dict[str, Any]) -> None:
        """Initialize factory.

        Args:
            lg: Logger instance.
            llm_config: LLM configuration dict (passed to subprocess).
        """
        self._lg = lg
        self._llm_config = llm_config

    def create(self, config: dict[str, Any], variables: dict[str, str] | None = None) -> Any:
        """Create an agent from configuration.

        Args:
            config: Agent configuration dictionary.
            variables: Optional variable substitutions.

        Returns:
            Configured Agent instance.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement create()")


# =============================================================================
# Configuration Helper Functions
# =============================================================================


def _substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Substitute {{VAR}} patterns in text with variable values.

    Falls back to environment variables if not in variables dict.
    """
    pattern = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return variables[var_name]
        if var_name in os.environ:
            return os.environ[var_name]
        raise ValueError(f"Variable {{{{{var_name}}}}} not found in variables or environment")

    return pattern.sub(replacer, text)


def _substitute_in_dict(data: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute variables in a dict/list structure."""
    if isinstance(data, str):
        return _substitute_variables(data, variables)
    if isinstance(data, dict):
        return {k: _substitute_in_dict(v, variables) for k, v in data.items()}
    if isinstance(data, list):
        return [_substitute_in_dict(item, variables) for item in data]
    return data
