"""Shared helper functions for the governor module."""

from __future__ import annotations


def truncate_str(s: str | None, max_len: int = 200) -> str:
    """Truncate string if it exceeds max length.

    Args:
        s: String to truncate (None returns empty string).
        max_len: Maximum length before truncation.

    Returns:
        Original string if within limit, or truncated with "..." suffix.
    """
    if not s:
        return ""
    return s[:max_len] + "..." if len(s) > max_len else s
