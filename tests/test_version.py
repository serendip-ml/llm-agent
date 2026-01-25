"""Version test."""

import llm_agent


def test_version() -> None:
    """Verify version is defined."""
    assert llm_agent.__version__ == "0.0.0"
