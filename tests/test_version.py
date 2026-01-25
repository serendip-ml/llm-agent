"""Version test."""

import pytest

import llm_agent


pytestmark = pytest.mark.unit


def test_version() -> None:
    """Verify version is defined."""
    assert llm_agent.__version__ == "0.0.0"
