"""Version test."""

import pytest

import llm_gent


pytestmark = pytest.mark.unit


def test_version() -> None:
    """Verify version is defined."""
    assert llm_gent.__version__ == "0.0.0"
