"""Data models for rating service."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from appinfra import DotDict


class ProviderType(StrEnum):
    """Type of rating provider."""

    LLM = "llm"
    MANUAL = "manual"


@dataclass
class Criteria:
    """Rating criteria for evaluating content."""

    name: str
    description: str
    weight: float = 1.0


@dataclass
class Request:
    """Request to rate content using LLM."""

    fact: Any  # Backend-specific identifier (e.g., fact_id for atomic facts)
    content: str
    prompt_template: str
    criteria: list[Criteria]
    model: str
    provider: str
    temperature: float = 0.3


@dataclass
class Result:
    """Result from rating."""

    fact: Any  # Backend-specific identifier (e.g., fact_id for atomic facts)
    signal: Literal["positive", "negative", "dismiss"]
    strength: float
    stars: int
    criteria_scores: dict[str, float]
    reasoning: str
    provider_type: ProviderType
    model: str | None  # LLM model name (None for manual ratings)
    provider: str  # Provider identifier


@dataclass
class ProviderConfig:
    """Configuration for a rating provider."""

    provider_type: ProviderType
    model: str
    backend: DotDict
    enabled: bool = True


@dataclass
class CriteriaConfig:
    """Configuration for rating criteria for a specific fact type."""

    fact_type: str  # e.g., "solution", "prediction", "feedback"
    prompt: str  # Type-specific base prompt
    criteria: list[Criteria]
