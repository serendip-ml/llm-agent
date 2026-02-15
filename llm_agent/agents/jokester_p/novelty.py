"""Novelty checking for joke generation.

Provides embedding-based similarity checking to ensure jokes are never repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ...core.traits.builtin.learn import LearnTrait


@dataclass
class NoveltyCheck:
    """Result of novelty checking against existing jokes."""

    is_novel: bool
    max_similarity: float
    similar_joke: str | None


class NoveltyChecker:
    """Checks joke novelty using embedding similarity.

    Uses RAG (recall) to find similar existing jokes and compares
    similarity scores against a threshold to determine novelty.
    Fails closed: if the check fails, the joke is rejected to
    maintain the "never repeat" guarantee.
    """

    def __init__(
        self, lg: Logger, learn_trait: LearnTrait, similarity_threshold: float = 0.85
    ) -> None:
        """Initialize novelty checker.

        Args:
            lg: Logger instance.
            learn_trait: Learn trait for RAG recall.
            similarity_threshold: Min similarity for duplicate (0.0-1.0).
        """
        self._lg = lg
        self._learn = learn_trait
        self._threshold = similarity_threshold

    @property
    def has_embedder(self) -> bool:
        """Whether embedder is available for novelty checking."""
        return self._learn.has_embedder

    def check(self, joke_text: str) -> NoveltyCheck:
        """Check if joke is novel using embedding similarity.

        Args:
            joke_text: The joke text to check.

        Returns:
            NoveltyCheck with is_novel, similarity score, and closest match.
        """
        self._lg.debug("checking joke novelty...", extra={"joke": joke_text})
        try:
            similar_facts = self._learn.recall(query=joke_text, top_k=1, categories=["joke"])

            if not similar_facts:
                self._lg.debug("joke is novel (no existing jokes)", extra={"joke": joke_text})
                return NoveltyCheck(is_novel=True, max_similarity=0.0, similar_joke=None)

            return self._evaluate_similarity(similar_facts[0])

        except Exception as e:
            self._lg.warning(
                "novelty check failed, rejecting joke to maintain never-repeat guarantee",
                extra={"exception": e, "joke": joke_text},
            )
            return NoveltyCheck(is_novel=False, max_similarity=1.0, similar_joke=None)

    def _evaluate_similarity(self, closest: Any) -> NoveltyCheck:
        """Evaluate similarity of joke against closest match."""
        if closest.score >= self._threshold:
            self._log_result(closest, is_novel=False)
            return NoveltyCheck(
                is_novel=False, max_similarity=closest.score, similar_joke=closest.entity.content
            )

        self._log_result(closest, is_novel=True)
        return NoveltyCheck(
            is_novel=True, max_similarity=closest.score, similar_joke=closest.entity.content
        )

    def _log_result(self, similar: Any, is_novel: bool) -> None:
        """Log similarity check result."""
        msg = "joke is novel" if is_novel else "joke too similar"
        self._lg.debug(msg, extra={"similarity": similar.score, "existing": similar.entity.content})
