"""Novelty checking for joke generation.

Provides embedding-based similarity checking to ensure jokes are never repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_kelt.memory.atomic import EmbeddingFilter, Fact
from sqlalchemy import or_, select

from .schema import TrainingMetadata


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

    Supports reference model isolation: when a reference model (e.g., Haiku) is
    configured, novelty checking stays within the same "space" - reference model
    jokes only compare against other reference model jokes, and local model jokes
    only compare against other local model jokes.
    """

    def __init__(
        self,
        lg: Logger,
        learn_trait: LearnTrait,
        similarity_threshold: float = 0.85,
        reference_model: str | None = None,
    ) -> None:
        """Initialize novelty checker.

        Args:
            lg: Logger instance.
            learn_trait: Learn trait for RAG recall.
            similarity_threshold: Min similarity for duplicate (0.0-1.0).
            reference_model: Model name pattern for reference models (e.g., "claude-haiku-4-5").
        """
        self._lg = lg
        self._learn = learn_trait
        self._threshold = similarity_threshold
        self._reference_model = reference_model

    @property
    def has_embedder(self) -> bool:
        """Whether embedder is available for novelty checking."""
        return self._learn.has_embedder

    def check(self, joke_text: str, current_model: str | None = None) -> NoveltyCheck:
        """Check if joke is novel using embedding similarity.

        Args:
            joke_text: The joke text to check.
            current_model: The model generating this joke (for reference model filtering).

        Returns:
            NoveltyCheck with is_novel, similarity score, and closest match.
        """
        self._lg.debug(
            "checking joke novelty...", extra={"joke": joke_text, "model": current_model}
        )
        try:
            embedding_filter = self._build_filter(current_model)
            similar_facts = self._learn.recall(
                query=joke_text, top_k=1, categories=["joke"], embedding_filter=embedding_filter
            )

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

    def _build_filter(self, current_model: str | None) -> EmbeddingFilter | None:
        """Build embedding filter for reference model isolation."""
        if not self._reference_model:
            return None
        if not current_model:
            # Reference model configured but current_model not provided - this shouldn't
            # happen in normal usage since current_model comes from LLM response
            self._lg.warning(
                "reference model isolation configured but current_model not provided, "
                "skipping filter"
            )
            return None

        is_reference = self._reference_model in current_model
        if is_reference:
            return self._build_reference_model_filter()
        return self._build_local_model_filter()

    def _build_reference_model_filter(self) -> EmbeddingFilter:
        """Build filter for reference model: only compare against reference model jokes."""
        subquery = select(TrainingMetadata.fact_id).where(
            TrainingMetadata.base_model.ilike(f"%{self._reference_model}%")
        )
        self._lg.debug("filtering to reference model jokes only")
        return EmbeddingFilter().where(Fact.id.in_(subquery))

    def _build_local_model_filter(self) -> EmbeddingFilter:
        """Build filter for local model: exclude reference jokes, include orphaned facts."""
        ref_model_facts = select(TrainingMetadata.fact_id).where(
            TrainingMetadata.base_model.ilike(f"%{self._reference_model}%")
        )
        all_facts_with_metadata = select(TrainingMetadata.fact_id)
        self._lg.debug("filtering to exclude reference model jokes")
        # Include facts that: (1) have non-reference metadata, OR (2) have no metadata
        return EmbeddingFilter().where(
            or_(Fact.id.not_in(ref_model_facts), ~Fact.id.in_(all_facts_with_metadata))
        )

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
