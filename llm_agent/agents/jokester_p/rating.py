"""Rating services for jokester-p agent.

Provides:
- InlineRater: For rating jokes during generation (uses RatingTrait)
- RatingService: For CLI batch rating outside agent context
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text

from llm_agent.core.memory.rating import BatchItem, BatchRequest, Result
from llm_agent.core.memory.rating import Service as CoreRatingService


if TYPE_CHECKING:
    from appinfra.db.pg import PG
    from appinfra.log import Logger
    from llm_infer.client import LLMRouter

    from ...core.traits.builtin.rating import RatingTrait


class InlineRater:
    """Rates jokes during generation using RatingTrait."""

    def __init__(self, lg: Logger, rating_trait: RatingTrait) -> None:
        """Initialize inline rater.

        Args:
            lg: Logger instance.
            rating_trait: RatingTrait for rating operations.
        """
        self._lg = lg
        self._rating = rating_trait

    def rate(self, fact_id: int, joke_text: str) -> None:
        """Rate a joke using all configured providers.

        Args:
            fact_id: ID of the saved joke fact.
            joke_text: Text of the joke to rate.
        """
        try:
            results = self._rating.rate_fact_with_all_providers(
                fact_id=fact_id, content=joke_text, fact_type="solution", category="joke"
            )
            for result in results:
                self._log_rating(fact_id, result)
        except Exception as e:
            self._lg.warning("inline rating failed", extra={"exception": e, "fact_id": fact_id})

    def _log_rating(self, fact_id: int, result: Result) -> None:
        """Log a single rating result."""
        stars_visual = "★" * result.stars + "☆" * (5 - result.stars)
        reasoning = (
            result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning
        )
        self._lg.info(
            f"inline rating: {stars_visual}",
            extra={
                "fact_id": fact_id,
                "stars": result.stars,
                "model": result.model,
                "reasoning": reasoning,
            },
        )


@dataclass
class UnratedJoke:
    """An unrated joke from the database."""

    id: int
    content: str


@dataclass
class BatchRatingResult:
    """Result of batch rating operation."""

    rated: int
    failed: int
    batches: int


class RatingService:
    """Service for batch rating jokes via CLI.

    Uses the core rating service but provides a simpler interface
    for CLI usage without requiring full agent setup.
    """

    CONTEXT_KEY = "jokester-p"

    def __init__(
        self,
        lg: Logger,
        pg: PG,
        llm_client: LLMRouter,
        prompt_template: str,
        batch_size: int = 5,
    ) -> None:
        self._lg = lg
        self._pg = pg
        self._core_service = CoreRatingService(lg, llm_client)
        self._prompt_template = prompt_template
        self._batch_size = batch_size

    def get_unrated_jokes(self, limit: int | None = None) -> list[UnratedJoke]:
        """Get unrated jokes from the database."""
        sql = text("""
            SELECT af.id, af.content
            FROM atomic_facts af
            LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key
              AND af.type = 'solution'
              AND afd.id IS NULL
            ORDER BY af.created_at DESC
            LIMIT :limit
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(
                sql, {"context_key": self.CONTEXT_KEY, "limit": limit or 10000}
            ).fetchall()
        return [UnratedJoke(id=r[0], content=r[1]) for r in rows]

    def rate_all(
        self, limit: int | None = None, batch_size: int | None = None
    ) -> BatchRatingResult:
        """Rate all unrated jokes in batches.

        Args:
            limit: Maximum jokes to rate (None = all).
            batch_size: Items per batch (None = use configured default).

        Returns:
            BatchRatingResult with counts.
        """
        jokes = self.get_unrated_jokes(limit)
        if not jokes:
            return BatchRatingResult(rated=0, failed=0, batches=0)

        size = batch_size or self._batch_size
        total_rated = 0
        total_failed = 0
        batch_count = 0

        for i in range(0, len(jokes), size):
            batch = jokes[i : i + size]
            batch_count += 1

            rated, failed = self._rate_batch(batch)
            total_rated += rated
            total_failed += failed

            self._lg.info(
                "batch completed",
                extra={"batch": batch_count, "rated": rated, "failed": failed},
            )

        return BatchRatingResult(rated=total_rated, failed=total_failed, batches=batch_count)

    def _rate_batch(self, jokes: list[UnratedJoke]) -> tuple[int, int]:
        """Rate a single batch. Returns (rated, failed)."""
        items = [BatchItem(fact=j.id, content=j.content) for j in jokes]
        request = BatchRequest(
            items=items,
            prompt_template=self._prompt_template,
            model="auto",
            provider="cli_batch_rating",
        )

        try:
            results = self._core_service.rate_batch(request)
            saved = 0
            for result in results:
                if self._save_rating(result):
                    saved += 1
            return (saved, len(items) - saved)
        except Exception as e:
            self._lg.warning("batch rating failed", extra={"exception": e})
            return (0, len(items))

    def _save_rating(self, result: Result) -> bool:
        """Save a rating to the database."""
        sql = text("""
            INSERT INTO atomic_feedback_details
                (fact_id, signal, strength, context, provider, created_at)
            VALUES (:fact_id, :signal, :strength, :context::jsonb, :provider, NOW())
        """)
        params = {
            "fact_id": result.fact,
            "signal": result.signal,
            "strength": result.strength,
            "context": json.dumps({"stars": result.stars, "reasoning": result.reasoning}),
            "provider": result.provider,
        }
        try:
            with self._pg.connect() as conn:
                conn.execute(sql, params)
                conn.commit()
            return True
        except Exception as e:
            self._lg.warning(
                "failed to save rating", extra={"exception": e, "fact_id": result.fact}
            )
            return False
