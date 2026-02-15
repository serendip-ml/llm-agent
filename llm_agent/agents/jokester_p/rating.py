"""Rating services for jokester-p agent.

Provides:
- InlineRater: For rating jokes during generation (uses RatingTrait)
- RatingService: For CLI batch rating outside agent context
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text

from llm_agent.core.memory.rating import (
    BatchItem,
    BatchRequest,
    Result,
)
from llm_agent.core.memory.rating import (
    Service as CoreRatingService,
)


if TYPE_CHECKING:
    from appinfra.db.pg import PG
    from appinfra.log import Logger

    from ...core.llm import LLMCaller
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


class BatchRater:
    """Batches jokes for efficient batch rating.

    Instead of rating each joke immediately (one LLM call per joke),
    this rater queues jokes and flushes when the batch is full.
    Reduces LLM calls by ~5x with default batch_size=5.

    Usage:
        rater = BatchRater(lg, rating_trait, batch_size=5)

        # During generation
        rater.queue(fact_id, joke_text)  # Auto-flushes when batch is full

        # On agent stop (rate partial batch)
        rater.flush()
    """

    def __init__(
        self,
        lg: Logger,
        rating_trait: RatingTrait,
        batch_size: int,
        fact_type: str = "solution",
        category: str = "joke",
    ) -> None:
        """Initialize batch rater.

        Args:
            lg: Logger instance.
            rating_trait: RatingTrait for batch rating operations.
            batch_size: Number of items to queue before auto-flushing.
            fact_type: Type of facts (default: "solution").
            category: Category for preference pairing (default: "joke").
        """
        self._lg = lg
        self._rating = rating_trait
        self._batch_size = batch_size
        self._fact_type = fact_type
        self._category = category
        self._queue: list[tuple[int, str]] = []

    def queue(self, fact_id: int, content: str) -> None:
        """Queue a fact for rating. Flushes when batch is full.

        Args:
            fact_id: ID of the saved joke fact.
            content: Text of the joke to rate.
        """
        self._queue.append((fact_id, content))
        self._lg.debug(
            "queued for batch rating",
            extra={"fact_id": fact_id, "queue_size": len(self._queue)},
        )
        if len(self._queue) >= self._batch_size:
            self.flush()

    def flush(self) -> int:
        """Rate all queued items. Returns count rated.

        Returns:
            Number of items successfully rated.
        """
        if not self._queue:
            return 0

        items = self._queue[:]
        self._queue.clear()

        try:
            rated = self._rating.rate_items(items, self._fact_type, self._category)
            self._log_batch_results(items, rated)
            return rated
        except Exception as e:
            self._lg.warning("batch rating failed", extra={"exception": e})
            return 0

    def _log_batch_results(self, items: list[tuple[int, str]], rated: int) -> None:
        """Log batch completion and individual ratings."""
        self._lg.debug(
            "flushed batch for rating",
            extra={"count": len(items), "rated": rated},
        )

        for fact_id, content in items:
            stars = self._rating.get_fact_rating(fact_id)
            if stars:
                stars_visual = "★" * stars + "☆" * (5 - stars)
                joke_preview = content[:80] + "..." if len(content) > 80 else content
                self._lg.info(
                    f"rated: {stars_visual}",
                    extra={"fact_id": fact_id, "stars": stars, "joke": joke_preview},
                )

    @property
    def pending_count(self) -> int:
        """Number of items currently queued."""
        return len(self._queue)


@dataclass
class UnratedJoke:
    """An unrated joke from the database."""

    id: int
    content: str


@dataclass
class RatedJoke:
    """A rated joke with its result."""

    id: int
    content: str
    stars: int


@dataclass
class BatchRatingResult:
    """Result of batch rating operation."""

    rated: int
    failed: int
    batches: int
    ratings: list[RatedJoke]


class RatingService:
    """Service for batch rating jokes via CLI.

    Uses the core rating service but provides a simpler interface
    for CLI usage without requiring full agent setup.
    """

    def __init__(
        self,
        lg: Logger,
        pg: PG,
        llm_caller: LLMCaller,
        prompt_template: str,
        context_key: str,
        batch_size: int = 5,
        provider: str = "local",
        model: str = "auto",
    ) -> None:
        self._lg = lg
        self._pg = pg
        self._context_key = context_key
        self._core_service = CoreRatingService(lg, llm_caller)
        self._prompt_template = prompt_template
        self._batch_size = batch_size
        self._provider = provider
        self._model = model

    def get_unrated_jokes(self, limit: int | None = None) -> list[UnratedJoke]:
        """Get unrated jokes from the database."""
        sql = text("""
            SELECT af.id, af.content
            FROM atomic_facts af
            LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key
              AND af.type = 'solution'
              AND afd.id IS NULL
            ORDER BY af.created_at ASC
            LIMIT :limit
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(
                sql, {"context_key": self._context_key, "limit": limit or 10000}
            ).fetchall()
        return [UnratedJoke(id=r[0], content=r[1]) for r in rows]

    def rate_batches(
        self, limit: int | None = None, batch_size: int | None = None
    ) -> Generator[list[RatedJoke], None, None]:
        """Rate unrated jokes, yielding results batch by batch.

        Args:
            limit: Maximum jokes to rate (None = all).
            batch_size: Items per batch (None = use configured default).

        Yields:
            List of RatedJoke for each batch.
        """
        jokes = self.get_unrated_jokes(limit)
        if not jokes:
            return

        size = batch_size or self._batch_size

        for i in range(0, len(jokes), size):
            batch = jokes[i : i + size]
            batch_ratings, failed = self._rate_batch(batch)

            self._lg.debug(
                "batch completed",
                extra={"rated": len(batch_ratings), "failed": failed},
            )

            yield batch_ratings

    def _rate_batch(self, jokes: list[UnratedJoke]) -> tuple[list[RatedJoke], int]:
        """Rate a single batch. Returns (rated_jokes, failed_count)."""
        items = [BatchItem(fact=j.id, content=j.content) for j in jokes]
        joke_map = {j.id: j.content for j in jokes}
        request = BatchRequest(
            items=items,
            prompt_template=self._prompt_template,
            model=self._model,
            provider=self._provider,
            backend=self._provider,  # Provider name matches backend name
        )

        try:
            results = self._core_service.rate_batch(request)
            rated: list[RatedJoke] = []
            for result in results:
                if self._save_rating(result):
                    rated.append(
                        RatedJoke(
                            id=result.fact,
                            content=joke_map.get(result.fact, ""),
                            stars=result.stars,
                        )
                    )
            return (rated, len(items) - len(rated))
        except Exception as e:
            self._lg.warning("batch rating failed", extra={"exception": e})
            return ([], len(items))

    def _save_rating(self, result: Result) -> bool:
        """Save a rating to the database."""
        sql = text("""
            INSERT INTO atomic_feedback_details
                (fact_id, signal, strength, context, provider)
            VALUES (:fact_id, :signal, :strength, CAST(:context AS jsonb), :provider)
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
