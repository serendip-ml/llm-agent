"""Preference pairing service for jokester-p agent.

Creates preference pairs from rated jokes for DPO training.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text


if TYPE_CHECKING:
    from appinfra.db.pg import PG
    from appinfra.log import Logger


@dataclass
class RatedJoke:
    """A rated joke."""

    id: int
    content: str
    stars: int


@dataclass
class PreferencePair:
    """A preference pair for DPO training."""

    chosen: RatedJoke
    rejected: RatedJoke

    @property
    def margin(self) -> int:
        """Star difference between chosen and rejected."""
        return self.chosen.stars - self.rejected.stars


@dataclass
class PairingResult:
    """Result of a pairing operation."""

    pairs: list[PreferencePair]
    total_rated: int
    strategy: str
    min_gap: int


class PairingService:
    """Service for creating preference pairs from rated jokes.

    Supports two pairing strategies:
    - relative: Pair highest with lowest, respecting min gap
    - threshold: Pair based on fixed star thresholds
    """

    def __init__(self, lg: Logger, pg: PG, context_key: str) -> None:
        self._lg = lg
        self._pg = pg
        self._context_key = context_key

    def get_rated_jokes(self) -> list[RatedJoke]:
        """Get all rated jokes sorted by stars (desc)."""
        sql = text("""
            SELECT
                af.id,
                af.content,
                (afd.context->>'stars')::int as stars
            FROM atomic_facts af
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key
              AND afd.context->>'stars' IS NOT NULL
            ORDER BY stars DESC, af.id
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, {"context_key": self._context_key}).fetchall()
        return [RatedJoke(id=r[0], content=r[1], stars=r[2]) for r in rows]

    def create_pairs(
        self,
        strategy: str = "relative",
        min_gap: int = 1,
        high_threshold: int = 4,
        low_threshold: int = 2,
    ) -> PairingResult:
        """Create preference pairs from rated jokes.

        Args:
            strategy: "relative" or "threshold"
            min_gap: Minimum star difference for relative pairing
            high_threshold: Minimum stars for chosen (threshold strategy)
            low_threshold: Maximum stars for rejected (threshold strategy)

        Returns:
            PairingResult with pairs and metadata.
        """
        rated = self.get_rated_jokes()

        if strategy == "relative":
            pairs = self._pair_relative(rated, min_gap)
        else:
            pairs = self._pair_threshold(rated, high_threshold, low_threshold)

        return PairingResult(
            pairs=pairs,
            total_rated=len(rated),
            strategy=strategy,
            min_gap=min_gap,
        )

    def save_pairs(self, pairs: list[PreferencePair]) -> int:
        """Save preference pairs to database.

        Returns:
            Number of pairs created.
        """
        created = 0
        for pair in pairs:
            try:
                if self._pair_exists(pair.chosen.id, pair.rejected.id):
                    self._lg.debug(
                        "pair already exists",
                        extra={"chosen_id": pair.chosen.id, "rejected_id": pair.rejected.id},
                    )
                    continue

                fact_id = self._create_preference_fact()
                self._create_preference_details(fact_id, pair)
                created += 1
            except Exception as e:
                self._lg.warning(
                    "failed to create pair",
                    extra={
                        "exception": e,
                        "chosen_id": pair.chosen.id,
                        "rejected_id": pair.rejected.id,
                    },
                )

        return created

    def _pair_relative(self, rated: list[RatedJoke], min_gap: int) -> list[PreferencePair]:
        """Pair highest with lowest, respecting min gap."""
        pairs = []
        used: set[int] = set()

        sorted_jokes = sorted(rated, key=lambda x: (-x.stars, x.id))

        for chosen in sorted_jokes:
            if chosen.id in used:
                continue

            for rejected in reversed(sorted_jokes):
                if rejected.id in used or rejected.id == chosen.id:
                    continue
                if chosen.stars - rejected.stars >= min_gap:
                    pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
                    used.add(chosen.id)
                    used.add(rejected.id)
                    break

        return pairs

    def _pair_threshold(self, rated: list[RatedJoke], high: int, low: int) -> list[PreferencePair]:
        """Pair based on fixed thresholds."""
        chosen_pool = [j for j in rated if j.stars >= high]
        rejected_pool = [j for j in rated if j.stars <= low]

        pairs = []
        for c, r in zip(chosen_pool, rejected_pool, strict=False):
            pairs.append(PreferencePair(chosen=c, rejected=r))

        return pairs

    def _pair_exists(self, chosen_id: int, rejected_id: int) -> bool:
        """Check if a preference pair already exists for these facts."""
        sql = text("""
            SELECT 1 FROM atomic_preference_details
            WHERE metadata->>'chosen_fact_id' = :chosen_id
               OR metadata->>'rejected_fact_id' = :rejected_id
            LIMIT 1
        """)
        with self._pg.connect() as conn:
            result = conn.execute(
                sql, {"chosen_id": str(chosen_id), "rejected_id": str(rejected_id)}
            ).fetchone()
        return result is not None

    def _create_preference_fact(self) -> int:
        """Create atomic_fact entry for preference."""
        content = "preference_pair"
        content_hash = uuid.uuid4().hex

        sql = text("""
            INSERT INTO atomic_facts
                (context_key, type, content, content_hash, category, source, confidence, active, created_at)
            VALUES
                (:context_key, 'preference', :content, :content_hash, 'joke', 'pairs_sync', 1.0, true, NOW())
            RETURNING id
        """)
        with self._pg.connect() as conn:
            result = conn.execute(
                sql,
                {
                    "context_key": self._context_key,
                    "content": content,
                    "content_hash": content_hash,
                },
            ).fetchone()
            conn.commit()
        return int(result[0])

    def _create_preference_details(self, fact_id: int, pair: PreferencePair) -> None:
        """Create atomic_preference_details entry."""
        metadata = json.dumps(
            {
                "chosen_fact_id": pair.chosen.id,
                "rejected_fact_id": pair.rejected.id,
                "chosen_stars": pair.chosen.stars,
                "rejected_stars": pair.rejected.stars,
            }
        )

        sql = text("""
            INSERT INTO atomic_preference_details
                (fact_id, context, chosen, rejected, margin, metadata)
            VALUES
                (:fact_id, :context, :chosen, :rejected, :margin, CAST(:metadata AS jsonb))
        """)
        with self._pg.connect() as conn:
            conn.execute(
                sql,
                {
                    "fact_id": fact_id,
                    "context": "Tell me a joke.",
                    "chosen": pair.chosen.content,
                    "rejected": pair.rejected.content,
                    "margin": pair.margin,
                    "metadata": metadata,
                },
            )
            conn.commit()
