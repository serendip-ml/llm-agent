"""Preference pairing service for jokester-p agent.

Creates preference pairs from rated jokes for DPO training.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
            SELECT DISTINCT ON (af.id)
                af.id,
                af.content,
                (afd.context->>'stars')::int as stars
            FROM atomic_facts af
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key
              AND afd.context->>'stars' IS NOT NULL
            ORDER BY af.id, afd.id DESC
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
        min_pairs: int | None = None,
        max_pairs: int | None = None,
    ) -> PairingResult:
        """Create preference pairs from rated jokes.

        Args:
            strategy: "relative" or "threshold"
            min_gap: Minimum star difference for relative pairing
            high_threshold: Minimum stars for chosen (threshold strategy)
            low_threshold: Maximum stars for rejected (threshold strategy)
            min_pairs: Minimum pairs to generate (reuses chosen jokes if needed)
            max_pairs: Maximum pairs to generate (caps output)

        Returns:
            PairingResult with pairs and metadata.
        """
        rated = self.get_rated_jokes()

        if strategy == "relative":
            pairs = self._pair_relative(rated, min_gap, min_pairs)
        else:
            pairs = self._pair_threshold(rated, high_threshold, low_threshold)

        # Apply max cap
        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

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

                self._create_pair_atomic(pair)
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

    def _create_pair_atomic(self, pair: PreferencePair, context: str = "Tell me a joke.") -> int:
        """Create preference fact and details in a single transaction."""
        with self._pg.connect() as conn:
            fact_id = self._insert_preference_fact(conn)
            self._insert_preference_details(conn, fact_id, pair, context)
            conn.commit()
        return fact_id

    def _insert_preference_fact(self, conn: Any) -> int:
        """Insert atomic_facts row for preference pair."""
        sql = text("""
            INSERT INTO atomic_facts
                (context_key, type, content, content_hash, category, source, confidence, active, created_at)
            VALUES (:context_key, 'preference', 'preference_pair', :hash, 'joke', 'pairs_sync', 1.0, true, NOW())
            RETURNING id
        """)
        result = conn.execute(
            sql, {"context_key": self._context_key, "hash": uuid.uuid4().hex}
        ).fetchone()
        return int(result[0])

    def _insert_preference_details(
        self, conn: Any, fact_id: int, pair: PreferencePair, context: str
    ) -> None:
        """Insert atomic_preference_details row."""
        metadata = json.dumps(
            {
                "chosen_fact_id": pair.chosen.id,
                "rejected_fact_id": pair.rejected.id,
                "chosen_stars": pair.chosen.stars,
                "rejected_stars": pair.rejected.stars,
            }
        )
        sql = text("""
            INSERT INTO atomic_preference_details (fact_id, context, chosen, rejected, margin, metadata)
            VALUES (:fact_id, :context, :chosen, :rejected, :margin, CAST(:metadata AS jsonb))
        """)
        conn.execute(
            sql,
            {
                "fact_id": fact_id,
                "context": context,
                "chosen": pair.chosen.content,
                "rejected": pair.rejected.content,
                "margin": pair.margin,
                "metadata": metadata,
            },
        )

    def _pair_relative(
        self, rated: list[RatedJoke], min_gap: int, min_pairs: int | None = None
    ) -> list[PreferencePair]:
        """Pair highest with lowest, respecting min gap.

        If min_pairs is set and 1:1 pairing yields fewer, reuses chosen jokes
        with multiple rejected jokes until min_pairs is reached or rejected
        pool is exhausted.
        """
        sorted_jokes = sorted(rated, key=lambda x: (-x.stars, x.id))
        chosen_pool, rejected_pool = self._build_pairing_pools(sorted_jokes, min_gap)

        if not chosen_pool or not rejected_pool:
            return []

        target = min_pairs if min_pairs else len(chosen_pool)
        return self._generate_pairs(chosen_pool, rejected_pool, min_gap, target)

    def _build_pairing_pools(
        self, sorted_jokes: list[RatedJoke], min_gap: int
    ) -> tuple[list[RatedJoke], list[RatedJoke]]:
        """Build chosen and rejected pools based on min_gap requirement."""
        chosen_pool = []
        rejected_pool = []

        for joke in sorted_jokes:
            max_rejected_stars = joke.stars - min_gap
            if any(j.stars <= max_rejected_stars for j in sorted_jokes if j.id != joke.id):
                chosen_pool.append(joke)
            min_chosen_stars = joke.stars + min_gap
            if any(j.stars >= min_chosen_stars for j in sorted_jokes if j.id != joke.id):
                rejected_pool.append(joke)

        # Sort: chosen by stars desc, rejected by stars asc (worst first)
        chosen_pool.sort(key=lambda x: (-x.stars, x.id))
        rejected_pool.sort(key=lambda x: (x.stars, x.id))
        return chosen_pool, rejected_pool

    def _generate_pairs(
        self,
        chosen_pool: list[RatedJoke],
        rejected_pool: list[RatedJoke],
        min_gap: int,
        target: int,
    ) -> list[PreferencePair]:
        """Generate pairs by cycling through chosen pool until target or exhausted."""
        pairs: list[PreferencePair] = []
        used_rejected: set[int] = set()
        chosen_idx = 0
        max_iterations = len(chosen_pool) * len(rejected_pool)

        while len(pairs) < target and len(used_rejected) < len(rejected_pool):
            chosen = chosen_pool[chosen_idx % len(chosen_pool)]

            for rejected in rejected_pool:
                if rejected.id not in used_rejected and chosen.stars - rejected.stars >= min_gap:
                    pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
                    used_rejected.add(rejected.id)
                    break

            chosen_idx += 1
            if chosen_idx >= max_iterations:
                break

        return pairs

    def _pair_threshold(self, rated: list[RatedJoke], high: int, low: int) -> list[PreferencePair]:
        """Pair based on fixed thresholds.

        Args:
            rated: List of rated jokes.
            high: Minimum stars for chosen pool (must be > low).
            low: Maximum stars for rejected pool.

        Raises:
            ValueError: If high <= low (would allow same joke in both pools).
        """
        if high <= low:
            raise ValueError(f"high threshold ({high}) must be greater than low ({low})")

        chosen_pool = [j for j in rated if j.stars >= high]
        chosen_ids = {j.id for j in chosen_pool}
        # Exclude jokes already in chosen pool to prevent same joke as both
        rejected_pool = [j for j in rated if j.stars <= low and j.id not in chosen_ids]

        pairs = []
        for c, r in zip(chosen_pool, rejected_pool, strict=False):
            pairs.append(PreferencePair(chosen=c, rejected=r))

        return pairs

    def _pair_exists(self, chosen_id: int, rejected_id: int) -> bool:
        """Check if this exact preference pair already exists.

        Allows same chosen with different rejected (cross-product pairing).
        Blocks: exact duplicate pairs and role reversals (same joke as both).
        """
        sql = text("""
            SELECT 1 FROM atomic_preference_details
            WHERE (metadata->>'chosen_fact_id' = :chosen_id
                   AND metadata->>'rejected_fact_id' = :rejected_id)
               OR (metadata->>'chosen_fact_id' = :rejected_id
                   AND metadata->>'rejected_fact_id' = :chosen_id)
            LIMIT 1
        """)
        with self._pg.connect() as conn:
            result = conn.execute(
                sql, {"chosen_id": str(chosen_id), "rejected_id": str(rejected_id)}
            ).fetchone()
        return result is not None
