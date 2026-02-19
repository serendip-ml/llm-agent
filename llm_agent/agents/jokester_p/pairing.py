"""Preference pairing service for jokester-p agent.

Creates preference pairs from rated jokes for DPO training.
"""

from __future__ import annotations

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
    margin: int


@dataclass
class StarFilter:
    """Filter for star ratings with comparison operator.

    Examples:
        StarFilter.parse("3")   -> exact match (== 3)
        StarFilter.parse(">=3") -> minimum (>= 3)
        StarFilter.parse("<=2") -> maximum (<= 2)
    """

    value: int
    op: str = "=="  # "==", ">=", "<=", ">", "<"

    def matches(self, stars: int) -> bool:
        """Check if stars value matches this filter."""
        if self.op == "==":
            return stars == self.value
        elif self.op == ">=":
            return stars >= self.value
        elif self.op == "<=":
            return stars <= self.value
        elif self.op == ">":
            return stars > self.value
        elif self.op == "<":
            return stars < self.value
        return False

    @classmethod
    def parse(cls, value: str | int | None) -> StarFilter | None:
        """Parse a star filter from string or int.

        Args:
            value: "3" (exact), ">=3" (min), "<=2" (max), or int for exact match

        Returns:
            StarFilter instance or None if value is None/empty
        """
        if value is None:
            return None
        if isinstance(value, int):
            return cls(value=value, op="==")

        value = value.strip()
        if not value:
            return None

        # Check for operator prefix
        for op in (">=", "<=", ">", "<"):
            if value.startswith(op):
                num = int(value[len(op) :])
                return cls(value=num, op=op)

        # No operator = exact match
        return cls(value=int(value), op="==")


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

    def get_rated_jokes(
        self, max_chars: int | None = None, model: str | None = None
    ) -> list[RatedJoke]:
        """Get all rated jokes sorted by stars (desc)."""
        joins_sql, filters_sql, params = self._build_rated_jokes_filters(max_chars, model)
        sql = text(f"""
            SELECT DISTINCT ON (af.id)
                af.id, af.content, (afd.context->>'stars')::int as stars
            FROM atomic_facts af
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            {joins_sql}
            WHERE af.context_key = :context_key
              AND afd.context->>'stars' IS NOT NULL {filters_sql}
            ORDER BY af.id, afd.id DESC
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [RatedJoke(id=r[0], content=r[1], stars=r[2]) for r in rows]

    def _build_rated_jokes_filters(
        self, max_chars: int | None, model: str | None
    ) -> tuple[str, str, dict[str, object]]:
        """Build SQL fragments and params for rated jokes query."""
        joins: list[str] = []
        filters: list[str] = []
        params: dict[str, object] = {"context_key": self._context_key}

        if max_chars:
            joins.append("JOIN atomic_solution_details asd ON asd.fact_id = af.id")
            filters.append("AND length(asd.answer_text) < :max_chars")
            params["max_chars"] = max_chars

        if model:
            joins.append("JOIN agent_jokester_model_usage u ON u.fact_id = af.id")
            filters.append("AND u.model_name LIKE :model_pattern")
            params["model_pattern"] = f"%{model}%"

        return "\n            ".join(joins), " ".join(filters), params

    def create_pairs(
        self,
        strategy: str = "relative",
        margin: int = 1,
        high_threshold: int = 4,
        low_threshold: int = 2,
        min_pairs: int | None = None,
        max_pairs: int | None = None,
        max_chars: int | None = None,
        no_reuse: bool = False,
        model: str | None = None,
        chosen_stars: StarFilter | None = None,
        rejected_stars: StarFilter | None = None,
    ) -> PairingResult:
        """Create preference pairs from rated jokes.

        Args:
            strategy: "relative" or "threshold"
            margin: Minimum star difference for relative pairing
            high_threshold: Minimum stars for chosen (threshold strategy)
            low_threshold: Maximum stars for rejected (threshold strategy)
            min_pairs: Minimum pairs to generate (reuses chosen jokes if needed)
            max_pairs: Maximum pairs to generate (caps output)
            max_chars: Only include jokes under this character length
            no_reuse: If True, each chosen joke is used at most once (1:1 pairing)
            model: Only include jokes generated by this model (substring match)
            chosen_stars: Filter for chosen jokes (e.g., StarFilter(3, ">=") for 3★+)
            rejected_stars: Filter for rejected jokes (e.g., StarFilter(2, "==") for 2★)

        Returns:
            PairingResult with pairs and metadata.
        """
        self._lg.debug(
            "fetching rated jokes...",
            extra={"max_chars": max_chars, "model": model},
        )
        rated = self.get_rated_jokes(max_chars=max_chars, model=model)
        self._lg.debug("fetched rated jokes", extra={"count": len(rated)})

        self._lg.debug("creating pairs...", extra={"strategy": strategy, "margin": margin})
        if strategy == "relative":
            pairs = self._pair_relative(
                rated, margin, min_pairs, no_reuse, chosen_stars, rejected_stars
            )
        else:
            pairs = self._pair_threshold(rated, high_threshold, low_threshold)
        self._lg.debug("created pairs", extra={"count": len(pairs)})

        # Apply max cap
        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
            self._lg.debug("capped pairs", extra={"max_pairs": max_pairs})

        return PairingResult(
            pairs=pairs,
            total_rated=len(rated),
            strategy=strategy,
            margin=margin,
        )

    def _pair_relative(
        self,
        rated: list[RatedJoke],
        margin: int,
        min_pairs: int | None = None,
        no_reuse: bool = False,
        chosen_stars: StarFilter | None = None,
        rejected_stars: StarFilter | None = None,
    ) -> list[PreferencePair]:
        """Pair highest with lowest, respecting min gap.

        If min_pairs is set and 1:1 pairing yields fewer, reuses chosen jokes
        with multiple rejected jokes until min_pairs is reached or rejected
        pool is exhausted.

        If no_reuse is True, each chosen joke is used at most once (1:1 pairing).
        """
        sorted_jokes = sorted(rated, key=lambda x: (-x.stars, x.id))
        chosen_pool, rejected_pool = self._build_pairing_pools(
            sorted_jokes, margin, chosen_stars, rejected_stars
        )

        if not chosen_pool or not rejected_pool:
            return []

        target = min_pairs if min_pairs is not None else len(chosen_pool)
        return self._generate_pairs(chosen_pool, rejected_pool, margin, target, no_reuse)

    def _build_pairing_pools(
        self,
        sorted_jokes: list[RatedJoke],
        margin: int,
        chosen_stars: StarFilter | None = None,
        rejected_stars: StarFilter | None = None,
    ) -> tuple[list[RatedJoke], list[RatedJoke]]:
        """Build chosen and rejected pools based on margin requirement.

        O(n log n) - single pass to build pools, then sort.

        Args:
            sorted_jokes: All rated jokes sorted by stars desc.
            margin: Minimum star difference for pairing.
            chosen_stars: Filter for chosen jokes (supports ==, >=, <=, >, <).
            rejected_stars: Filter for rejected jokes (supports ==, >=, <=, >, <).
        """
        if not sorted_jokes:
            return [], []

        # Precompute min/max stars - O(n)
        min_stars = min(j.stars for j in sorted_jokes)
        max_stars = max(j.stars for j in sorted_jokes)

        # A joke can be chosen if there exists a rejected joke with stars <= (joke.stars - margin)
        # A joke can be rejected if there exists a chosen joke with stars >= (joke.stars + margin)
        chosen_pool = [j for j in sorted_jokes if min_stars <= j.stars - margin]
        rejected_pool = [j for j in sorted_jokes if max_stars >= j.stars + margin]

        # Apply chosen_stars filter
        if chosen_stars is not None:
            chosen_pool = [j for j in chosen_pool if chosen_stars.matches(j.stars)]

        # Apply rejected_stars filter
        if rejected_stars is not None:
            rejected_pool = [j for j in rejected_pool if rejected_stars.matches(j.stars)]

        # Sort: chosen by stars desc, rejected by stars asc (worst first)
        chosen_pool.sort(key=lambda x: (-x.stars, x.id))
        rejected_pool.sort(key=lambda x: (x.stars, x.id))
        return chosen_pool, rejected_pool

    def _generate_pairs(
        self,
        chosen_pool: list[RatedJoke],
        rejected_pool: list[RatedJoke],
        margin: int,
        target: int,
        no_reuse: bool = False,
    ) -> list[PreferencePair]:
        """Generate pairs by cycling through chosen pool until target or exhausted."""
        pairs: list[PreferencePair] = []
        used_rejected: set[int] = set()
        used_chosen: set[int] = set()
        chosen_idx = 0
        max_iterations = len(chosen_pool) * len(rejected_pool)

        while len(pairs) < target and len(used_rejected) < len(rejected_pool):
            chosen = chosen_pool[chosen_idx % len(chosen_pool)]
            chosen_idx += 1

            if no_reuse and chosen.id in used_chosen:
                if chosen_idx >= max_iterations:
                    break
                continue

            match = self._find_rejected_match(chosen, rejected_pool, used_rejected, margin)
            if match:
                pairs.append(PreferencePair(chosen=chosen, rejected=match))
                used_rejected.add(match.id)
                used_chosen.add(chosen.id)

            if chosen_idx >= max_iterations:
                break

        return pairs

    def _find_rejected_match(
        self, chosen: RatedJoke, pool: list[RatedJoke], used: set[int], margin: int
    ) -> RatedJoke | None:
        """Find first unused rejected joke that meets gap requirement."""
        for rejected in pool:
            is_valid = (
                rejected.id not in used
                and rejected.id != chosen.id
                and chosen.stars - rejected.stars >= margin
            )
            if is_valid:
                return rejected
        return None

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
