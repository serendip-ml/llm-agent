"""Star-based preference pairing.

Specialization for integer star ratings (1-5 scale).
Provides margin-based and threshold-based pairing algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import PairingResult, PreferencePair, RatedItem


# Type alias for star-rated items
StarRatedItem = RatedItem[int]
StarPreferencePair = PreferencePair[int]
StarPairingResult = PairingResult[int]


@dataclass
class StarFilter:
    """Filter for star ratings with comparison operator.

    Examples:
        StarFilter.parse("3")   -> exact match (== 3)
        StarFilter.parse(">=3") -> minimum (>= 3)
        StarFilter.parse("<=2") -> maximum (<= 2)
        StarFilter.parse(">1")  -> greater than (> 1)
        StarFilter.parse("<5")  -> less than (< 5)
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

        Raises:
            ValueError: If value is not a valid star filter format
        """
        if value is None:
            return None
        if isinstance(value, int):
            return cls(value=value, op="==")

        value = value.strip()
        if not value:
            return None

        try:
            for op in (">=", "<=", ">", "<"):
                if value.startswith(op):
                    num = int(value[len(op) :])
                    return cls(value=num, op=op)
            return cls(value=int(value), op="==")
        except ValueError:
            raise ValueError(
                f"Invalid star filter: '{value}'. "
                f"Use a number (3) or operator with number (>=3, <=2, >1, <5)"
            ) from None


def pair_by_margin(
    rated_items: list[StarRatedItem],
    margin: int = 1,
    min_pairs: int | None = None,
    max_pairs: int | None = None,
    no_reuse: bool = False,
    chosen_filter: StarFilter | None = None,
    rejected_filter: StarFilter | None = None,
) -> StarPairingResult:
    """Pair highest-rated with lowest-rated, respecting minimum margin.

    This is the "relative" pairing strategy - pairs are formed based on
    the relative difference in ratings rather than fixed thresholds.

    Args:
        rated_items: List of star-rated items to pair.
        margin: Minimum star difference for valid pairs (default: 1).
        min_pairs: Target number of pairs (reuses chosen if needed).
        max_pairs: Maximum pairs to return (caps output).
        no_reuse: If True, each chosen item is used at most once (1:1 pairing).
        chosen_filter: Filter for chosen pool (e.g., StarFilter(4, ">=")).
        rejected_filter: Filter for rejected pool (e.g., StarFilter(2, "<=")).

    Returns:
        PairingResult with pairs and metadata.
    """
    if not rated_items:
        return StarPairingResult(
            pairs=[], total_rated=0, strategy="margin", params={"margin": margin}
        )

    sorted_items = sorted(rated_items, key=lambda x: (-x.score, x.id))
    chosen_pool, rejected_pool = _build_pools(sorted_items, margin, chosen_filter, rejected_filter)

    if not chosen_pool or not rejected_pool:
        return StarPairingResult(
            pairs=[],
            total_rated=len(rated_items),
            strategy="margin",
            params={"margin": margin},
        )

    target = min_pairs if min_pairs is not None else len(chosen_pool)
    pairs = _generate_pairs(chosen_pool, rejected_pool, margin, target, no_reuse)

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    return StarPairingResult(
        pairs=pairs,
        total_rated=len(rated_items),
        strategy="margin",
        params={"margin": margin},
    )


def pair_by_threshold(
    rated_items: list[StarRatedItem],
    high_threshold: int = 4,
    low_threshold: int = 2,
    max_pairs: int | None = None,
) -> StarPairingResult:
    """Pair based on fixed star thresholds.

    Chosen items must be at or above high_threshold, rejected items
    must be at or below low_threshold.

    Args:
        rated_items: List of star-rated items to pair.
        high_threshold: Minimum stars for chosen pool (default: 4).
        low_threshold: Maximum stars for rejected pool (default: 2).
        max_pairs: Maximum pairs to return (caps output).

    Returns:
        PairingResult with pairs and metadata.

    Raises:
        ValueError: If high_threshold <= low_threshold.
    """
    if high_threshold <= low_threshold:
        raise ValueError(
            f"high_threshold ({high_threshold}) must be greater than "
            f"low_threshold ({low_threshold})"
        )

    params = {"high_threshold": high_threshold, "low_threshold": low_threshold}

    if not rated_items:
        return StarPairingResult(pairs=[], total_rated=0, strategy="threshold", params=params)

    chosen_pool = [item for item in rated_items if item.score >= high_threshold]
    chosen_ids = {item.id for item in chosen_pool}
    rejected_pool = [
        item for item in rated_items if item.score <= low_threshold and item.id not in chosen_ids
    ]

    pairs: list[StarPreferencePair] = []
    for chosen, rejected in zip(chosen_pool, rejected_pool, strict=False):
        pairs.append(PreferencePair(chosen=chosen, rejected=rejected))

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    return StarPairingResult(
        pairs=pairs, total_rated=len(rated_items), strategy="threshold", params=params
    )


def _build_pools(
    sorted_items: list[StarRatedItem],
    margin: int,
    chosen_filter: StarFilter | None,
    rejected_filter: StarFilter | None,
) -> tuple[list[StarRatedItem], list[StarRatedItem]]:
    """Build chosen and rejected pools based on margin requirement."""
    if not sorted_items:
        return [], []

    min_score = min(item.score for item in sorted_items)
    max_score = max(item.score for item in sorted_items)

    chosen_pool = [item for item in sorted_items if min_score <= item.score - margin]
    rejected_pool = [item for item in sorted_items if max_score >= item.score + margin]

    if chosen_filter is not None:
        chosen_pool = [item for item in chosen_pool if chosen_filter.matches(item.score)]
    if rejected_filter is not None:
        rejected_pool = [item for item in rejected_pool if rejected_filter.matches(item.score)]

    chosen_pool.sort(key=lambda x: (-x.score, x.id))
    rejected_pool.sort(key=lambda x: (x.score, x.id))

    return chosen_pool, rejected_pool


def _generate_pairs(
    chosen_pool: list[StarRatedItem],
    rejected_pool: list[StarRatedItem],
    margin: int,
    target: int,
    no_reuse: bool,
) -> list[StarPreferencePair]:
    """Generate pairs by cycling through chosen pool until target or exhausted."""
    pairs: list[StarPreferencePair] = []
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

        match = _find_match(chosen, rejected_pool, used_rejected, margin)
        if match:
            pairs.append(PreferencePair(chosen=chosen, rejected=match))
            used_rejected.add(match.id)
            used_chosen.add(chosen.id)

        if chosen_idx >= max_iterations:
            break

    return pairs


def _find_match(
    chosen: StarRatedItem,
    pool: list[StarRatedItem],
    used: set[int],
    margin: int,
) -> StarRatedItem | None:
    """Find first unused rejected item that meets margin requirement."""
    for rejected in pool:
        if (
            rejected.id not in used
            and rejected.id != chosen.id
            and chosen.score - rejected.score >= margin
        ):
            return rejected
    return None
