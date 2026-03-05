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
        raise ValueError(f"Unsupported operator for StarFilter: {self.op!r}")

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


def _validate_pairing_params(margin: int, min_pairs: int | None, max_pairs: int | None) -> None:
    """Validate pairing parameters."""
    if margin < 1:
        raise ValueError(f"margin must be >= 1, got {margin}")
    if min_pairs is not None and min_pairs < 0:
        raise ValueError(f"min_pairs cannot be negative, got {min_pairs}")
    if max_pairs is not None and max_pairs < 0:
        raise ValueError(f"max_pairs cannot be negative, got {max_pairs}")
    if min_pairs is not None and max_pairs is not None and min_pairs > max_pairs:
        raise ValueError(f"min_pairs ({min_pairs}) cannot be greater than max_pairs ({max_pairs})")


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

    Raises:
        ValueError: If margin < 1 or min/max_pairs are invalid.
    """
    _validate_pairing_params(margin, min_pairs, max_pairs)

    if not rated_items:
        return StarPairingResult(
            pairs=[], total_rated=0, strategy="margin", params={"margin": margin}
        )

    chosen_pool, rejected_pool = _build_pools(rated_items, margin, chosen_filter, rejected_filter)

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
    if max_pairs is not None and max_pairs < 0:
        raise ValueError(f"max_pairs cannot be negative, got {max_pairs}")

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
    """Generate pairs by iterating rejected pool once (O(n) instead of O(n*m))."""
    if not chosen_pool or not rejected_pool:
        return []

    pairs: list[StarPreferencePair] = []
    used_chosen: set[int] = set()
    chosen_idx = 0

    # Iterate rejected once - each rejected item is used at most once
    for rejected in rejected_pool:
        if len(pairs) >= target:
            break

        # Find a chosen that can pair with this rejected
        match = _find_chosen_match(rejected, chosen_pool, used_chosen, margin, no_reuse, chosen_idx)
        if match:
            chosen, new_idx = match
            pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
            if no_reuse:
                used_chosen.add(chosen.id)
            chosen_idx = new_idx

    return pairs


def _find_chosen_match(
    rejected: StarRatedItem,
    chosen_pool: list[StarRatedItem],
    used: set[int],
    margin: int,
    no_reuse: bool,
    start_idx: int,
) -> tuple[StarRatedItem, int] | None:
    """Find a chosen item that can pair with this rejected item.

    Returns (chosen, next_idx) or None if no match found.
    """
    # Try from start_idx first (for cycling through chosen in order)
    pool_size = len(chosen_pool)
    for offset in range(pool_size):
        idx = (start_idx + offset) % pool_size
        chosen = chosen_pool[idx]
        if (
            chosen.score - rejected.score >= margin
            and chosen.id != rejected.id
            and (not no_reuse or chosen.id not in used)
        ):
            return chosen, (idx + 1) % pool_size
    return None
