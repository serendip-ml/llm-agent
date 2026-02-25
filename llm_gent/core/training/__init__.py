"""Training infrastructure for agents.

Provides preference pairing algorithms for DPO training.
"""

from .pairing import (
    # Generic types
    PairingResult,
    PreferencePair,
    RatedItem,
    # Star-based specialization
    StarFilter,
    StarPairingResult,
    StarPreferencePair,
    StarRatedItem,
    pair_by_margin,
    pair_by_threshold,
)


__all__ = [
    # Generic types
    "RatedItem",
    "PreferencePair",
    "PairingResult",
    # Star-based types
    "StarRatedItem",
    "StarPreferencePair",
    "StarPairingResult",
    "StarFilter",
    # Star-based algorithms
    "pair_by_margin",
    "pair_by_threshold",
]
