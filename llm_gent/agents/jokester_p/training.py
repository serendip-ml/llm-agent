"""Training data provider for jokester-p agent."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from appinfra.db.pg import PG

from .pairing import PairingService, StarFilter


if TYPE_CHECKING:
    from appinfra.log import Logger


class JokesterTrainingProvider:
    """Provides training data for jokester-p agent."""

    def __init__(self, lg: Logger, pg: PG, context_key: str) -> None:
        self._lg = lg
        self._pg = pg
        self._context_key = context_key

    def add_args(self, parser: argparse.ArgumentParser, method: str = "sft") -> None:
        """Add jokester-specific training arguments."""
        self._add_common_args(parser)
        if method == "sft":
            self._add_sft_args(parser)
        elif method == "dpo":
            self._add_dpo_args(parser)

    def _add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common training arguments (SFT and DPO)."""
        parser.add_argument(
            "--max-chars",
            type=int,
            default=140,
            help="Maximum joke length in characters (default: 140)",
        )
        parser.add_argument(
            "--filter-model",
            type=str,
            help="Filter jokes by source model (substring match)",
        )
        parser.add_argument(
            "--filter-schema",
            type=str,
            help="Schema to get training data from (default: agent's kelt.schema)",
        )
        parser.add_argument(
            "--max",
            type=int,
            dest="max_examples",
            help="Maximum examples to include",
        )

    def _add_sft_args(self, parser: argparse.ArgumentParser) -> None:
        """Add SFT-specific training arguments."""
        parser.add_argument(
            "--min-stars",
            type=int,
            default=4,
            help="Minimum star rating (default: 4)",
        )

    def _add_dpo_args(self, parser: argparse.ArgumentParser) -> None:
        """Add DPO-specific training arguments."""
        parser.add_argument(
            "--margin",
            type=int,
            help="Minimum star difference for DPO pairs (default: 2)",
        )
        parser.add_argument(
            "--min-pairs",
            type=int,
            help="Minimum pairs to generate",
        )
        parser.add_argument(
            "--max-pairs",
            type=int,
            help="Maximum pairs to include",
        )
        parser.add_argument(
            "--no-reuse",
            action="store_true",
            help="1:1 pairing only (no reuse of chosen jokes)",
        )
        parser.add_argument(
            "--chosen-stars",
            type=str,
            help="Stars for chosen (e.g., 4 or >=4)",
        )
        parser.add_argument(
            "--rejected-stars",
            type=str,
            help="Stars for rejected (e.g., 2 or <=2)",
        )

    def get_sft_examples(self, args: argparse.Namespace) -> list[dict[str, str]]:
        """Get SFT training examples from high-rated jokes."""
        filter_schema = getattr(args, "filter_schema", None)
        service = PairingService(self._lg, self._pg, self._context_key, schema=filter_schema)
        filter_model = getattr(args, "filter_model", None)
        all_jokes = service.get_rated_jokes(
            max_chars=args.max_chars,
            model=filter_model,
        )

        # Filter by minimum stars
        min_stars = getattr(args, "min_stars", 4)
        filtered = [j for j in all_jokes if j.score >= min_stars]

        # Sort by stars descending
        filtered.sort(key=lambda j: j.score, reverse=True)

        # Apply max cap
        max_examples = getattr(args, "max_examples", None)
        if max_examples is not None and len(filtered) > max_examples:
            filtered = filtered[:max_examples]

        # Convert to SFT format
        return [{"instruction": "Tell me a joke.", "output": j.content} for j in filtered]

    def get_dpo_pairs(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        """Get DPO preference pairs."""
        filter_schema = getattr(args, "filter_schema", None)
        service = PairingService(self._lg, self._pg, self._context_key, schema=filter_schema)

        result = service.create_pairs(
            margin=getattr(args, "margin", None) or 2,
            max_chars=args.max_chars,
            model=getattr(args, "filter_model", None),
            min_pairs=getattr(args, "min_pairs", None),
            max_pairs=getattr(args, "max_pairs", None),
            no_reuse=getattr(args, "no_reuse", False),
            chosen_stars=StarFilter.parse(getattr(args, "chosen_stars", None)),
            rejected_stars=StarFilter.parse(getattr(args, "rejected_stars", None)),
        )

        return self._format_dpo_pairs(result.pairs)

    def _format_dpo_pairs(self, pairs: list[Any]) -> list[dict[str, Any]]:
        """Convert preference pairs to DPO format."""
        return [
            {
                "prompt": "Tell me a joke.",
                "chosen": p.chosen.content,
                "rejected": p.rejected.content,
            }
            for p in pairs
        ]

    def get_context_key(self) -> str:
        """Get context key for this agent."""
        return self._context_key

    def get_description(self, method: str, count: int) -> str:
        """Get description for the training manifest."""
        if method == "sft":
            return f"SFT from {count} high-rated jokes"
        elif method == "dpo":
            return f"DPO from {count} preference pairs"
        return f"{method.upper()} training with {count} examples"
