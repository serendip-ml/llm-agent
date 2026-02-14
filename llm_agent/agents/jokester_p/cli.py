"""CLI tool for jokester-p agent."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from sqlalchemy import text

from .pairing import PairingService, PreferencePair
from .rating import RatingService


if TYPE_CHECKING:
    from llm_infer.client import LLMRouter


class JokesterCLI(Tool):
    """CLI commands for jokester-p agent.

    Provides agent-specific commands:
    - stats: Show joke rating statistics
    - pairs-sync: Create preference pairs from rated jokes
    """

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="jokester-p", help_text="Jokester agent CLI commands")
        super().__init__(parent, config)
        self._pg: PG | None = None

    def configure(self) -> None:
        """Set up database connection."""
        if not hasattr(self.app, "config"):
            raise RuntimeError("App does not have config")

        db_config = self.app.config.learn.db
        if db_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        self._pg = PG(self.lg, db_config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        subparsers.add_parser("stats", help="Show joke rating statistics")
        self._add_pairs_sync_args(subparsers)
        self._add_rate_args(subparsers)

    def _add_pairs_sync_args(self, subparsers: Any) -> None:
        """Add pairs-sync command arguments."""
        p = subparsers.add_parser("pairs-sync", help="Create preference pairs from rated jokes")
        p.add_argument(
            "--strategy",
            choices=["relative", "threshold"],
            default="relative",
            help="Pairing strategy (default: relative)",
        )
        p.add_argument(
            "--min-gap",
            type=int,
            default=1,
            help="Minimum star difference for pairing (default: 1)",
        )
        p.add_argument(
            "--dry-run", action="store_true", help="Show what would be created without saving"
        )

    def _add_rate_args(self, subparsers: Any) -> None:
        """Add rate command arguments."""
        p = subparsers.add_parser("rate", help="Rate unrated jokes in batches")
        p.add_argument("--limit", type=int, help="Maximum jokes to rate (default: all unrated)")
        p.add_argument(
            "--batch-size", type=int, help="Jokes per API call (default: from config, typically 5)"
        )

    def run(self, **kwargs: Any) -> int:
        command = self.args.command

        if command == "stats":
            return self._cmd_stats()
        elif command == "pairs-sync":
            return self._cmd_pairs_sync()
        elif command == "rate":
            return self._cmd_rate()
        else:
            print("Usage: agent jokester-p <command>")
            print("Commands: stats, pairs-sync, rate")
            return 1

    def _cmd_stats(self) -> int:
        """Show joke rating statistics."""
        assert self._pg is not None

        stats = self._get_rating_stats()
        dist = self._get_rating_distribution()

        print("\n=== Jokester-p Stats ===\n")
        print(f"Total jokes:  {stats['total']}")
        print(f"Rated:        {stats['rated']}")
        print(f"Unrated:      {stats['unrated']}")
        print(
            f"Avg stars:    {stats['avg_stars']:.2f}" if stats["avg_stars"] else "Avg stars:    N/A"
        )
        print()
        print("Rating Distribution:")
        for row in dist:
            stars_visual = "★" * row["stars"] + "☆" * (5 - row["stars"])
            print(f"  {stars_visual}  {row['count']:4d}  ({row['pct']:.1f}%)")
        print()

        return 0

    def _cmd_pairs_sync(self) -> int:
        """Create preference pairs from rated jokes."""
        assert self._pg is not None

        service = PairingService(self.lg, self._pg)
        result = service.create_pairs(strategy=self.args.strategy, min_gap=self.args.min_gap)

        if result.total_rated == 0:
            print("No rated jokes found.")
            return 0

        self._print_pairs_summary(result)
        if not result.pairs:
            print("No pairs could be created.")
            return 0

        print(f"Pairs to create: {len(result.pairs)}\n")
        self._print_pairs_preview(result.pairs)

        if self.args.dry_run:
            print("Dry run complete. Use without --dry-run to save pairs.")
            return 0

        print(f"\n✓ Created {service.save_pairs(result.pairs)} preference pairs")
        return 0

    def _print_pairs_summary(self, result: Any) -> None:
        """Print pairing summary info."""
        print(f"\nRated jokes: {result.total_rated}")
        print(f"Strategy: {result.strategy}")
        print(f"Min gap: {result.min_gap} stars")
        if self.args.dry_run:
            print("(Dry run - no changes will be saved)")
        print()

    def _cmd_rate(self) -> int:
        """Rate unrated jokes in batches."""
        assert self._pg is not None

        rate_config = self._load_rate_config()
        if rate_config is None:
            return 1

        prompt_template, batch_size = rate_config

        llm_client = self._create_llm_client()
        if llm_client is None:
            print("Error: Could not create LLM client")
            return 1

        try:
            service = RatingService(
                lg=self.lg,
                pg=self._pg,
                llm_client=llm_client,
                prompt_template=prompt_template,
                batch_size=batch_size,
            )
            return self._run_rating(service, batch_size)
        finally:
            llm_client.close()

    def _load_rate_config(self) -> tuple[str, int] | None:
        """Load rating config. Returns (prompt_template, batch_size) or None on error."""
        agent_config = self._get_agent_config()
        if agent_config is None:
            print("Error: Could not load agent config")
            return None

        rating_config = agent_config.get("rating", {})
        prompt_template = self._get_prompt_template(rating_config)
        config_batch_size = rating_config.get("batch_size", 5)
        batch_size = self.args.batch_size or config_batch_size

        return (prompt_template, batch_size)

    def _run_rating(self, service: RatingService, batch_size: int) -> int:
        """Execute rating and print results."""
        unrated = service.get_unrated_jokes(self.args.limit)
        if not unrated:
            print("No unrated jokes found.")
            return 0

        print(f"\nUnrated jokes: {len(unrated)}")
        print(f"Batch size: {batch_size}")
        print(f"Estimated batches: {(len(unrated) + batch_size - 1) // batch_size}")
        print()

        result = service.rate_all(limit=self.args.limit, batch_size=batch_size)

        print(f"\n✓ Rated: {result.rated}")
        if result.failed > 0:
            print(f"  Failed: {result.failed}")
        print(f"  Batches: {result.batches}")

        return 0

    def _get_agent_config(self) -> dict[str, Any] | None:
        """Get jokester-p agent config."""
        if not hasattr(self.app, "config") or not hasattr(self.app.config, "agents"):
            return None
        config = self.app.config.agents.get("jokester-p")
        return dict(config) if config else None

    def _get_prompt_template(self, rating_config: dict[str, Any]) -> str:
        """Extract prompt template from rating config."""
        models = rating_config.get("models", {})
        atomic = models.get("atomic", {})
        solution = atomic.get("solution", {})
        return str(solution.get("prompt", "Rate this content 1-5 stars."))

    def _create_llm_client(self) -> LLMRouter | None:
        """Create LLM client from config."""
        from llm_infer.client import Factory as LLMClientFactory

        llm_config = self.app.config.get("llm", {})
        try:
            return LLMClientFactory(self.lg).from_config(llm_config)
        except Exception as e:
            self.lg.error("failed to create LLM client", extra={"exception": e})
            return None

    def _print_pairs_preview(self, pairs: list[PreferencePair]) -> None:
        """Print preview of pairs."""
        for i, pair in enumerate(pairs, 1):
            c = (
                pair.chosen.content[:40] + "..."
                if len(pair.chosen.content) > 40
                else pair.chosen.content
            )
            r = (
                pair.rejected.content[:40] + "..."
                if len(pair.rejected.content) > 40
                else pair.rejected.content
            )
            print(f"{i}. [{pair.chosen.stars}★] {c}")
            print(f"   vs [{pair.rejected.stars}★] {r}")
            print()

    def _get_rating_stats(self) -> dict[str, Any]:
        """Get rating statistics."""
        assert self._pg is not None
        sql = text("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE afd.id IS NOT NULL) as rated,
                COUNT(*) FILTER (WHERE afd.id IS NULL) as unrated,
                AVG((afd.context->>'stars')::int) as avg_stars
            FROM atomic_facts af
            LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = 'jokester-p'
        """)
        with self._pg.connect() as conn:
            result = conn.execute(sql).fetchone()
        return {
            "total": result[0],
            "rated": result[1],
            "unrated": result[2],
            "avg_stars": float(result[3]) if result[3] else None,
        }

    def _get_rating_distribution(self) -> list[dict[str, Any]]:
        """Get rating distribution."""
        assert self._pg is not None
        sql = text("""
            SELECT
                (context->>'stars')::int as stars,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
            FROM atomic_feedback_details afd
            JOIN atomic_facts af ON af.id = afd.fact_id
            WHERE af.context_key = 'jokester-p'
            GROUP BY (context->>'stars')::int
            ORDER BY stars DESC
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [{"stars": r[0], "count": r[1], "pct": float(r[2])} for r in rows]
