"""CLI tool for jokester-p agent."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_learn.core import Database
from llm_learn.training import DpoClient
from sqlalchemy import text

from .pairing import PairingService
from .rating import RatingService


if TYPE_CHECKING:
    from ...core.llm import LLMCaller


class JokesterCLI(Tool):
    """CLI commands for jokester-p agent.

    Provides agent-specific commands:
    - stats: Show joke rating statistics
    - rate: Rate unrated jokes in batches
    - train: Create preference pairs and training run for DPO
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
        self._add_rate_args(subparsers)
        self._add_train_args(subparsers)

    def _add_rate_args(self, subparsers: Any) -> None:
        """Add rate command arguments."""
        p = subparsers.add_parser("rate", help="Rate unrated jokes in batches")
        p.add_argument("--limit", type=int, help="Maximum jokes to rate (default: all unrated)")
        p.add_argument(
            "--batch-size", type=int, help="Jokes per API call (default: from config, typically 5)"
        )
        p.add_argument(
            "--dry-run", action="store_true", help="Show what would be rated without sending to LLM"
        )

    def _add_train_args(self, subparsers: Any) -> None:
        """Add train command arguments."""
        p = subparsers.add_parser("train", help="Create preference pairs and training run")
        p.add_argument(
            "--adapter-name",
            type=str,
            default="jokester-dpo",
            help="Name for the trained adapter (default: jokester-dpo)",
        )
        p.add_argument(
            "--min-gap",
            type=int,
            default=2,
            help="Minimum star difference for pairing (default: 2 = 3★ vs 1★)",
        )
        p.add_argument(
            "--limit", type=int, help="Maximum pairs to include (default: all available)"
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without saving",
        )

    def run(self, **kwargs: Any) -> int:
        command = self.args.command

        if command == "stats":
            return self._cmd_stats()
        elif command == "rate":
            return self._cmd_rate()
        elif command == "train":
            return self._cmd_train()
        else:
            print("Usage: agent jokester-p <command>")
            print("Commands: stats, rate, train")
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

        # Show per-training-run stats
        self._print_training_run_stats()

        return 0

    def _print_training_run_stats(self) -> None:
        """Print stats broken down by training runs."""
        runs = self._get_training_runs()
        if not runs:
            return

        print("\n=== By Training Run ===\n")
        period_stats = self._get_stats_by_training_period(runs)

        for period in period_stats:
            if period["count"] == 0:
                continue
            avg = f"{period['avg_stars']:.2f}" if period["avg_stars"] else "N/A"
            dist = period.get("distribution", {})
            total = period["count"]
            stars_str = self._format_star_distribution(dist, total)
            print(f"{period['period']:20s}  {period['count']:5d} jokes  avg={avg}  {stars_str}")
        print()

    def _format_star_distribution(self, dist: dict[int, int], total: int) -> str:
        """Format star distribution as: ★★★★★ 0(0.0%)  ★★★★☆ 1(0.1%)  ★★★☆☆ 10(10.0%)"""
        parts = []
        for stars in (5, 4, 3):
            count = dist.get(stars, 0)
            pct = count * 100 / total if total else 0
            symbol = "★" * stars + "☆" * (5 - stars)
            parts.append(f"{symbol} {count}({pct:.1f}%)")
        return "  ".join(parts)

    def _cmd_rate(self) -> int:
        """Rate unrated jokes in batches."""
        assert self._pg is not None

        rate_config = self._load_rate_config()
        if rate_config is None:
            return 1

        prompt_template, batch_size, provider, model = rate_config
        dry_run = getattr(self.args, "dry_run", False)

        llm_caller = self._create_llm_caller(dry_run=dry_run)
        if llm_caller is None:
            print("Error: Could not create LLM caller")
            return 1

        try:
            service = RatingService(
                lg=self.lg,
                pg=self._pg,
                llm_caller=llm_caller,
                prompt_template=prompt_template,
                context_key=self._get_context_key(),
                batch_size=batch_size,
                provider=provider,
                model=model,
            )
            return self._run_rating(service, batch_size)
        finally:
            llm_caller.close()

    def _load_rate_config(self) -> tuple[str, int, str, str] | None:
        """Load rating config. Returns (prompt, batch_size, provider, model) or None."""
        agent_config = self._get_agent_config()
        if agent_config is None:
            print("Error: Could not load agent config")
            return None

        rating_config = agent_config.get("rating", {})
        prompt_template = self._get_prompt_template(rating_config)
        config_batch_size = rating_config.get("batch_size", 5)
        batch_size = self.args.batch_size or config_batch_size

        # Extract provider and model from first enabled provider
        provider, model = self._get_provider_settings(rating_config)

        return (prompt_template, batch_size, provider, model)

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

        total_rated = 0
        batch_count = 0

        for batch in service.rate_batches(limit=self.args.limit, batch_size=batch_size):
            batch_count += 1
            for r in batch:
                total_rated += 1
                stars = "★" * r.stars + "☆" * (5 - r.stars)
                joke = r.content[:90] + "..." if len(r.content) > 90 else r.content
                print(f"{r.id}  ({r.stars})  {stars}  {joke}")

        print(f"\n✓ Rated: {total_rated}")
        print(f"  Batches: {batch_count}")

        return 0

    def _cmd_train(self) -> int:
        """Create preference pairs from rated jokes and set up training run."""
        assert self._pg is not None

        min_gap = getattr(self.args, "min_gap", 2)
        dry_run = getattr(self.args, "dry_run", False)

        # Step 1: Create new preference pairs from rated jokes
        new_pairs = self._create_preference_pairs(min_gap, dry_run)

        # Step 2: Get all untrained pairs and create training run
        client = self._create_training_client()
        limit = getattr(self.args, "limit", None)
        pairs = client.get_untrained_pairs(min_margin=float(min_gap), limit=limit)

        if not pairs:
            print("\nNo untrained preference pairs available for training.")
            return 0

        adapter_name = getattr(self.args, "adapter_name", "jokester-dpo")
        self._print_train_summary(len(pairs), new_pairs, adapter_name, min_gap, dry_run)
        self._print_pairs_sample(pairs)

        if dry_run:
            print("Dry run complete. Use without --dry-run to create training run.")
            return 0

        self._execute_training_run(client, pairs, adapter_name)
        return 0

    def _create_preference_pairs(self, min_gap: int, dry_run: bool) -> int:
        """Create preference pairs from rated jokes. Returns count created."""
        assert self._pg is not None

        service = PairingService(self.lg, self._pg, self._get_context_key())
        result = service.create_pairs(strategy="relative", min_gap=min_gap)

        print("\n=== Pairing ===")
        print(f"Rated jokes:    {result.total_rated}")
        print(f"Min gap:        {min_gap} stars")
        print(f"New pairs:      {len(result.pairs)}")

        if not result.pairs:
            return 0

        if dry_run:
            print("(Dry run - pairs not saved)")
            return len(result.pairs)

        saved = service.save_pairs(result.pairs)
        print(f"Pairs saved:    {saved}")
        return saved

    def _create_training_client(self) -> DpoClient:
        """Create DpoClient for agent's context."""
        assert self._pg is not None
        db = Database(self.lg, self._pg)
        return DpoClient(
            lg=self.lg,
            session_factory=db.session,
            context_key=self._get_context_key(),
            ensure_schema=True,
        )

    def _get_context_key(self) -> str:
        """Get context_key from agent identity config."""
        from llm_agent.core.agent import Identity

        agent_config = self._get_agent_config()
        if agent_config is None:
            raise RuntimeError("Could not load agent config")
        identity_config = agent_config.get("identity", {})
        identity = Identity.from_config(identity_config)
        return identity.context_key

    def _execute_training_run(
        self, client: DpoClient, pairs: list[tuple[Any, Any]], adapter_name: str
    ) -> None:
        """Create run and assign pairs, printing results."""
        run_info = client.create(adapter_name=adapter_name)
        pair_fact_ids = [fact.id for fact, _ in pairs]
        assigned = client.assign_pairs(run_info.id, pair_fact_ids)

        print(f"\n✓ Created training run #{run_info.id}")
        print(f"  Adapter: {adapter_name}")
        print(f"  Pairs assigned: {assigned}")
        print(f"  Status: {run_info.status}")

    def _print_train_summary(
        self,
        total_pairs: int,
        new_pairs: int,
        adapter_name: str,
        min_gap: int,
        dry_run: bool,
    ) -> None:
        """Print training run summary."""
        print("\n=== Training Run ===")
        print(f"Adapter:        {adapter_name}")
        print(f"Min gap:        {min_gap} stars")
        print(f"Total pairs:    {total_pairs}")
        if new_pairs > 0:
            verb = "would be" if dry_run else "newly"
            print(f"(includes {new_pairs} {verb} created)")
        if dry_run:
            print("(Dry run - training run not created)")
        print()

    def _print_pairs_sample(self, pairs: list[tuple[Any, Any]], max_show: int = 5) -> None:
        """Print sample of pairs to be included."""
        print("Sample pairs:")
        for i, (_fact, details) in enumerate(pairs[:max_show], 1):
            chosen = details.chosen[:50] + "..." if len(details.chosen) > 50 else details.chosen
            rejected = (
                details.rejected[:50] + "..." if len(details.rejected) > 50 else details.rejected
            )
            print(f"  {i}. margin={details.margin:.1f}")
            print(f"     chosen:   {chosen}")
            print(f"     rejected: {rejected}")
        if len(pairs) > max_show:
            print(f"  ... and {len(pairs) - max_show} more")
        print()

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

    def _get_provider_settings(self, rating_config: dict[str, Any]) -> tuple[str, str]:
        """Extract provider name and model from first enabled provider.

        Returns:
            Tuple of (provider, model). Provider is used for both routing and identification.
        """
        providers = rating_config.get("providers", {})
        for name, config in providers.items():
            if not config.get("enabled", True):
                continue
            backend_config = config.get("backend", {})
            model = backend_config.get("model", "auto")
            return (name, model)
        return ("local", "auto")

    def _create_llm_caller(self, dry_run: bool = False) -> LLMCaller | None:
        """Create LLM caller from config.

        Args:
            dry_run: If True, caller will log but not send requests.
        """
        from llm_infer.client import Factory as LLMClientFactory

        from ...core.llm import LLMCaller

        llm_config = self.app.config.get("llm", {})
        try:
            router = LLMClientFactory(self.lg).from_config(llm_config)
            return LLMCaller(self.lg, router, dry_run=dry_run)
        except Exception as e:
            self.lg.error("failed to create LLM caller", extra={"exception": e})
            return None

    def _get_rating_stats(self) -> dict[str, Any]:
        """Get rating statistics.

        Uses lateral subquery to get only the latest feedback per fact,
        avoiding duplicate counting when a joke has multiple ratings.
        """
        assert self._pg is not None
        context_key = self._get_context_key()
        sql = text("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE afd.id IS NOT NULL) as rated,
                COUNT(*) FILTER (WHERE afd.id IS NULL) as unrated,
                AVG((afd.context->>'stars')::int) as avg_stars
            FROM atomic_facts af
            LEFT JOIN LATERAL (
                SELECT afd2.id, afd2.context
                FROM atomic_feedback_details afd2
                WHERE afd2.fact_id = af.id
                ORDER BY afd2.id DESC
                LIMIT 1
            ) afd ON true
            WHERE af.context_key = :context_key
              AND af.type = 'solution'
        """)
        with self._pg.connect() as conn:
            result = conn.execute(sql, {"context_key": context_key}).fetchone()
        return {
            "total": result[0],
            "rated": result[1],
            "unrated": result[2],
            "avg_stars": float(result[3]) if result[3] else None,
        }

    def _get_rating_distribution(self) -> list[dict[str, Any]]:
        """Get rating distribution.

        Uses DISTINCT ON to count only the latest rating per fact,
        avoiding inflated counts when jokes have multiple ratings.
        """
        assert self._pg is not None
        context_key = self._get_context_key()
        sql = text("""
            WITH latest_ratings AS (
                SELECT DISTINCT ON (af.id)
                    (afd.context->>'stars')::int as stars
                FROM atomic_facts af
                JOIN atomic_feedback_details afd ON af.id = afd.fact_id
                WHERE af.context_key = :context_key
                  AND af.type = 'solution'
                ORDER BY af.id, afd.id DESC
            )
            SELECT
                stars,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
            FROM latest_ratings
            GROUP BY stars
            ORDER BY stars DESC
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, {"context_key": context_key}).fetchall()
        return [{"stars": r[0], "count": r[1], "pct": float(r[2])} for r in rows]

    def _get_training_runs(self) -> list[dict[str, Any]]:
        """Get completed training runs ordered by completion time."""
        assert self._pg is not None
        context_key = self._get_context_key()
        sql = text("""
            SELECT id, adapter_name, completed_at
            FROM dpo_runs
            WHERE context_key = :context_key AND status = 'completed'
            ORDER BY completed_at
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, {"context_key": context_key}).fetchall()
        return [{"id": r[0], "adapter_name": r[1], "completed_at": r[2]} for r in rows]

    def _get_stats_by_training_period(self, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get rating stats for each training period."""
        assert self._pg is not None
        context_key = self._get_context_key()
        periods = []

        # Build time boundaries: [None, run1_completed, run2_completed, ...]
        boundaries = [None] + [r["completed_at"] for r in runs] + [None]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            period_name = "pre-training" if i == 0 else f"after run #{runs[i - 1]['id']}"

            stats = self._get_period_stats(context_key, start, end)
            stats["period"] = period_name
            periods.append(stats)

        return periods

    def _get_period_stats(self, context_key: str, start: Any, end: Any) -> dict[str, Any]:
        """Get stats for jokes created between start and end times."""
        assert self._pg is not None

        where, params = self._build_period_conditions(context_key, start, end)
        sql = text(f"""
            SELECT COUNT(*) as count, AVG((afd.context->>'stars')::int) as avg_stars
            FROM atomic_facts af
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE {where}
        """)

        with self._pg.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            count, avg_stars = row[0], float(row[1]) if row[1] else None
            distribution = self._get_star_distribution(conn, where, params)

        return {"count": count, "avg_stars": avg_stars, "distribution": distribution}

    def _build_period_conditions(
        self, context_key: str, start: Any, end: Any
    ) -> tuple[str, dict[str, Any]]:
        """Build WHERE conditions and params for period queries."""
        conditions = ["af.context_key = :context_key", "af.type = 'solution'"]
        params: dict[str, Any] = {"context_key": context_key}

        if start is not None:
            conditions.append("af.created_at >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("af.created_at < :end")
            params["end"] = end

        return " AND ".join(conditions), params

    def _get_star_distribution(
        self, conn: Any, where: str, params: dict[str, Any]
    ) -> dict[int, int]:
        """Get distribution of 3-5 star ratings."""
        sql = text(f"""
            SELECT (afd.context->>'stars')::int as stars, COUNT(*) as cnt
            FROM atomic_facts af
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE {where} AND (afd.context->>'stars')::int >= 3
            GROUP BY (afd.context->>'stars')::int
        """)
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: r[1] for r in rows}
