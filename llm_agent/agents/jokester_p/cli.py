"""CLI tool for jokester-p agent."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from sqlalchemy import text

from llm_agent.core.memory.rating import BatchRatingService


if TYPE_CHECKING:
    from ...core.llm import LLMCaller


class JokesterCLI(Tool):
    """CLI commands for jokester-p agent.

    Provides agent-specific commands:
    - stats: Show joke rating statistics
    - rate: Rate unrated jokes in batches
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
        self._add_stats_args(subparsers)
        self._add_rate_args(subparsers)

    def _add_stats_args(self, subparsers: Any) -> None:
        """Add stats command arguments."""
        p = subparsers.add_parser("stats", help="Show joke rating statistics")
        p.add_argument(
            "--max-chars",
            type=int,
            default=140,
            help="Only count jokes under this character length (default: 140)",
        )
        p.add_argument(
            "--model",
            type=str,
            help="Filter by model (case-insensitive, supports * wildcard, e.g. 'qwen*14b')",
        )

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

    def run(self, **kwargs: Any) -> int:
        command = self.args.command

        if command == "stats":
            return self._cmd_stats()
        elif command == "rate":
            return self._cmd_rate()
        else:
            print("Usage: agent jokester-p <command>")
            print("Commands: stats, rate")
            return 1

    def _cmd_stats(self) -> int:
        """Show joke rating statistics."""
        assert self._pg is not None

        max_chars = getattr(self.args, "max_chars", 130)
        model = getattr(self.args, "model", None)
        stats = self._get_rating_stats(exclude_haiku=True, max_chars=max_chars, model=model)
        dist = self._get_rating_distribution(exclude_haiku=True, max_chars=max_chars, model=model)

        self._print_stats_header(max_chars, model)
        self._print_rating_summary(stats, dist)
        self._print_stats_by_adapter(max_chars=max_chars, model=model)
        if not model:  # Only show haiku stats when not filtering by model
            self._print_haiku_stats(max_chars=max_chars)

        return 0

    def _print_stats_header(self, max_chars: int, model_filter: str | None = None) -> None:
        """Print stats header with model and adapter info."""
        context_key = self._get_context_key()
        model = self._get_latest_model(context_key) or "unknown"
        adapter = self._get_latest_adapter_from_jokes(context_key)
        print("\n=== Jokester-p Stats ===\n")
        filter_info = f"<{max_chars} chars"
        if model_filter:
            filter_info += f", model~'{model_filter}'"
        print(f"Model:   {model}, {filter_info}")
        if adapter:
            print(f"Adapter: {adapter['name']} (md5: {adapter['md5']})")
        print()

    def _print_rating_summary(self, stats: dict[str, Any], dist: list[dict[str, Any]]) -> None:
        """Print rating counts and distribution."""
        print(f"Total jokes:  {stats['total']}")
        print(f"Rated:        {stats['rated']}")
        print(f"Unrated:      {stats['unrated']}")
        avg = f"{stats['avg_stars']:.2f}" if stats["avg_stars"] else "N/A"
        print(f"Avg stars:    {avg}")
        print()
        print("Rating Distribution:")
        for row in dist:
            stars_visual = "★" * row["stars"] + "☆" * (5 - row["stars"])
            print(f"  {stars_visual}  {row['count']:4d}  ({row['pct']:.1f}%)")

    def _print_stats_by_adapter(
        self, max_chars: int | None = None, model: str | None = None
    ) -> None:
        """Print stats broken down by adapter md5."""
        adapter_stats = self._get_stats_by_adapter(max_chars=max_chars, model=model)
        if not adapter_stats:
            return

        print("\n=== By Adapter ===\n")
        for stats in adapter_stats:
            if stats["count"] == 0:
                continue
            avg = f"{stats['avg_stars']:.2f}" if stats["avg_stars"] else "N/A"
            dist = stats.get("distribution", {})
            total = stats["count"]
            stars_str = self._format_star_distribution(dist, total)
            # Show short md5 (first 8 chars) or "base" for no adapter
            label = stats["md5"][:8] if stats["md5"] else "base"
            print(f"{label:20s}  {stats['count']:5d} jokes  avg={avg}  {stars_str}")
        print()

    def _format_star_distribution(self, dist: dict[int, int], total: int) -> str:
        """Format star distribution as: 5★:0(0.0%)  4★:18(0.2%)  3★:504(6.3%)  ..."""
        parts = []
        for stars in (5, 4, 3, 2, 1):
            count = dist.get(stars, 0)
            pct = count * 100 / total if total else 0
            parts.append(f"{stars}★:{count}({pct:.1f}%)")
        return "  ".join(parts)

    def _print_haiku_stats(self, max_chars: int | None = None) -> None:
        """Print stats for Haiku-generated jokes separately."""
        assert self._pg is not None

        stats = self._query_haiku_stats(max_chars)
        if stats["total"] == 0:
            return

        chars_label = f", <{max_chars} chars" if max_chars else ""
        print(f"=== Haiku (Reference{chars_label}) ===\n")
        avg_str = f"{float(stats['avg_stars']):.2f}" if stats["avg_stars"] else "N/A"
        stars_str = self._format_star_distribution(stats["dist"], stats["rated"])
        print(f"{'haiku':20s}  {stats['total']:5d} jokes  avg={avg_str}  {stars_str}")
        print()

    def _get_latest_adapter_from_jokes(self, context_key: str) -> dict[str, Any] | None:
        """Get adapter info from most recent joke."""
        assert self._pg is not None
        sql = text("""
            SELECT t.adapter_info
            FROM agent_jokester_training t
            JOIN atomic_facts af ON t.fact_id = af.id
            WHERE af.context_key = :context_key AND t.adapter_info IS NOT NULL
            ORDER BY af.id DESC
            LIMIT 1
        """)
        with self._pg.connect() as conn:
            row = conn.execute(sql, {"context_key": context_key}).fetchone()
        if not row or not row[0]:
            return None
        info = row[0]
        return {"name": info.get("actual") or info.get("requested"), "md5": info.get("md5")}

    def _get_stats_by_adapter(
        self, max_chars: int | None = None, model: str | None = None
    ) -> list[dict[str, Any]]:
        """Get rating stats grouped by adapter md5."""
        assert self._pg is not None
        context_key = self._get_context_key()
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params)
        model_join, model_filter = self._build_model_filter(model, params)

        sql = self._build_adapter_stats_sql(chars_join, model_join, chars_filter, model_filter)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return self._build_adapter_results(conn, context_key, rows, params, max_chars, model)

    def _build_adapter_stats_sql(
        self, chars_join: str, model_join: str, chars_filter: str, model_filter: str
    ) -> Any:
        """Build SQL for adapter stats aggregation."""
        return text(f"""
            SELECT t.adapter_info->>'md5' as adapter_md5,
                   COUNT(*) as total, COUNT(afd.id) as rated,
                   AVG((afd.context->>'stars')::int) as avg_stars
            FROM atomic_facts af
            JOIN agent_jokester_training t ON t.fact_id = af.id
            {chars_join} {model_join}
            LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key AND af.type = 'solution'
              {chars_filter} {model_filter}
            GROUP BY t.adapter_info->>'md5'
            ORDER BY MIN(af.id)
        """)

    def _build_adapter_results(
        self,
        conn: Any,
        context_key: str,
        rows: list[Any],
        params: dict[str, Any],
        max_chars: int | None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build result dicts with distribution for each adapter."""
        results = []
        for row in rows:
            md5, total, rated, avg_stars = row
            dist = self._query_adapter_distribution(
                conn, context_key, md5, params, max_chars, model
            )
            results.append(
                {
                    "md5": md5,
                    "count": total,
                    "rated": rated,
                    "avg_stars": avg_stars,
                    "distribution": dist,
                }
            )
        return results

    def _query_adapter_distribution(
        self,
        conn: Any,
        context_key: str,
        adapter_md5: str | None,
        base_params: dict[str, Any],
        max_chars: int | None = None,
        model: str | None = None,
    ) -> dict[int, int]:
        """Query star distribution for a specific adapter md5."""
        params = dict(base_params)
        chars_join, chars_filter = self._build_chars_filter(max_chars, params)
        model_join, model_filter = self._build_model_filter(model, params)

        if adapter_md5 is None:
            md5_filter = "AND t.adapter_info->>'md5' IS NULL"
        else:
            md5_filter = "AND t.adapter_info->>'md5' = :adapter_md5"
            params["adapter_md5"] = adapter_md5

        sql = text(f"""
            SELECT (afd.context->>'stars')::int as stars, COUNT(*) as cnt
            FROM atomic_facts af
            JOIN agent_jokester_training t ON t.fact_id = af.id
            {chars_join}
            {model_join}
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key AND af.type = 'solution'
              {chars_filter} {model_filter} {md5_filter}
            GROUP BY (afd.context->>'stars')::int
        """)
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: r[1] for r in rows}

    def _get_latest_model(self, context_key: str) -> str | None:
        """Get base model name from most recent joke."""
        assert self._pg is not None
        sql = text("""
            SELECT t.base_model
            FROM agent_jokester_training t
            JOIN atomic_facts af ON t.fact_id = af.id
            WHERE af.context_key = :context_key
            ORDER BY af.id DESC
            LIMIT 1
        """)
        with self._pg.connect() as conn:
            row = conn.execute(sql, {"context_key": context_key}).fetchone()
        return row[0] if row else None

    def _query_haiku_stats(self, max_chars: int | None = None) -> dict[str, Any]:
        """Query stats for Haiku-generated jokes."""
        assert self._pg is not None
        context_key = self._get_context_key()
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params)

        sql = text(f"""
            SELECT COUNT(*) as total, COUNT(afd.id) as rated,
                   AVG((afd.context->>'stars')::int) as avg_stars
            FROM atomic_facts af
            JOIN agent_jokester_model_usage u ON u.fact_id = af.id
            {chars_join}
            LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key AND af.type = 'solution'
              AND u.model_name LIKE '%haiku%' {chars_filter}
        """)
        with self._pg.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            total, rated, avg_stars = row[0], row[1], row[2]
            dist = self._query_haiku_distribution(
                conn, context_key, chars_join, chars_filter, max_chars
            )

        return {"total": total, "rated": rated, "avg_stars": avg_stars, "dist": dist}

    def _query_haiku_distribution(
        self,
        conn: Any,
        context_key: str,
        chars_join: str,
        chars_filter: str,
        max_chars: int | None = None,
    ) -> dict[int, int]:
        """Query star distribution for Haiku jokes."""
        params: dict[str, Any] = {"context_key": context_key}
        if max_chars:
            params["max_chars"] = max_chars
        sql = text(f"""
            SELECT (afd.context->>'stars')::int as stars, COUNT(*) as cnt
            FROM atomic_facts af
            JOIN agent_jokester_model_usage u ON u.fact_id = af.id
            {chars_join}
            JOIN atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key AND af.type = 'solution'
              AND u.model_name LIKE '%haiku%' {chars_filter}
            GROUP BY (afd.context->>'stars')::int
        """)
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: r[1] for r in rows}

    def _build_chars_filter(
        self, max_chars: int | None, params: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """Build JOIN and WHERE clause for character length filtering.

        If params dict provided, adds max_chars to it automatically.
        """
        if not max_chars:
            return "", ""
        if params is not None:
            params["max_chars"] = max_chars
        chars_join = "JOIN atomic_solution_details asd ON asd.fact_id = af.id"
        chars_filter = "AND length(asd.answer_text) < :max_chars"
        return chars_join, chars_filter

    def _build_model_filter(
        self, model: str | None, params: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """Build JOIN and WHERE clause for model filtering.

        Supports glob-style patterns (e.g., "qwen*14b") with case-insensitive matching.
        If params dict provided, adds model_pattern to it automatically.
        """
        if not model:
            return "", ""
        if params is not None:
            # Normalize to lowercase and convert glob * to SQL %
            pattern = model.lower().replace("*", "%")
            # Wrap with % if no wildcards present (substring match)
            if "%" not in pattern:
                pattern = f"%{pattern}%"
            params["model_pattern"] = pattern
        model_join = "JOIN agent_jokester_model_usage mu ON mu.fact_id = af.id"
        model_filter = "AND LOWER(mu.model_name) LIKE :model_pattern"
        return model_join, model_filter

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
            service = BatchRatingService(
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

    def _run_rating(self, service: BatchRatingService, batch_size: int) -> int:
        """Execute rating and print results."""
        unrated = service.get_unrated(self.args.limit)
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
                stars = "★" * r.score + "☆" * (5 - r.score)
                joke = r.content[:90] + "..." if len(r.content) > 90 else r.content
                print(f"{r.id}  ({r.score})  {stars}  {joke}")

        print(f"\n✓ Rated: {total_rated}")
        print(f"  Batches: {batch_count}")

        return 0

    def _get_context_key(self) -> str:
        """Get context_key from agent identity config."""
        from llm_agent.core.agent import Identity

        agent_config = self._get_agent_config()
        if agent_config is None:
            raise RuntimeError("Could not load agent config")
        identity_config = agent_config.get("identity", {})
        identity = Identity.from_config(identity_config)
        return identity.context_key

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

    def _get_rating_stats(
        self,
        exclude_haiku: bool = False,
        max_chars: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Get rating statistics using lateral subquery for latest feedback per fact."""
        assert self._pg is not None
        context_key = self._get_context_key()
        haiku_filter = self._build_haiku_filter(exclude_haiku)
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params)
        model_join, model_filter = self._build_model_filter(model, params)

        sql = text(f"""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE afd.id IS NOT NULL) as rated,
                   COUNT(*) FILTER (WHERE afd.id IS NULL) as unrated,
                   AVG((afd.context->>'stars')::int) as avg_stars
            FROM atomic_facts af
            {chars_join}
            {model_join}
            LEFT JOIN LATERAL (
                SELECT afd2.id, afd2.context FROM atomic_feedback_details afd2
                WHERE afd2.fact_id = af.id ORDER BY afd2.id DESC LIMIT 1
            ) afd ON true
            WHERE af.context_key = :context_key AND af.type = 'solution'
              {haiku_filter} {chars_filter} {model_filter}
        """)
        with self._pg.connect() as conn:
            result = conn.execute(sql, params).fetchone()
        return {
            "total": result[0],
            "rated": result[1],
            "unrated": result[2],
            "avg_stars": float(result[3]) if result[3] else None,
        }

    def _build_haiku_filter(self, exclude_haiku: bool) -> str:
        """Build WHERE clause to exclude Haiku jokes."""
        if not exclude_haiku:
            return ""
        return """AND NOT EXISTS (
            SELECT 1 FROM agent_jokester_model_usage u
            WHERE u.fact_id = af.id AND u.model_name LIKE '%haiku%'
        )"""

    def _get_rating_distribution(
        self,
        exclude_haiku: bool = False,
        max_chars: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get rating distribution using DISTINCT ON for latest rating per fact."""
        assert self._pg is not None
        context_key = self._get_context_key()
        haiku_filter = self._build_haiku_filter(exclude_haiku)
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params)
        model_join, model_filter = self._build_model_filter(model, params)

        sql = text(f"""
            WITH latest_ratings AS (
                SELECT DISTINCT ON (af.id) (afd.context->>'stars')::int as stars
                FROM atomic_facts af
                JOIN atomic_feedback_details afd ON af.id = afd.fact_id
                {chars_join}
                {model_join}
                WHERE af.context_key = :context_key AND af.type = 'solution'
                  {haiku_filter} {chars_filter} {model_filter}
                ORDER BY af.id, afd.id DESC
            )
            SELECT stars, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
            FROM latest_ratings GROUP BY stars ORDER BY stars DESC
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{"stars": r[0], "count": r[1], "pct": float(r[2])} for r in rows]
