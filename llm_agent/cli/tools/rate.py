"""Rate tool - rate agent content for training."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_learn.core import Database
from sqlalchemy import text


class RateTool(Tool):
    """Rate unrated content from agents for training."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="rate", help_text="Rate unrated content from agents for training"
        )
        super().__init__(parent, config)
        self._db: Database | None = None

    def configure(self) -> None:
        """Set up database connection."""
        # Get database config from app config (learn.db)
        if not hasattr(self.app, "config"):
            raise RuntimeError("App does not have config")

        db_config = self.app.config.learn.db
        if db_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        # Create database connection
        pg = PG(self.lg, db_config)
        self._db = Database(self.lg, pg)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "agent_name",
            help="Name of the agent to rate content from (e.g., 'jokester-p')",
        )
        parser.add_argument(
            "--cont",
            action="store_true",
            help="Continuous mode - rate all unrated items",
        )
        parser.add_argument(
            "--category",
            help="Filter by category (e.g., 'joke')",
        )
        parser.add_argument(
            "--type",
            help="Filter by fact type (e.g., 'solution')",
        )

    def run(self, **kwargs: Any) -> int:
        agent_name = self.args.agent_name
        context_key = self._resolve_context_key(agent_name)

        if context_key is None:
            print(f"Error: Could not resolve context key for agent '{agent_name}'")
            return 1

        self.lg.info(
            "starting rating session",
            extra={"agent": agent_name, "context_key": context_key},
        )

        if self.args.cont:
            return self._continuous_mode(context_key)
        return self._single_rating(context_key)

    def _resolve_context_key(self, agent_name: str) -> str | None:
        """Resolve context_key from agent config.

        Args:
            agent_name: Agent name from command line

        Returns:
            context_key: Resolved context key (from config or agent name)
            None: If agent config not found
        """
        # Try to load agent config from etc/agents/<agent_name>.yaml
        # Look in the etc directory relative to project root
        agent_config_path = Path(__file__).parent.parent.parent.parent / "etc" / "agents" / f"{agent_name}.yaml"

        if not agent_config_path.exists():
            self.lg.warning(
                "agent config not found, using agent name as context_key",
                extra={"agent": agent_name, "path": str(agent_config_path)},
            )
            return agent_name

        try:
            import yaml

            with agent_config_path.open() as f:
                config = yaml.safe_load(f)

            # Check for identity.context_key
            if config and "identity" in config:
                identity = config["identity"]
                if isinstance(identity, dict) and "context_key" in identity:
                    context_key = identity["context_key"]
                    self.lg.info(
                        "resolved context_key from agent config",
                        extra={"agent": agent_name, "context_key": context_key},
                    )
                    return context_key

            # Fallback to agent name
            self.lg.info(
                "no context_key in agent config, using agent name",
                extra={"agent": agent_name},
            )
            return agent_name

        except Exception as e:
            self.lg.warning(
                "failed to load agent config",
                extra={"exception": e, "agent": agent_name},
            )
            return agent_name

    def _continuous_mode(self, context_key: str) -> int:
        """Rate multiple items in continuous mode."""
        count = 0
        while True:
            fact = self._get_next_unrated(context_key)
            if fact is None:
                if count == 0:
                    print("\nNo unrated facts found.")
                else:
                    print(f"\n✓ Rated {count} items. No more unrated facts.")
                return 0

            self._display_fact(fact)
            result = self._get_rating_input()

            if result is None:  # quit
                print(f"\n✓ Rated {count} items. Exiting.")
                return 0

            if result == "skip":
                continue

            signal, strength, stars = result
            stars_visual = self._format_stars(stars)
            print(f"{stars_visual} ", end="")
            self._save_feedback(fact["id"], signal, strength)
            count += 1
            print(f"✓ Saved ({count} rated)")

    def _single_rating(self, context_key: str) -> int:
        """Rate a single item."""
        fact = self._get_next_unrated(context_key)
        if fact is None:
            print("No unrated facts found.")
            return 0

        self._display_fact(fact)
        result = self._get_rating_input()

        if result is None or result == "skip":
            print("Skipped.")
            return 0

        signal, strength, stars = result
        stars_visual = self._format_stars(stars)
        print(f"{stars_visual} ✓ Saved")
        self._save_feedback(fact["id"], signal, strength)
        return 0

    def _get_next_unrated(self, context_key: str) -> dict[str, Any] | None:
        """Get next unrated fact matching filters.

        Args:
            context_key: Context key to filter by

        Returns:
            dict with fact details or None if no unrated facts
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")

        # Build query with filters
        query = """
        SELECT af.id, af.type, af.category, af.source, af.content,
               af.context_key, af.created_at
        FROM atomic_facts af
        LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
        WHERE afd.id IS NULL
          AND af.context_key LIKE :context_pattern
        """

        params: dict[str, Any] = {"context_pattern": f"{context_key}%"}

        # Add optional filters
        if self.args.category:
            query += " AND af.category = :category"
            params["category"] = self.args.category

        if self.args.type:
            query += " AND af.type = :type"
            params["type"] = self.args.type

        query += " ORDER BY af.created_at ASC LIMIT 1"

        with self._db.session() as session:
            result = session.execute(text(query), params)
            row = result.fetchone()

            if row is None:
                return None

            return {
                "id": row[0],
                "type": row[1],
                "category": row[2],
                "source": row[3],
                "content": row[4],
                "context_key": row[5],
                "created_at": row[6],
            }

    def _display_fact(self, fact: dict[str, Any]) -> None:
        """Display fact in generic format."""
        print("\n" + "=" * 80)
        print(f"Type: {fact['type']} | Category: {fact['category'] or 'N/A'} | Source: {fact['source']}")
        print(f"Context: {fact['context_key']}")

        # Format datetime
        created_at = fact["created_at"]
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_at_str = str(created_at)
        print(f"Created: {created_at_str}")
        print("-" * 80)

        # Display content
        content = fact["content"]
        # Try to parse as JSON for pretty printing
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2))
        except (json.JSONDecodeError, TypeError):
            # Plain text content
            print(content)

        print("=" * 80)

    def _get_rating_input(self) -> tuple[str, float, int] | str | None:
        """Get rating input from user.

        Returns:
            tuple[str, float, int]: (signal, strength, stars) for positive/negative/dismiss
            str: "skip" to skip this item without recording
            None: to quit
        """
        prompt = "\nRating (1-5 stars, g/👍=★★★★★, b/👎=★☆☆☆☆, s/skip, q/quit): "

        while True:
            try:
                user_input = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None

            if not user_input:
                continue

            # Quit
            if user_input in ("q", "quit"):
                return None

            # Skip (don't record anything)
            if user_input in ("s", "skip"):
                return "skip"

            # Binary shortcuts
            if user_input in ("g", "good", "👍"):
                return ("positive", 1.0, 5)  # 5 stars

            if user_input in ("b", "bad", "👎"):
                return ("negative", 1.0, 1)  # 1 star

            # Star rating (1-5)
            try:
                stars = int(user_input)
                if stars == 1:
                    return ("negative", 1.0, 1)  # 1★ - bad
                elif stars == 2:
                    return ("negative", 0.5, 2)  # 2★ - not good
                elif stars == 3:
                    return ("dismiss", 1.0, 3)  # 3★ - neutral/meh
                elif stars == 4:
                    return ("positive", 0.5, 4)  # 4★ - good
                elif stars == 5:
                    return ("positive", 1.0, 5)  # 5★ - excellent
                else:
                    print("Stars must be 1-5")
                    continue
            except ValueError:
                pass

            print("Invalid input. Use: 1-5 (stars), g/👍 (5★), b/👎 (1★), s/skip, q/quit")

    def _format_stars(self, stars: int) -> str:
        """Format star rating visually.

        Args:
            stars: Number of stars (0-5)

        Returns:
            Star visualization (e.g., "★★★☆☆")
        """
        filled = "★" * stars
        empty = "☆" * (5 - stars)
        return filled + empty

    def _save_feedback(self, fact_id: int, signal: str, strength: float) -> None:
        """Save feedback to database.

        Args:
            fact_id: ID of the fact being rated
            signal: "positive" or "negative"
            strength: Confidence [0.0-1.0]
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")

        # Insert feedback_details directly for the fact being rated
        query = """
        INSERT INTO atomic_feedback_details (fact_id, signal, strength, context)
        VALUES (:fact_id, :signal, :strength, :context)
        """

        context = {
            "source": "cli_rate_tool",
            "rating_type": "hybrid",
        }

        params = {
            "fact_id": fact_id,
            "signal": signal,
            "strength": strength,
            "context": json.dumps(context),
        }

        with self._db.session() as session:
            session.execute(text(query), params)
