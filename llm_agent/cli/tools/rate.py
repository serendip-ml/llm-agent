"""Rate tool - rate agent content for training."""

import argparse
import json
from datetime import datetime
from typing import Any, cast

from appinfra import DotDict
from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_learn.core import Database
from sqlalchemy import text


class RateTool(Tool):
    """Rate unrated content from agents for training."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="rate", help_text="Rate unrated content from agents for training")
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
        parser.add_argument(
            "--auto",
            action="store_true",
            help="Automated mode - use LLM for rating (requires RatingTrait configuration)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Maximum number of items to rate in auto mode (0 = unlimited, default: 10)",
        )

    def run(self, **kwargs: Any) -> int:
        agent_name = self.args.agent_name
        context_key = self._resolve_context_key(agent_name)

        self.lg.info(
            "starting rating session",
            extra={"agent": agent_name, "context_key": context_key},
        )

        if self.args.auto:
            return self._auto_mode(context_key)
        if self.args.cont:
            return self._continuous_mode(context_key)
        return self._single_rating(context_key)

    def _resolve_context_key(self, agent_name: str) -> str:
        """Resolve context_key from agent config.

        Args:
            agent_name: Agent name from command line

        Returns:
            Resolved context key from config, or falls back to agent_name
        """
        config = self._load_agent_config(agent_name)
        if config is None:
            return agent_name

        context_key = self._extract_context_key(config, agent_name)
        return context_key if context_key else agent_name

    def _load_agent_config(self, agent_name: str) -> DotDict | None:
        """Load agent config from app config."""
        if not hasattr(self.app.config, "agents"):
            self.lg.warning(
                "no agents in config, using agent name as context_key",
                extra={"agent": agent_name},
            )
            return None

        agents = self.app.config.agents
        if agent_name not in agents:
            self.lg.warning(
                "agent config not found, using agent name as context_key",
                extra={"agent": agent_name, "available_agents": list(agents.keys())},
            )
            return None

        return cast(DotDict, agents[agent_name])

    def _extract_context_key(self, config: DotDict, agent_name: str) -> str | None:
        """Extract context_key from agent config."""
        if config and "identity" in config:
            identity = config["identity"]
            if isinstance(identity, dict) and "context_key" in identity:
                context_key = identity["context_key"]
                if isinstance(context_key, str):
                    self.lg.info(
                        "resolved context_key from agent config",
                        extra={"agent": agent_name, "context_key": context_key},
                    )
                    return context_key

        self.lg.info(
            "no context_key in agent config, using agent name",
            extra={"agent": agent_name},
        )
        return None

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

            assert isinstance(result, tuple), "result should be tuple after skip check"
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

        assert isinstance(result, tuple), "result should be tuple after None/skip check"
        signal, strength, stars = result
        stars_visual = self._format_stars(stars)
        print(f"{stars_visual} ✓ Saved")
        self._save_feedback(fact["id"], signal, strength)
        return 0

    def _build_unrated_query(self, context_key: str) -> tuple[str, dict[str, Any]]:
        """Build SQL query for unrated facts with filters."""
        query = """
        SELECT af.id, af.type, af.category, af.source, af.content,
               af.context_key, af.created_at
        FROM atomic_facts af
        LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
        WHERE afd.id IS NULL
          AND af.context_key LIKE :context_pattern ESCAPE '\\'
        """

        # Escape SQL LIKE wildcards to prevent unintended pattern matching
        escaped = context_key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params: dict[str, Any] = {"context_pattern": f"{escaped}%"}

        if self.args.category:
            query += " AND af.category = :category"
            params["category"] = self.args.category

        if self.args.type:
            query += " AND af.type = :type"
            params["type"] = self.args.type

        query += " ORDER BY af.created_at ASC LIMIT 1"
        return query, params

    def _get_next_unrated(self, context_key: str) -> dict[str, Any] | None:
        """Get next unrated fact matching filters.

        Args:
            context_key: Context key to filter by

        Returns:
            dict with fact details or None if no unrated facts
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")

        query, params = self._build_unrated_query(context_key)

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
        print(
            f"Type: {fact['type']} | Category: {fact['category'] or 'N/A'} | Source: {fact['source']}"
        )
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

            # Handle special commands
            if user_input in ("q", "quit"):
                return None
            if user_input in ("s", "skip"):
                return "skip"
            if user_input in ("g", "good", "👍"):
                return ("positive", 1.0, 5)
            if user_input in ("b", "bad", "👎"):
                return ("negative", 1.0, 1)

            # Try star rating
            result = self._parse_star_rating(user_input)
            if result:
                return result

    def _parse_star_rating(self, user_input: str) -> tuple[str, float, int] | None:
        """Parse numeric star rating (1-5)."""
        try:
            stars = int(user_input)
            if stars == 1:
                return ("negative", 1.0, 1)
            elif stars == 2:
                return ("negative", 0.5, 2)
            elif stars == 3:
                return ("dismiss", 1.0, 3)
            elif stars == 4:
                return ("positive", 0.5, 4)
            elif stars == 5:
                return ("positive", 1.0, 5)
            else:
                print("Stars must be 1-5")
                return None
        except ValueError:
            print("Invalid input. Use: 1-5 (stars), g/👍 (5★), b/👎 (1★), s/skip, q/quit")
            return None

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

    def _auto_mode(self, context_key: str) -> int:
        """Automated rating mode using LLM directly.

        Args:
            context_key: Context key for filtering facts.

        Returns:
            Exit code (0 = success).
        """
        # Load agent config to get rating configuration
        config = self._load_agent_config(self.args.agent_name)
        if config is None:
            print(f"Error: Agent config not found for '{self.args.agent_name}'")
            return 1

        if "rating" not in config:
            print(f"Error: No rating configuration found in {self.args.agent_name} config")
            print("Add a 'rating:' section to the agent's YAML config")
            return 1

        # Create LLM client for rating (use global LLM config)
        from llm_infer.client import Factory as LLMClientFactory

        llm_config = self.app.config.get("llm", {})
        llm_client = LLMClientFactory(self.lg).from_config(llm_config)

        try:
            return self._run_auto_rating_direct(config, llm_client, context_key)
        finally:
            llm_client.close()

    def _run_auto_rating_direct(self, config: Any, llm_client: Any, context_key: str) -> int:
        """Run automated rating using database and LLM directly.

        Args:
            config: Agent configuration.
            llm_client: LLM client for rating.
            context_key: Context key for filtering.

        Returns:
            Exit code (0 = success).
        """
        rating_config = config.get("rating", {})

        # Get conductor and model
        model = self._get_conductor_model(rating_config)
        if not model:
            return 1

        # Get criteria configuration
        criteria_config = self._get_criteria_config(rating_config)
        if not criteria_config:
            return 1

        # Determine how many facts to rate
        to_rate = self._determine_rating_count(context_key)
        if to_rate == 0:
            print("No unrated facts found.")
            return 0

        # Rate in batches
        total_rated = self._rate_facts_batch(
            llm_client, model, criteria_config, context_key, to_rate
        )

        print(f"\n✓ Rated {total_rated} items using automated LLM rating")
        return 0

    def _get_conductor_model(self, rating_config: dict[str, Any]) -> str | None:
        """Extract model from first conductor configuration."""
        conductors = rating_config.get("conductors", [])
        if not conductors:
            print("Error: No rating conductors configured")
            return None

        conductor = conductors[0]  # Use first conductor
        backend_config = conductor.get("backend", {})
        return cast(str, backend_config.get("model", "auto"))

    def _get_criteria_config(self, rating_config: dict[str, Any]) -> dict[str, Any] | None:
        """Get type-specific criteria configuration."""
        fact_type = self.args.type or "solution"
        criteria_config = rating_config.get("models", {}).get("atomic", {}).get(fact_type)
        if not criteria_config:
            print(f"Error: No model configured for fact type: {fact_type}")
            return None
        return cast(dict[str, Any], criteria_config)

    def _determine_rating_count(self, context_key: str) -> int:
        """Determine how many facts to rate based on limit and available unrated facts."""
        unrated_count = self._count_unrated(context_key)
        if unrated_count == 0:
            return 0

        limit = self.args.limit
        if limit == 0:
            print(f"Found {unrated_count} unrated facts. Rating all...")
            return unrated_count
        else:
            to_rate = min(limit, unrated_count)
            print(f"Found {unrated_count} unrated facts. Rating {to_rate}...")
            return cast(int, to_rate)

    def _count_unrated(self, context_key: str) -> int:
        """Count unrated facts."""
        query, params = self._build_unrated_query(context_key)
        query_count = query.replace(
            "SELECT af.id, af.type, af.category, af.source, af.content, af.created_at",
            "SELECT COUNT(*)",
        ).replace("LIMIT :limit", "")
        params_count = {k: v for k, v in params.items() if k != "limit"}

        assert self._db is not None
        with self._db.session() as session:
            result = session.execute(text(query_count), params_count)
            return result.scalar() or 0

    def _rate_facts_batch(
        self, llm_client: Any, model: str, criteria_config: Any, context_key: str, to_rate: int
    ) -> int:
        """Rate facts in batches.

        Args:
            llm_client: LLM client to use.
            model: Model name.
            criteria_config: Criteria configuration.
            context_key: Context key.
            to_rate: Number to rate.

        Returns:
            Total rated.
        """
        batch_size = 10
        total_rated = 0
        remaining = to_rate if self.args.limit > 0 else None

        while remaining is None or remaining > 0:
            batch_limit = batch_size if remaining is None else min(batch_size, remaining)
            facts = self._get_unrated_facts_batch(context_key, batch_limit)
            if not facts:
                break

            total_rated += self._rate_fact_list(llm_client, model, criteria_config, facts)

            if remaining is not None:
                remaining -= len(facts)

            print(f"Rated {total_rated}/{to_rate if self.args.limit > 0 else '?'}...")

        return total_rated

    def _rate_fact_list(
        self, llm_client: Any, model: str, criteria_config: Any, facts: list[dict[str, Any]]
    ) -> int:
        """Rate a list of facts and return count of successful ratings."""
        count = 0
        for fact in facts:
            try:
                rating, actual_model = self._rate_single_fact(
                    llm_client, model, fact["content"], criteria_config
                )
                self._save_auto_rating(fact["id"], rating, actual_model)

                stars_visual = self._format_stars(rating["stars"])
                print(f"{stars_visual} Fact {fact['id']}: {fact['content']}")

                count += 1
            except Exception as e:
                self.lg.warning(
                    "rating failed",
                    extra={"exception": e, "fact_id": fact["id"]},
                )
        return count

    def _get_unrated_facts_batch(self, context_key: str, limit: int) -> list[dict[str, Any]]:
        """Get batch of unrated facts."""
        query, params = self._build_unrated_query(context_key)
        params["limit"] = limit

        assert self._db is not None
        with self._db.session() as session:
            result = session.execute(text(query), params)
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "type": row[1],
                    "category": row[2],
                    "source": row[3],
                    "content": row[4],
                    "created_at": row[5],
                }
                for row in rows
            ]

    def _rate_single_fact(
        self, llm_client: Any, model: str, content: str, criteria_config: Any
    ) -> tuple[dict[str, Any], str]:
        """Rate a single fact using LLM.

        Returns:
            Tuple of (rating dict, actual_model) where rating contains stars, criteria_scores, reasoning
        """
        criteria_list = criteria_config.get("criteria", [])
        prompt = self._build_rating_prompt(content, criteria_config, criteria_list)

        # Call LLM (use chat_full to get actual model name)
        response = llm_client.chat_full(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,  # Higher temperature for more rating variance
        )

        # Parse response and return with actual model
        rating = self._parse_rating_json(response.content, criteria_list)
        actual_model = response.model or model
        return rating, actual_model

    def _build_rating_prompt(
        self, content: str, criteria_config: Any, criteria_list: list[dict[str, Any]]
    ) -> str:
        """Build rating prompt from criteria config and content."""
        prompt_template = criteria_config.get("prompt", "Rate the following:")
        criteria_desc = "\n".join(
            f"- {c['name']}: {c.get('description', '')} (weight: {c.get('weight', 1.0)})"
            for c in criteria_list
        )

        rating_instructions = self._get_rating_instructions()
        return f"{prompt_template}\n\nCriteria:\n{criteria_desc}\n\nContent to rate:\n{content}\n\n{rating_instructions}"

    def _get_rating_instructions(self) -> str:
        """Get standard rating instructions and JSON format."""
        return """**Important:** Be critical and use the full rating scale (1-5). Most content should be 2-4 stars.
Reserve 5 stars for truly exceptional content. Don't hesitate to give 1-2 stars for poor content.

Rating scale:
- 5 stars: Exceptional - truly outstanding, memorable
- 4 stars: Good - above average, works well
- 3 stars: Acceptable - meets basic expectations
- 2 stars: Below expectations - significant flaws
- 1 star: Poor - fails to meet criteria

Provide your rating in JSON format:
{
  "stars": <1-5>,
  "criteria_scores": {
    "criterion_name": <0.0-1.0>,
    ...
  },
  "reasoning": "<brief explanation>"
}

Respond only with the JSON, no additional text."""

    def _parse_rating_json(
        self, response: str, criteria_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Parse JSON rating from LLM response."""
        try:
            # Clean response (remove markdown code fences)
            json_str = response.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])
            if json_str.startswith("json"):
                json_str = "\n".join(json_str.split("\n")[1:])

            data = json.loads(json_str)
            return {
                "stars": data.get("stars", 3),
                "criteria_scores": data.get("criteria_scores", {}),
                "reasoning": data.get("reasoning", ""),
            }
        except json.JSONDecodeError:
            # Fallback to neutral rating
            return {
                "stars": 3,
                "criteria_scores": {c["name"]: 0.5 for c in criteria_list},
                "reasoning": "Failed to parse rating",
            }

    def _save_auto_rating(self, fact_id: int, rating: dict[str, Any], model: str) -> None:
        """Save automated rating to database."""
        from datetime import datetime

        stars = rating["stars"]
        signal, strength = self._map_stars_to_signal(stars)

        context = {
            "source": "llm_rater",
            "conductor_type": "llm",
            "model": model,
            "rating_type": "automated",
            "stars": stars,
            "criteria_scores": rating["criteria_scores"],
            "reasoning": rating["reasoning"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        query = """
        INSERT INTO atomic_feedback_details (fact_id, signal, strength, context)
        VALUES (:fact_id, :signal, :strength, :context)
        """
        params = {
            "fact_id": fact_id,
            "signal": signal,
            "strength": strength,
            "context": json.dumps(context),
        }

        assert self._db is not None
        with self._db.session() as session:
            session.execute(text(query), params)

    def _map_stars_to_signal(self, stars: int) -> tuple[str, float]:
        """Map star rating to signal and strength."""
        signal_map = {
            1: ("negative", 1.0),
            2: ("negative", 0.5),
            3: ("dismiss", 1.0),
            4: ("positive", 0.5),
            5: ("positive", 1.0),
        }
        return signal_map.get(stars, ("dismiss", 1.0))

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

        assert self._db is not None
        with self._db.session() as session:
            session.execute(text(query), params)
