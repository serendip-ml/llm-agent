"""Rating persistence backends for different storage systems."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from sqlalchemy import text

from .models import ProviderType, Result


if TYPE_CHECKING:
    from llm_learn.core import Database


class AtomicFactsBackend:
    """Persistence backend for llm-learn atomic facts model.

    Saves ratings to atomic_feedback_details table in the llm-learn schema.

    Example:
        from llm_agent.core.memory.rating import AtomicFactsBackend

        backend = AtomicFactsBackend(lg, database)
        backend.save_rating(result, source="llm_rater")
    """

    def __init__(self, lg: Logger, database: Database) -> None:
        """Initialize atomic facts backend.

        Args:
            lg: Logger instance.
            database: llm-learn Database instance.
        """
        self._lg = lg
        self._db = database

    def save_rating(
        self,
        result: Result,
        source: str = "llm_rater",
    ) -> None:
        """Save rating to atomic_feedback_details table.

        Args:
            result: Rating result to save (result.fact must be int fact_id).
            source: Source identifier for the rating.

        Raises:
            TypeError: If result.fact is not an int.
        """
        if not isinstance(result.fact, int):
            raise TypeError(
                f"AtomicFactsBackend requires result.fact to be int, got {type(result.fact)}"
            )

        fact_id = result.fact
        query, params = self._build_insert_query(result, fact_id, source)

        with self._db.session() as session:
            session.execute(text(query), params)
            session.commit()

        self._log_rating_saved(result, fact_id)

    def _build_insert_query(
        self, result: Result, fact_id: int, source: str
    ) -> tuple[str, dict[str, Any]]:
        """Build INSERT query and params for rating."""
        query = """
        INSERT INTO atomic_feedback_details (
            fact_id, signal, strength, provider_type, provider, context
        )
        VALUES (
            :fact_id, :signal, :strength, :provider_type, :provider, :context
        )
        """
        context = self._build_context(result, source)
        params = {
            "fact_id": fact_id,
            "signal": result.signal,
            "strength": result.strength,
            "provider_type": result.provider_type.value,
            "provider": result.provider,
            "context": json.dumps(context),
        }
        return query, params

    def _log_rating_saved(self, result: Result, fact_id: int) -> None:
        """Log successful rating save."""
        self._lg.debug(
            "rating saved to atomic_feedback_details",
            extra={
                "fact_id": fact_id,
                "stars": result.stars,
                "signal": result.signal,
                "provider_type": result.provider_type,
                "model": result.model,
                "provider": result.provider,
            },
        )

    def unrated_facts(
        self,
        context_key: str,
        fact_type: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield unrated facts one at a time.

        Args:
            context_key: Context key to filter facts (e.g., from LearnTrait).
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            limit: Maximum number of facts to yield.

        Yields:
            Dict with keys: id, type, category, source, content, created_at.
        """
        query, params = self._build_unrated_query(context_key, fact_type, category, limit)

        with self._db.session() as session:
            result = session.execute(text(query), params)
            for row in result:
                yield {
                    "id": row[0],
                    "type": row[1],
                    "category": row[2],
                    "source": row[3],
                    "content": row[4],
                    "created_at": row[5],
                }

    def get_unrated_count(
        self,
        context_key: str,
        fact_type: str | None = None,
        category: str | None = None,
    ) -> int:
        """Count unrated facts.

        Args:
            context_key: Context key to filter facts (e.g., from LearnTrait).
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").

        Returns:
            Number of unrated facts matching filters.
        """
        query, params = self._build_unrated_query(context_key, fact_type, category, limit=None)
        query = query.replace(
            "SELECT af.id, af.type, af.category, af.source, af.content, af.created_at",
            "SELECT COUNT(*)",
        )
        query = query.replace("LIMIT :limit", "")  # Remove limit for count

        with self._db.session() as session:
            result = session.execute(text(query), params)
            return result.scalar() or 0

    def _build_unrated_query(
        self,
        context_key: str,
        fact_type: str | None,
        category: str | None,
        limit: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build SQL query for unrated facts."""
        query = self._get_base_unrated_query()
        params = self._escape_context_key(context_key)
        query, params = self._add_query_filters(query, params, fact_type, category, limit)
        return query, params

    def _get_base_unrated_query(self) -> str:
        """Get base SQL query for unrated facts."""
        return """
        SELECT af.id, af.type, af.category, af.source, af.content, af.created_at
        FROM atomic_facts af
        LEFT JOIN atomic_feedback_details afd ON af.id = afd.fact_id
        WHERE afd.id IS NULL
          AND af.context_key LIKE :context_pattern ESCAPE '\\\\'
          AND af.active = true
        """

    def _escape_context_key(self, context_key: str) -> dict[str, Any]:
        """Escape context key for SQL LIKE pattern."""
        escaped = str(context_key).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return {"context_pattern": f"{escaped}%"}

    def _add_query_filters(
        self,
        query: str,
        params: dict[str, Any],
        fact_type: str | None,
        category: str | None,
        limit: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Add optional filters to unrated query."""
        if fact_type:
            query += " AND af.type = :fact_type"
            params["fact_type"] = fact_type

        if category:
            query += " AND af.category = :category"
            params["category"] = category

        query += " ORDER BY af.created_at ASC"

        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        return query, params

    def _build_context(
        self,
        result: Result,
        source: str,
    ) -> dict[str, Any]:
        """Build context dict for atomic facts storage.

        Note: provider_type and provider are now saved as columns,
        but we keep them in context for backwards compatibility.
        """
        return {
            "source": source,
            "rating_type": "automated" if result.provider_type == ProviderType.LLM else "manual",
            "model": result.model,
            "stars": result.stars,
            "criteria_scores": result.criteria_scores,
            "reasoning": result.reasoning,
        }

    # =========================================================================
    # Preference pairing support
    # =========================================================================

    def get_rated_unpaired_facts(
        self,
        context_key: str,
        category: str,
        min_stars: int,
        max_stars: int,
        exclude_fact_id: int | None = None,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Get rated facts that haven't been paired for DPO training.

        Args:
            context_key: Context key to filter facts.
            category: Category to match (e.g., "joke").
            min_stars: Minimum star rating (inclusive).
            max_stars: Maximum star rating (inclusive).
            exclude_fact_id: Fact ID to exclude from results.
            limit: Maximum number of facts to return.

        Returns:
            List of dicts with: id, content, stars, feedback_id.
        """
        query = """
        SELECT af.id, af.content, (afd.context->>'stars')::int as stars, afd.id as feedback_id
        FROM atomic_facts af
        JOIN atomic_feedback_details afd ON af.id = afd.fact_id
        WHERE af.context_key LIKE :context_pattern ESCAPE '\\\\'
          AND af.category = :category
          AND af.active = true
          AND (afd.context->>'stars')::int >= :min_stars
          AND (afd.context->>'stars')::int <= :max_stars
          AND (afd.context->>'paired_with') IS NULL
        """
        params = self._escape_context_key(context_key)
        params.update({
            "category": category,
            "min_stars": min_stars,
            "max_stars": max_stars,
        })

        if exclude_fact_id is not None:
            query += " AND af.id != :exclude_fact_id"
            params["exclude_fact_id"] = exclude_fact_id

        query += " ORDER BY af.created_at ASC LIMIT :limit"
        params["limit"] = limit

        with self._db.session() as session:
            result = session.execute(text(query), params)
            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "stars": row[2],
                    "feedback_id": row[3],
                }
                for row in result
            ]

    def mark_facts_paired(
        self,
        fact_id_1: int,
        fact_id_2: int,
        preference_id: int,
    ) -> None:
        """Mark two facts as paired for DPO training.

        Updates the context JSON in atomic_feedback_details to include:
        - paired_with: the other fact's ID
        - preference_id: the preference record ID

        Args:
            fact_id_1: First fact ID.
            fact_id_2: Second fact ID.
            preference_id: ID of the created preference record.
        """
        query = """
        UPDATE atomic_feedback_details
        SET context = context || :pairing_info
        WHERE fact_id = :fact_id
        """

        with self._db.session() as session:
            # Update first fact
            session.execute(
                text(query),
                {
                    "fact_id": fact_id_1,
                    "pairing_info": json.dumps({
                        "paired_with": fact_id_2,
                        "preference_id": preference_id,
                    }),
                },
            )
            # Update second fact
            session.execute(
                text(query),
                {
                    "fact_id": fact_id_2,
                    "pairing_info": json.dumps({
                        "paired_with": fact_id_1,
                        "preference_id": preference_id,
                    }),
                },
            )
            session.commit()

        self._lg.debug(
            "marked facts as paired",
            extra={
                "fact_id_1": fact_id_1,
                "fact_id_2": fact_id_2,
                "preference_id": preference_id,
            },
        )
