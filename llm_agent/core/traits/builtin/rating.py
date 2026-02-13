"""Rating trait for automated LLM-based content rating."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from appinfra import DotDict
from llm_infer.client import LLMClient
from sqlalchemy import text

from ..base import BaseTrait


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent


# Type alias for Rating configuration
RatingConfig = DotDict
"""Rating configuration as DotDict.

Expected fields:
    backends: List of rating backend configurations
        - name: Backend identifier (e.g., "claude-sonnet")
        - model: Model name for rating
        - criteria: List of rating criteria (e.g., ["humor", "originality"])
    batch_size: Number of items to rate per batch (default: 10)
    auto: Skip confirmation prompts (default: False)
"""


@dataclass
class RatingCriteria:
    """Rating criteria for evaluating content."""

    name: str
    description: str
    weight: float = 1.0


@dataclass
class TypeCriteria:
    """Type-specific criteria configuration."""

    fact_type: str  # e.g., "solution", "prediction", "feedback"
    prompt: str  # Type-specific base prompt
    criteria: list[RatingCriteria]


@dataclass
class RatingConductor:
    """Configuration for a rating conductor."""

    conductor_type: str  # "llm", "human", "api", etc.
    model: str
    backend_config: DotDict
    enabled: bool = True


@dataclass
class RatingResult:
    """Result from rating a single fact."""

    fact_id: int
    signal: Literal["positive", "negative", "dismiss"]
    strength: float
    stars: int
    criteria_scores: dict[str, float]
    reasoning: str
    model: str  # Model that performed the rating


class RatingTrait(BaseTrait):
    """Automated LLM-based content rating trait.

    Provides automated rating capabilities using LLM backends to evaluate
    agent-generated content. Stores ratings in atomic_feedback_details with
    source tracking for multi-rater scenarios.

    **IMPORTANT:** RatingTrait depends on LearnTrait. LearnTrait must be attached
    to the agent BEFORE RatingTrait.

    Capabilities:
        - rate_unrated(): Find and rate unrated content automatically
        - rate_fact(): Rate a specific fact by ID
        - get_unrated_count(): Count unrated facts
        - Multi-backend support: Use multiple LLMs for rating comparison

    Example:
        from llm_agent.core.traits import RatingTrait, LearnTrait

        # Configure rating
        rating_config = RatingConfig(
            backends=[{
                "name": "claude-sonnet",
                "model": "claude-sonnet-4.5",
                "criteria": [
                    {"name": "humor", "description": "Is it funny?"},
                    {"name": "originality", "description": "Is it original?"}
                ]
            }],
            batch_size=10,
            auto=True
        )

        # Attach traits
        agent.add_trait(LearnTrait(agent, learn_config))
        agent.add_trait(RatingTrait(agent, rating_config))
        agent.start()

        # Rate unrated content
        results = agent.get_trait(RatingTrait).rate_unrated(limit=10)

    Lifecycle:
        - on_start(): Initialize rating backends and validate configuration
        - on_stop(): Cleanup resources
    """

    def __init__(
        self,
        agent: Agent,
        config: RatingConfig | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize rating trait.

        Args:
            agent: The agent this trait belongs to.
            config: Rating configuration (None = use defaults).
            llm_config: Platform LLM configuration for creating LLM client.
        """
        super().__init__(agent)
        self.config = config or RatingConfig()
        self._llm_config = llm_config
        self._conductors: list[RatingConductor] = []
        self._type_criteria: dict[str, TypeCriteria] = {}  # fact_type -> TypeCriteria
        self._llm_client: LLMClient | None = None

    def on_start(self) -> None:
        """Initialize rating conductors and LLM client.

        Raises:
            TraitNotFoundError: If LearnTrait is not attached.
            RuntimeError: If LLM config was not provided during initialization.
        """
        from llm_infer.client import Factory as LLMClientFactory

        from .learn import LearnTrait

        # Require LearnTrait for database access
        self.agent.require_trait(LearnTrait)

        # Create LLM client from platform config passed during initialization
        if not self._llm_config:
            raise RuntimeError("RatingTrait requires llm_config to be passed during initialization")
        self._llm_client = LLMClientFactory(self.agent.lg).from_config(self._llm_config)

        # Parse conductor and criteria configurations
        self._conductors = self._parse_conductors()
        self._type_criteria = self._parse_type_criteria()

        self.agent.lg.debug(
            "rating trait started",
            extra={
                "conductors": len(self._conductors),
                "enabled": sum(1 for c in self._conductors if c.enabled),
                "types": list(self._type_criteria.keys()),
                "auto": self.config.get("auto", False),
            },
        )

    def on_stop(self) -> None:
        """Clean up rating resources."""
        self._conductors = []
        self._type_criteria = {}
        if self._llm_client:
            self._llm_client.close()
            self._llm_client = None
        self.agent.lg.debug("rating trait stopped")

    def _parse_conductors(self) -> list[RatingConductor]:
        """Parse conductor configurations from config.

        Format:
          conductors:
            - type: llm
              backend: ${llm.backends.local}
              enabled: true
        """
        conductors_config = self.config.get("conductors", [])
        if not conductors_config:
            return []

        conductors = []
        for conductor_cfg in conductors_config:
            conductor = self._parse_conductor(conductor_cfg)
            if conductor:
                conductors.append(conductor)

        return conductors

    def _parse_conductor(self, config: dict[str, Any] | DotDict) -> RatingConductor | None:
        """Parse a single conductor configuration.

        Args:
            config: Conductor config with type, backend, enabled.

        Returns:
            RatingConductor or None if invalid.
        """
        if isinstance(config, dict) and not isinstance(config, DotDict):
            config = DotDict(config)

        # Get conductor type (llm, human, api, etc.)
        conductor_type = config.get("type", "llm")

        # Extract backend config (from ${llm.backends.local} expansion)
        backend_config = config.get("backend")
        if not backend_config:
            self.agent.lg.warning("conductor missing backend config, skipping")
            return None

        if isinstance(backend_config, dict) and not isinstance(backend_config, DotDict):
            backend_config = DotDict(backend_config)

        # Get model name from backend config
        model = backend_config.get("model", "auto")

        # Get enabled flag (default: true)
        enabled = config.get("enabled", True)

        return RatingConductor(
            conductor_type=conductor_type,
            model=str(model),
            backend_config=backend_config,
            enabled=enabled,
        )

    def _parse_type_criteria(self) -> dict[str, TypeCriteria]:
        """Parse type-specific criteria from config.

        Format:
          models:
            atomic:
              solution:
                prompt: "You are rating a joke..."
                criteria:
                  - name: humor
                    description: "Is it funny?"
                    weight: 1.0

        Returns:
            Dict mapping fact_type to TypeCriteria.
        """
        criteria_config = self.config.get("models", {})
        type_criteria = {}

        # Parse atomic fact types
        atomic_config = criteria_config.get("atomic", {})
        for fact_type, type_config in atomic_config.items():
            parsed = self._parse_single_type_criteria(fact_type, type_config)
            if parsed:
                type_criteria[fact_type] = parsed

        return type_criteria

    def _parse_single_type_criteria(
        self, fact_type: str, config: dict[str, Any] | DotDict
    ) -> TypeCriteria | None:
        """Parse criteria for a single fact type.

        Args:
            fact_type: Type of fact (e.g., "solution", "prediction").
            config: Type-specific config with prompt and criteria.

        Returns:
            TypeCriteria or None if invalid.
        """
        if isinstance(config, dict) and not isinstance(config, DotDict):
            config = DotDict(config)

        # Get type-specific prompt
        prompt = self._get_or_default_prompt(fact_type, config)

        # Parse criteria list
        criteria = self._parse_criteria_list(config.get("criteria", []))
        if not criteria:
            self.agent.lg.warning(
                "no criteria for fact type, skipping",
                extra={"fact_type": fact_type},
            )
            return None

        return TypeCriteria(fact_type=fact_type, prompt=prompt, criteria=criteria)

    def _get_or_default_prompt(self, fact_type: str, config: dict[str, Any] | DotDict) -> str:
        """Get prompt from config or return default."""
        prompt = config.get("prompt", "")
        if not prompt:
            self.agent.lg.warning(
                "missing prompt for fact type, using default",
                extra={"fact_type": fact_type},
            )
            prompt = f"Rate the following {fact_type}:"
        return str(prompt)

    def _parse_criteria_list(self, criteria_list: list[Any]) -> list[RatingCriteria]:
        """Parse criteria list into RatingCriteria objects."""
        criteria = []
        for crit_cfg in criteria_list:
            if isinstance(crit_cfg, str):
                criteria.append(RatingCriteria(name=crit_cfg, description=f"Evaluate {crit_cfg}"))
            else:
                criteria.append(
                    RatingCriteria(
                        name=crit_cfg["name"],
                        description=crit_cfg.get("description", f"Evaluate {crit_cfg['name']}"),
                        weight=crit_cfg.get("weight", 1.0),
                    )
                )
        return criteria

    # =========================================================================
    # Rating operations
    # =========================================================================

    def get_unrated_count(
        self,
        fact_type: str | None = None,
        category: str | None = None,
    ) -> int:
        """Count unrated facts.

        Args:
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").

        Returns:
            Number of unrated facts matching filters.
        """
        from .learn import LearnTrait

        learn_trait = self.agent.require_trait(LearnTrait)
        query, params = self._build_unrated_query(fact_type, category, limit=None)
        query = query.replace("SELECT af.*", "SELECT COUNT(*)")
        query = query.replace("LIMIT :limit", "")  # Remove limit for count

        with learn_trait.learn.database.session() as session:
            result = session.execute(text(query), params)
            return result.scalar() or 0

    def rate_unrated(
        self,
        limit: int = 10,
        fact_type: str | None = None,
        category: str | None = None,
        conductor_index: int = 0,
    ) -> list[RatingResult]:
        """Rate unrated facts automatically using configured conductors.

        Args:
            limit: Maximum number of facts to rate.
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            conductor_index: Index of conductor to use (default: 0 = first enabled).

        Returns:
            List of rating results.
        """
        # Select conductor
        conductor = self._select_conductor(conductor_index)

        # Get unrated facts
        facts = self._get_unrated_facts(limit, fact_type, category)

        if not facts:
            return []

        # Rate each fact
        results = []
        for fact in facts:
            try:
                # Use fact type from the fact data
                result = self.rate_fact(fact["id"], fact["content"], fact["type"], conductor)
                self._save_rating(result, conductor)
                results.append(result)
            except Exception as e:
                self.agent.lg.warning(
                    "rating failed",
                    extra={
                        "exception": e,
                        "fact_id": fact["id"],
                        "fact_type": fact["type"],
                        "conductor_type": conductor.conductor_type,
                    },
                )

        return results

    def rate_fact(
        self,
        fact_id: int,
        content: str,
        fact_type: str,
        conductor: RatingConductor,
    ) -> RatingResult:
        """Rate a specific fact using an LLM conductor.

        Args:
            fact_id: ID of the fact to rate.
            content: Content to rate.
            fact_type: Type of fact (e.g., "solution", "prediction").
            conductor: Rating conductor to use.

        Returns:
            Rating result with scores and reasoning.
        """
        if self._llm_client is None:
            raise RuntimeError("RatingTrait not started")

        # Get type-specific criteria
        type_criteria = self._type_criteria.get(fact_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {fact_type}")

        # Build rating prompt with type-specific context
        prompt = self._build_rating_prompt(content, type_criteria)

        # Get LLM rating
        response = self._llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=conductor.model,
            temperature=0.3,  # Lower temperature for consistent ratings
        )

        # Parse rating response
        return self._parse_rating_response(
            fact_id, response, type_criteria.criteria, conductor.model
        )

    def rate_fact_with_all_conductors(
        self, fact_id: int, content: str, fact_type: str = "solution"
    ) -> list[RatingResult]:
        """Rate a fact with all enabled conductors and save ratings.

        This is the primary method for agents to use for inline rating.
        Rates the content with each enabled LLM conductor and saves all ratings.

        Args:
            fact_id: ID of the fact to rate.
            content: Content to rate (e.g., joke text).
            fact_type: Type of fact (default: "solution").

        Returns:
            List of rating results from each conductor.
        """
        if self._llm_client is None:
            raise RuntimeError("RatingTrait not started")

        results = []
        enabled_conductors = [
            c for c in self._conductors if c.enabled and c.conductor_type == "llm"
        ]

        for conductor in enabled_conductors:
            result = self._rate_and_save_with_conductor(fact_id, content, fact_type, conductor)
            if result:
                results.append(result)

        return results

    def _rate_and_save_with_conductor(
        self, fact_id: int, content: str, fact_type: str, conductor: RatingConductor
    ) -> RatingResult | None:
        """Rate a fact with a single conductor and save the result."""
        try:
            # Rate with this conductor
            result = self.rate_fact(fact_id, content, fact_type, conductor)

            # Save rating to database
            self._save_rating(result, conductor)

            self.agent.lg.debug(
                "rated fact with conductor",
                extra={
                    "fact_id": fact_id,
                    "conductor_model": conductor.model,
                    "stars": result.stars,
                },
            )

            return result

        except Exception as e:
            self.agent.lg.warning(
                "rating failed for conductor",
                extra={
                    "exception": e,
                    "fact_id": fact_id,
                    "conductor_type": conductor.conductor_type,
                },
            )
            return None

    def _select_conductor(self, index: int = 0) -> RatingConductor:
        """Select rating conductor by index.

        Args:
            index: Index of enabled conductor to use (default: 0).

        Returns:
            Selected conductor.

        Raises:
            RuntimeError: If no enabled conductors configured.
            IndexError: If index out of range.
        """
        enabled_conductors = [c for c in self._conductors if c.enabled]

        if not enabled_conductors:
            raise RuntimeError("No enabled rating conductors configured")

        if index >= len(enabled_conductors):
            raise IndexError(
                f"Conductor index {index} out of range (have {len(enabled_conductors)} enabled)"
            )

        return enabled_conductors[index]

    def _get_unrated_facts(
        self,
        limit: int,
        fact_type: str | None,
        category: str | None,
    ) -> list[dict[str, Any]]:
        """Get unrated facts from database."""
        from .learn import LearnTrait

        learn_trait = self.agent.require_trait(LearnTrait)
        query, params = self._build_unrated_query(fact_type, category, limit)

        with learn_trait.learn.database.session() as session:
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

    def _build_unrated_query(
        self,
        fact_type: str | None,
        category: str | None,
        limit: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build SQL query for unrated facts."""
        from .learn import LearnTrait

        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = learn_trait.learn.context.context_key

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

    def _escape_context_key(self, context_key: str | None) -> dict[str, Any]:
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

    def _build_rating_prompt(self, content: str, type_criteria: TypeCriteria) -> str:
        """Build prompt for LLM rating with type-specific context.

        Args:
            content: Content to rate.
            type_criteria: Type-specific criteria with prompt and criteria list.

        Returns:
            Complete rating prompt.
        """
        criteria_desc = "\n".join(
            f"- {c.name}: {c.description} (weight: {c.weight})" for c in type_criteria.criteria
        )

        return f"""{type_criteria.prompt}

Criteria:
{criteria_desc}

Content to rate:
{content}

Provide your rating in JSON format:
{{
  "stars": <1-5>,
  "criteria_scores": {{
    "criterion_name": <0.0-1.0>,
    ...
  }},
  "reasoning": "<brief explanation>"
}}

Rating scale:
- 5 stars (★★★★★): Excellent - exceeds expectations on all criteria
- 4 stars (★★★★☆): Good - meets expectations well
- 3 stars (★★★☆☆): Neutral - acceptable but unremarkable
- 2 stars (★★☆☆☆): Below expectations - has issues
- 1 star (★☆☆☆☆): Poor - fails to meet criteria

Respond only with the JSON, no additional text."""

    def _parse_rating_response(
        self,
        fact_id: int,
        response: str,
        criteria: list[RatingCriteria],
        model: str,
    ) -> RatingResult:
        """Parse LLM rating response into RatingResult."""
        data = self._extract_rating_json(response, criteria)

        stars = data.get("stars", 3)
        criteria_scores = data.get("criteria_scores", {})
        reasoning = data.get("reasoning", "")

        # Map stars to signal/strength
        signal, strength = self._stars_to_signal(stars)

        return RatingResult(
            fact_id=fact_id,
            signal=signal,
            strength=strength,
            stars=stars,
            criteria_scores=criteria_scores,
            reasoning=reasoning,
            model=model,
        )

    def _extract_rating_json(self, response: str, criteria: list[RatingCriteria]) -> dict[str, Any]:
        """Extract and parse JSON from LLM rating response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])  # Remove code fence
            if json_str.startswith("json"):
                json_str = "\n".join(json_str.split("\n")[1:])

            return cast(dict[str, Any], json.loads(json_str))
        except json.JSONDecodeError as e:
            self.agent.lg.warning(
                "failed to parse rating JSON",
                extra={"exception": e, "response": response},
            )
            # Fallback: neutral rating
            return {
                "stars": 3,
                "criteria_scores": {c.name: 0.5 for c in criteria},
                "reasoning": "Failed to parse rating response",
            }

    def _stars_to_signal(
        self, stars: int
    ) -> tuple[Literal["positive", "negative", "dismiss"], float]:
        """Convert star rating to signal and strength."""
        mapping: dict[int, tuple[Literal["positive", "negative", "dismiss"], float]] = {
            1: ("negative", 1.0),
            2: ("negative", 0.5),
            3: ("dismiss", 1.0),
            4: ("positive", 0.5),
            5: ("positive", 1.0),
        }
        return mapping.get(stars, ("dismiss", 1.0))

    def _save_rating(self, result: RatingResult, conductor: RatingConductor) -> None:
        """Save rating to atomic_feedback_details."""
        from .learn import LearnTrait

        learn_trait = self.agent.require_trait(LearnTrait)
        context = self._build_rating_context(result, conductor)

        query = """
        INSERT INTO atomic_feedback_details (fact_id, signal, strength, context)
        VALUES (:fact_id, :signal, :strength, :context)
        """
        params = {
            "fact_id": result.fact_id,
            "signal": result.signal,
            "strength": result.strength,
            "context": json.dumps(context),
        }

        with learn_trait.learn.database.session() as session:
            session.execute(text(query), params)

        self.agent.lg.debug(
            "rating saved",
            extra={
                "fact_id": result.fact_id,
                "stars": result.stars,
                "signal": result.signal,
                "conductor_type": conductor.conductor_type,
                "model": conductor.model,
            },
        )

    def _build_rating_context(
        self, result: RatingResult, conductor: RatingConductor
    ) -> dict[str, Any]:
        """Build context dict for rating storage."""
        return {
            "source": "llm_rater",
            "conductor_type": conductor.conductor_type,
            "model": conductor.model,
            "rating_type": "automated",
            "stars": result.stars,
            "criteria_scores": result.criteria_scores,
            "reasoning": result.reasoning,
            "timestamp": datetime.utcnow().isoformat(),
        }
