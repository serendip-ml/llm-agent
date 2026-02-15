"""Rating trait for automated LLM-based content rating."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra import DotDict

from llm_agent.core.memory.rating import (
    AtomicFactsBackend,
    BatchConfig,
    BatchItem,
    BatchRequest,
    ConfigParser,
    CriteriaConfig,
    PairingConfig,
    ProviderConfig,
    ProviderType,
)
from llm_agent.core.memory.rating import Request as RatingRequest
from llm_agent.core.memory.rating import Result as RatingResult
from llm_agent.core.memory.rating import Service as RatingService

from ..base import BaseTrait


if TYPE_CHECKING:
    from llm_agent.core.agent import Agent

    from .learn import LearnTrait


# Type alias for Rating configuration
RatingConfig = DotDict
"""Rating configuration as DotDict.

Expected fields:
    providers: List of rating provider configurations
        - type: Provider type ("llm" or "manual")
        - backend: LLM backend configuration
        - enabled: Whether this provider is enabled (default: True)
    models: Type-specific criteria and prompts
        atomic: Criteria for atomic facts
            solution: Prompt and criteria for solution facts
    pairing: Preference pairing configuration for DPO training
        - enabled: Enable automatic preference pairing (default: True)
        - high_threshold: Minimum stars for "chosen" content (default: 4)
        - low_threshold: Maximum stars for "rejected" content (default: 2)
        - prompt: Context prompt for preference pairs (can use {category})
    batch_size: Number of items to rate per batch (default: 10)
    auto: Skip confirmation prompts (default: False)
"""


class RatingTrait(BaseTrait):
    """Automated LLM-based content rating trait with DPO preference pairing.

    Provides automated rating capabilities using LLM backends to evaluate
    agent-generated content. Stores ratings in atomic_feedback_details with
    source tracking for multi-rater scenarios.

    **Preference Pairing:** When a rating qualifies (4+ stars or 2- stars),
    the trait automatically looks for an unpaired opposite in the same category
    and creates a preference pair for DPO (Direct Preference Optimization) training.

    **IMPORTANT:** RatingTrait depends on LearnTrait. LearnTrait must be attached
    to the agent BEFORE RatingTrait.

    Capabilities:
        - rate_unrated(): Find and rate unrated content automatically
        - rate_fact(): Rate a specific fact by ID
        - rate_fact_with_all_providers(): Rate with all providers (for inline rating)
        - get_unrated_count(): Count unrated facts
        - Multi-provider support: Use multiple LLMs for rating comparison
        - Automatic preference pairing for DPO training

    Example:
        from llm_agent.core.traits import RatingTrait, LearnTrait

        # Configure rating with pairing
        rating_config = RatingConfig(
            providers=[{
                "type": "llm",
                "backend": {"model": "claude-sonnet-4.5", ...},
                "enabled": True
            }],
            models={
                "atomic": {
                    "solution": {
                        "prompt": "Rate this joke...",
                        "criteria": [
                            {"name": "humor", "description": "Is it funny?"},
                            {"name": "originality", "description": "Is it original?"}
                        ]
                    }
                }
            },
            pairing={
                "enabled": True,
                "high_threshold": 4,
                "low_threshold": 2,
                "prompt": "Tell me a {category}."
            },
            batch_size=10,
            auto=True
        )

        # Attach traits
        agent.add_trait(LearnTrait(agent, learn_config))
        agent.add_trait(RatingTrait(agent, rating_config))
        agent.start()

        # Rate content with automatic preference pairing
        results = agent.get_trait(RatingTrait).rate_fact_with_all_providers(
            fact_id=123, content="Why did the...", category="joke"
        )

    Lifecycle:
        - on_start(): Initialize rating providers and validate configuration
        - on_stop(): Cleanup resources
    """

    def __init__(
        self,
        agent: Agent,
        config: RatingConfig | None = None,
    ) -> None:
        """Initialize rating trait.

        Args:
            agent: The agent this trait belongs to.
            config: Rating configuration (None = use defaults).
        """
        super().__init__(agent)
        self.config = config or RatingConfig()
        self._providers: list[ProviderConfig] = []
        self._criteria: dict[str, CriteriaConfig] = {}  # fact_type -> CriteriaConfig
        self._service: RatingService | None = None
        self._backend: AtomicFactsBackend | None = None
        self._pairing: PairingConfig | None = None
        self._batch: BatchConfig | None = None

    def on_start(self) -> None:
        """Initialize rating service and backend.

        Raises:
            TraitNotFoundError: If LearnTrait or LLMTrait are not attached.
        """
        from llm_agent.core.llm import LLMCaller

        from .learn import LearnTrait
        from .llm import LLMTrait

        # Require dependencies
        learn_trait = self.agent.require_trait(LearnTrait)
        llm_trait = self.agent.require_trait(LLMTrait)

        # Initialize rating service and backend with LLMCaller wrapper
        caller = LLMCaller(self.agent.lg, llm_trait.router)
        self._service = RatingService(self.agent.lg, caller)
        self._backend = AtomicFactsBackend(self.agent.lg, learn_trait.learn.database)

        self._parse_config()
        self._log_started()

    def _parse_config(self) -> None:
        """Parse provider, criteria, pairing, and batch configurations."""
        parser = ConfigParser(self.agent.lg)
        self._providers = parser.parse_providers(self.config.get("providers", []))
        self._criteria = parser.parse_criteria(self.config.get("models", {}))
        self._pairing = parser.parse_pairing(self.config.get("pairing"))
        self._batch = parser.parse_batch(self.config.get("batch_size"))

    def _log_started(self) -> None:
        """Log trait started with config summary."""
        self.agent.lg.debug(
            "rating trait started",
            extra={
                "providers": len(self._providers),
                "enabled": sum(1 for p in self._providers if p.enabled),
                "types": list(self._criteria.keys()),
                "auto": self.config.get("auto", False),
                "pairing_enabled": self._pairing.enabled if self._pairing else False,
            },
        )

    def on_stop(self) -> None:
        """Clean up rating resources."""
        self._providers = []
        self._criteria = {}
        self._service = None
        self._backend = None
        self._pairing = None
        self._batch = None
        self.agent.lg.debug("rating trait stopped")

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

        if not self._backend:
            raise RuntimeError("Rating backend not initialized - call on_start() first")

        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = str(learn_trait.learn.context.context_key)

        return self._backend.get_unrated_count(context_key, fact_type, category)

    def rate_unrated(
        self,
        limit: int = 10,
        fact_type: str | None = None,
        category: str | None = None,
        provider_index: int = 0,
    ) -> list[RatingResult]:
        """Rate unrated facts automatically using configured providers.

        Args:
            limit: Maximum number of facts to rate.
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            provider_index: Index of provider to use (default: 0 = first enabled).

        Returns:
            List of rating results.
        """
        from .learn import LearnTrait

        if not self._backend:
            raise RuntimeError("Rating backend not initialized - call on_start() first")

        provider = self._select_provider(provider_index)
        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = str(learn_trait.learn.context.context_key)

        results = []
        for fact in self._backend.unrated_facts(context_key, fact_type, category, limit):
            result = self._try_rate_fact(fact, provider)
            if result:
                results.append(result)

        return results

    def rate_batch(
        self,
        limit: int | None = None,
        batch_size: int | None = None,
        fact_type: str | None = None,
        category: str | None = None,
        provider_index: int = 0,
    ) -> tuple[int, int]:
        """Rate unrated facts in batches for efficiency.

        Batches multiple facts per LLM call to reduce API costs.

        Args:
            limit: Maximum facts to rate (None = all unrated).
            batch_size: Items per batch (None = use config default).
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            provider_index: Index of provider to use (default: 0).

        Returns:
            Tuple of (rated_count, failed_count).
        """
        from .learn import LearnTrait

        if not self._backend or not self._service:
            raise RuntimeError("Rating not initialized - call on_start() first")

        provider = self._select_provider(provider_index)
        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = str(learn_trait.learn.context.context_key)

        # Determine batch size
        size = batch_size or (self._batch.size if self._batch else 5)

        # Get type-specific criteria for prompt template
        resolved_type = fact_type or "solution"
        type_criteria = self._criteria.get(resolved_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {resolved_type}")

        # Build provider identifier
        backend_type = provider.backend.get("type", "unknown")
        provider_id = f"llm_{provider.model}_{backend_type}"

        # Collect unrated facts
        facts = list(self._backend.unrated_facts(context_key, fact_type, category, limit or 10000))
        if not facts:
            return (0, 0)

        return self._rate_facts_in_batches(
            facts, size, type_criteria.prompt, provider.model, provider_id
        )

    def _rate_facts_in_batches(
        self,
        facts: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str,
        model: str,
        provider_id: str,
    ) -> tuple[int, int]:
        """Rate facts in batches and save results."""
        assert self._service is not None
        total_rated, total_failed = 0, 0

        for i in range(0, len(facts), batch_size):
            batch_facts = facts[i : i + batch_size]
            rated, failed = self._process_batch(batch_facts, prompt_template, model, provider_id, i)
            total_rated += rated
            total_failed += failed

        return (total_rated, total_failed)

    def _process_batch(
        self,
        batch_facts: list[dict[str, Any]],
        prompt_template: str,
        model: str,
        provider_id: str,
        batch_start: int,
    ) -> tuple[int, int]:
        """Process a single batch of facts. Returns (rated, failed) counts."""
        assert self._service is not None
        items = [BatchItem(fact=f["id"], content=f["content"]) for f in batch_facts]
        request = BatchRequest(
            items=items, prompt_template=prompt_template, model=model, provider=provider_id
        )

        try:
            results = self._service.rate_batch(request)
            for result in results:
                self._save_rating(result)
            return (len(results), len(items) - len(results))
        except Exception as e:
            self.agent.lg.warning(
                "batch rating failed",
                extra={"exception": e, "batch_start": batch_start, "batch_size": len(items)},
            )
            return (0, len(items))

    def _try_rate_fact(self, fact: dict[str, Any], provider: ProviderConfig) -> RatingResult | None:
        """Attempt to rate a single fact, save, and try pairing."""
        try:
            result = self.rate_fact(fact["id"], fact["content"], fact["type"], provider)
            self._save_rating(result)

            # Try to create preference pair for DPO training
            self._try_create_preference_pair(
                fact["id"], fact["content"], fact.get("category"), result.stars
            )

            return result
        except Exception as e:
            self.agent.lg.warning(
                "rating failed",
                extra={
                    "exception": e,
                    "fact_id": fact["id"],
                    "fact_type": fact["type"],
                    "provider_type": provider.provider_type,
                },
            )
            return None

    def rate_fact(
        self,
        fact_id: int,
        content: str,
        fact_type: str,
        provider: ProviderConfig,
    ) -> RatingResult:
        """Rate a specific fact using an LLM provider.

        Args:
            fact_id: ID of the fact to rate.
            content: Content to rate.
            fact_type: Type of fact (e.g., "solution", "prediction").
            provider: Rating provider to use.

        Returns:
            Rating result with scores and reasoning.
        """
        if not self._service:
            raise RuntimeError("Rating service not initialized - call on_start() first")

        # Get type-specific criteria
        type_criteria = self._criteria.get(fact_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {fact_type}")

        # Use rating service to rate content
        # Build unique provider identifier (handles model="auto" case)
        backend_type = provider.backend.get("type", "unknown")
        provider_id = f"llm_{provider.model}_{backend_type}"
        request = RatingRequest(
            fact=fact_id,
            content=content,
            prompt_template=type_criteria.prompt,
            criteria=type_criteria.criteria,
            model=provider.model,
            provider=provider_id,
            temperature=0.3,
        )
        return self._service.rate_content(request)

    def rate_fact_with_all_providers(
        self,
        fact_id: int,
        content: str,
        fact_type: str = "solution",
        category: str | None = None,
    ) -> list[RatingResult]:
        """Rate a fact with all enabled providers and save ratings.

        This is the primary method for agents to use for inline rating.
        Rates the content with each enabled LLM provider and saves all ratings.
        If pairing is enabled and the rating qualifies (4+ or 2- stars),
        attempts to create a preference pair for DPO training.

        Args:
            fact_id: ID of the fact to rate.
            content: Content to rate (e.g., joke text).
            fact_type: Type of fact (default: "solution").
            category: Category for preference pairing (e.g., "joke").

        Returns:
            List of rating results from each provider.
        """
        from .llm import LLMTrait

        # Verify LLMTrait is available
        self.agent.require_trait(LLMTrait)

        results = []
        enabled_providers = [
            p for p in self._providers if p.enabled and p.provider_type == ProviderType.LLM
        ]

        for provider in enabled_providers:
            result = self._rate_and_save_with_provider(
                fact_id, content, fact_type, category, provider
            )
            if result:
                results.append(result)

        return results

    def rate_items(
        self,
        items: list[tuple[int, str]],
        fact_type: str = "solution",
        category: str | None = None,
    ) -> int:
        """Rate specific items in a batch.

        Primary method for batch auto-rating. Takes a list of (fact_id, content)
        tuples and rates them together in a single LLM call for efficiency.

        Args:
            items: List of (fact_id, content) tuples to rate.
            fact_type: Type of facts (default: "solution").
            category: Category for preference pairing (e.g., "joke").

        Returns:
            Number of items successfully rated.
        """
        if not items:
            return 0

        if not self._service:
            raise RuntimeError("Rating service not initialized - call on_start() first")

        provider = self._select_provider(0)
        type_criteria = self._criteria.get(fact_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {fact_type}")

        backend_type = provider.backend.get("type", "unknown")
        provider_id = f"llm_{provider.model}_{backend_type}"

        # Convert tuples to dict format expected by _process_batch
        batch_facts = [{"id": fact_id, "content": content} for fact_id, content in items]

        rated, _ = self._process_batch(
            batch_facts, type_criteria.prompt, provider.model, provider_id, 0
        )

        # Attempt preference pairing for each rated item
        if category and self._pairing and self._pairing.enabled:
            self._try_pair_batch_items(items, category)

        return rated

    def _try_pair_batch_items(
        self,
        items: list[tuple[int, str]],
        category: str,
    ) -> None:
        """Attempt preference pairing for batch-rated items.

        After batch rating, check each item's rating and try to create
        preference pairs for qualifying ratings (4+ or 2- stars).
        """
        if not self._backend:
            return

        for fact_id, content in items:
            rating = self._backend.get_fact_rating(fact_id)
            if rating:
                self._try_create_preference_pair(fact_id, content, category, rating)

    def _rate_and_save_with_provider(
        self,
        fact_id: int,
        content: str,
        fact_type: str,
        category: str | None,
        provider: ProviderConfig,
    ) -> RatingResult | None:
        """Rate a fact with a single provider, save, and attempt pairing."""
        try:
            result = self.rate_fact(fact_id, content, fact_type, provider)
            self._save_rating(result)
            self.agent.lg.debug(
                "rated fact",
                extra={"fact_id": fact_id, "model": provider.model, "stars": result.stars},
            )
            self._try_create_preference_pair(fact_id, content, category, result.stars)
            return result
        except Exception as e:
            self.agent.lg.warning(
                "rating failed",
                extra={"exception": e, "fact_id": fact_id, "provider": provider.provider_type},
            )
            return None

    def _select_provider(self, index: int = 0) -> ProviderConfig:
        """Select rating provider by index.

        Args:
            index: Index of enabled provider to use (default: 0).

        Returns:
            Selected provider.

        Raises:
            RuntimeError: If no enabled providers configured.
            IndexError: If index out of range.
        """
        enabled_providers = [p for p in self._providers if p.enabled]

        if not enabled_providers:
            raise RuntimeError("No enabled rating providers configured")

        if index >= len(enabled_providers):
            raise IndexError(
                f"Provider index {index} out of range (have {len(enabled_providers)} enabled)"
            )

        return enabled_providers[index]

    def _save_rating(self, result: RatingResult) -> None:
        """Save rating using backend."""
        if not self._backend:
            raise RuntimeError("Rating backend not initialized - call on_start() first")

        self._backend.save_rating(result=result, source="llm_rater")

    def get_fact_rating(self, fact_id: int) -> int | None:
        """Get the star rating for a specific fact.

        Args:
            fact_id: ID of the fact to look up.

        Returns:
            Star rating (1-5) if rated, None if not rated.
        """
        if not self._backend:
            return None
        return self._backend.get_fact_rating(fact_id)

    # =========================================================================
    # Preference pairing for DPO training
    # =========================================================================

    def _try_create_preference_pair(
        self,
        fact_id: int,
        content: str,
        category: str | None,
        stars: int,
    ) -> int | None:
        """Try to create a preference pair if this rating qualifies.

        Called after saving a rating. If the rating is 4+ stars (high) or 2- stars (low),
        looks for an unpaired opposite in the same category and creates a preference pair.

        Args:
            fact_id: ID of the just-rated fact.
            content: Content of the fact.
            category: Category of the fact (e.g., "joke").
            stars: Star rating (1-5).

        Returns:
            Preference ID if a pair was created, None otherwise.
        """
        if not self._should_attempt_pairing(category, stars):
            return None

        return self._find_and_create_pair(fact_id, content, category, stars)

    def _should_attempt_pairing(self, category: str | None, stars: int) -> bool:
        """Check if pairing should be attempted for this rating."""
        if not self._pairing or not self._pairing.enabled:
            return False

        if not category:
            self.agent.lg.debug("skipping pairing - no category")
            return False

        if not self._backend:
            return False

        is_high = stars >= self._pairing.high_threshold
        is_low = stars <= self._pairing.low_threshold

        if not is_high and not is_low:
            self.agent.lg.debug(
                "skipping pairing - stars in neutral range",
                extra={
                    "stars": stars,
                    "high": self._pairing.high_threshold,
                    "low": self._pairing.low_threshold,
                },
            )
            return False

        return True

    def _find_and_create_pair(
        self, fact_id: int, content: str, category: str | None, stars: int
    ) -> int | None:
        """Find an opposite-rated fact and create preference pair."""
        from .learn import LearnTrait

        assert self._pairing is not None
        assert self._backend is not None
        assert category is not None

        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = str(learn_trait.learn.context.context_key)
        is_high = stars >= self._pairing.high_threshold

        opposite = self._find_opposite_fact(context_key, category, fact_id, is_high)
        if not opposite:
            self.agent.lg.debug(
                "no unpaired opposite found for pairing",
                extra={"fact_id": fact_id, "category": category, "is_high": is_high},
            )
            return None

        return self._create_preference_pair(
            fact_id, content, stars, opposite, category, is_high, learn_trait
        )

    def _find_opposite_fact(
        self, context_key: str, category: str, exclude_fact_id: int, is_high: bool
    ) -> dict[str, Any] | None:
        """Find an unpaired fact with opposite rating (high->low or low->high)."""
        assert self._pairing is not None
        assert self._backend is not None

        if is_high:
            min_stars, max_stars = 1, self._pairing.low_threshold
        else:
            min_stars, max_stars = self._pairing.high_threshold, 5

        opposites = self._backend.get_rated_unpaired_facts(
            context_key=context_key,
            category=category,
            min_stars=min_stars,
            max_stars=max_stars,
            exclude_fact_id=exclude_fact_id,
            limit=1,
        )
        return opposites[0] if opposites else None

    def _create_preference_pair(
        self,
        fact_id: int,
        content: str,
        stars: int,
        opposite: dict[str, Any],
        category: str,
        is_high: bool,
        learn_trait: LearnTrait,
    ) -> int:
        """Create preference pair and mark facts as paired."""
        assert self._backend is not None
        assert self._pairing is not None

        chosen = (
            (content, fact_id, stars)
            if is_high
            else (opposite["content"], opposite["id"], opposite["stars"])
        )
        rejected = (
            (opposite["content"], opposite["id"], opposite["stars"])
            if is_high
            else (content, fact_id, stars)
        )

        preference_id = learn_trait.record_preference(
            context=self._pairing.prompt.format(category=category),
            chosen=chosen[0],
            rejected=rejected[0],
        )

        self._backend.mark_facts_paired(fact_id, opposite["id"], preference_id)
        self._log_preference_pair(preference_id, chosen, rejected, category)
        return preference_id

    def _log_preference_pair(
        self,
        preference_id: int,
        chosen: tuple[str, int, int],
        rejected: tuple[str, int, int],
        category: str,
    ) -> None:
        """Log creation of preference pair."""
        self.agent.lg.info(
            "created preference pair for DPO training",
            extra={
                "preference_id": preference_id,
                "chosen_fact_id": chosen[1],
                "chosen_stars": chosen[2],
                "rejected_fact_id": rejected[1],
                "rejected_stars": rejected[2],
                "category": category,
            },
        )
