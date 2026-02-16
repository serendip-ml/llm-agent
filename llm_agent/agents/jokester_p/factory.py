"""Factory for creating jokester agent instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.log import Logger
from sqlalchemy import text

from ...core.agent import Factory as BaseFactory
from ...core.traits import TraitName as TN
from ...core.traits.builtin.directive import DirectiveTrait
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.llm import LLMTrait
from ...core.traits.builtin.rating import RatingTrait
from ...core.traits.builtin.storage import StorageTrait
from .agent import JokesterAgent
from .cli import JokesterCLI
from .generate import ExpectedAdapter, JokeGenerator
from .novelty import NoveltyChecker
from .rating import BatchRater
from .schema import ModelUsage, TrainingMetadata
from .storage import Storage


if TYPE_CHECKING:
    pass


@dataclass
class Components:
    """Wired components for jokester agent."""

    generator: JokeGenerator
    storage: Storage
    rater: BatchRater | None


class Factory(BaseFactory):
    """Factory for jokester agent.

    Handles all component wiring: NoveltyChecker, JokeGenerator, Storage.
    Agent just uses the pre-configured components.
    """

    agent_class = JokesterAgent
    required_traits = [TN.DIRECTIVE, TN.LLM, TN.LEARN, TN.STORAGE, TN.RATING]
    default_tools = {}
    cli_tool = JokesterCLI

    @classmethod
    def create_components(cls, agent: JokesterAgent) -> Components:
        """Create and wire all components for the agent.

        Called by agent.start() after traits are started.

        Args:
            agent: The agent instance with started traits.

        Returns:
            Components dataclass with generator and storage.

        Raises:
            RuntimeError: If embedder not configured.
        """
        lg = agent._lg
        config = agent.config

        storage_trait = agent.require_trait(StorageTrait)
        storage_trait.storage.register_table(ModelUsage)
        storage_trait.storage.register_table(TrainingMetadata)

        learn_trait = agent.require_trait(LearnTrait)
        cls._validate_embedder(lg, learn_trait)

        generator = cls._create_generator(agent, lg, config, learn_trait)
        storage = Storage(lg, storage_trait, learn_trait, agent.name)
        rater = cls._create_rater(agent, lg, config)

        return Components(generator=generator, storage=storage, rater=rater)

    @classmethod
    def _validate_embedder(cls, lg: Logger, learn_trait: LearnTrait) -> None:
        """Validate embedder is configured for novelty checking."""
        if not learn_trait.has_embedder:
            lg.error("embedder not configured - required for novelty checking")
            raise RuntimeError(
                "Jokester agent requires embedder for guaranteed novelty checking. "
                "Configure embedder_url in learn section."
            )

    @classmethod
    def _create_generator(
        cls, agent: JokesterAgent, lg: Logger, config: DotDict, learn_trait: LearnTrait
    ) -> JokeGenerator:
        """Create JokeGenerator with novelty checker."""
        llm_trait = agent.require_trait(LLMTrait)
        directive_trait = agent.get_trait(DirectiveTrait)
        novelty_checker = NoveltyChecker(lg, learn_trait, config.get("similarity_threshold", 0.85))
        expected_adapter = cls._get_expected_adapter(lg, learn_trait, agent.name)

        return JokeGenerator(
            lg=lg,
            llm_trait=llm_trait,
            novelty_checker=novelty_checker,
            directive_trait=directive_trait,
            max_retries=config.get("max_retries", 3),
            denylist=config.get("denylist", []),
            expected_adapter=expected_adapter,
        )

    @classmethod
    def _get_expected_adapter(
        cls, lg: Logger, learn_trait: LearnTrait, context_key: str
    ) -> ExpectedAdapter | None:
        """Query latest completed DPO run for expected adapter info."""
        try:
            sql = text("""
                SELECT adapter_name, metrics->'adapter'->>'md5' as md5,
                       metrics->'adapter'->>'mtime' as mtime
                FROM dpo_runs
                WHERE context_key = :context_key AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
            """)
            with learn_trait.learn.database.session() as session:
                row = session.execute(sql, {"context_key": context_key}).fetchone()

            if row is None:
                lg.debug("no completed DPO runs found", extra={"context_key": context_key})
                return None

            name, md5, mtime = row
            if not md5 or not mtime:
                lg.debug("DPO run missing adapter md5/mtime", extra={"adapter": name})
                return None

            lg.debug(
                "loaded expected adapter from DPO run",
                extra={"adapter": name, "md5": md5, "mtime": mtime},
            )
            return ExpectedAdapter(name=name, md5=md5, mtime=mtime)

        except Exception as e:
            lg.warning("failed to query expected adapter", extra={"exception": e})
            return None

    @classmethod
    def _create_rater(cls, agent: JokesterAgent, lg: Logger, config: DotDict) -> BatchRater | None:
        """Create BatchRater if auto-rating is enabled."""
        rating_config = config.get("rating", {})
        if not rating_config.get("auto", False):
            return None
        rating_trait = agent.get_trait(RatingTrait)
        if not rating_trait:
            return None
        batch_size = rating_config.get("batch_size", 5)
        return BatchRater(lg, rating_trait, batch_size)
