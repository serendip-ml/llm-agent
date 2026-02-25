"""Storage helper for jokester-p agent.

Handles joke persistence, model usage tracking, and training metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from appinfra.log import Logger
from llm_infer.client.types import AdapterInfo

from .schema import ModelUsage, TrainingMetadata


if TYPE_CHECKING:
    from ...core.traits.builtin.learn import LearnTrait
    from ...core.traits.builtin.storage import StorageTrait
    from .generate import Joke


class Storage:
    """Manages joke persistence and metadata.

    Handles:
    - Saving jokes to atomic_facts via LearnTrait
    - Model usage tracking (which model generated each joke)
    - Training metadata (adapter versions, fallback tracking)
    """

    def __init__(
        self, lg: Logger, storage_trait: StorageTrait, learn_trait: LearnTrait, agent_name: str
    ) -> None:
        """Initialize storage helper.

        Args:
            lg: Logger instance.
            storage_trait: StorageTrait for database access.
            learn_trait: LearnTrait for saving jokes.
            agent_name: Name of the agent.
        """
        self._lg = lg
        self._storage = storage_trait.storage
        self._learn = learn_trait
        self._agent_name = agent_name

    def save_joke(
        self,
        joke: Joke,
        model_name: str,
        attempts: int,
        adapter: AdapterInfo | None = None,
    ) -> int:
        """Save joke and record metadata.

        Args:
            joke: Generated joke.
            model_name: Actual model used.
            attempts: Number of generation attempts.
            adapter: Full adapter info from LLM response.

        Returns:
            The fact_id of the saved joke.
        """
        fact_id = self._learn.kelt.atomic.solutions.record(
            agent_name=self._agent_name,
            problem="Tell one short, original joke",
            problem_context={"style_preference": "varied"},
            answer={"text": joke.text, "style": joke.style},
            answer_text=joke.text,
            tokens_used=0,
            latency_ms=0,
            category="joke",
            source="agent",
        )
        self._lg.debug("joke saved", extra={"fact_id": fact_id, "style": joke.style})

        self.record_joke_metadata(fact_id, model_name, attempts, adapter)
        return fact_id

    def record_joke_metadata(
        self,
        fact_id: int,
        model_name: str,
        attempts: int,
        adapter: AdapterInfo | None = None,
    ) -> None:
        """Record model usage and training metadata for a joke.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Actual model used (from LLM response).
            attempts: Number of generation attempts needed.
            adapter: Full adapter info from LLM response.

        Raises:
            Exception: If metadata recording fails critically (re-raises after logging).
        """
        usage_failed = self._try_record_model_usage(fact_id, model_name, attempts)
        training_failed = self._try_record_training_metadata(fact_id, model_name, adapter)
        self._log_metadata_failures(fact_id, usage_failed, training_failed)

    def _try_record_model_usage(self, fact_id: int, model_name: str, attempts: int) -> bool:
        """Try to record model usage, return True if failed."""
        try:
            self._record_model_usage(fact_id, model_name, attempts)
            return False
        except Exception as e:
            self._lg.warning(
                "model usage recording failed",
                extra={
                    "exception": e,
                    "fact_id": fact_id,
                    "model": model_name,
                    "attempts": attempts,
                },
            )
            return True

    def _try_record_training_metadata(
        self,
        fact_id: int,
        model_name: str,
        adapter: AdapterInfo | None,
    ) -> bool:
        """Try to record training metadata, return True if failed."""
        try:
            self._record_training_metadata(fact_id, model_name, adapter)
            return False
        except Exception as e:
            self._lg.warning(
                "training metadata recording failed",
                extra={
                    "exception": e,
                    "fact_id": fact_id,
                    "model": model_name,
                    "adapter": adapter,
                },
            )
            return True

    def _log_metadata_failures(
        self, fact_id: int, usage_failed: bool, training_failed: bool
    ) -> None:
        """Log summary if any metadata recording failed."""
        if usage_failed or training_failed:
            self._lg.warning(
                "metadata recording incomplete",
                extra={
                    "fact_id": fact_id,
                    "usage_failed": usage_failed,
                    "training_failed": training_failed,
                },
            )

    def _record_model_usage(self, fact_id: int, model_name: str, attempts: int) -> None:
        """Record model usage metadata in agent-specific table.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
            attempts: Number of generation attempts needed.

        Raises:
            Exception: If database insert fails.
        """
        self._storage.insert(
            ModelUsage,
            fact_id=fact_id,
            model_name=model_name,
            model_role="sole",  # Single model for now, extensible for multi-model
            attempts=attempts,
            tokens_in=None,  # TODO: Track from LLM response
            tokens_out=None,  # TODO: Track from LLM response
            cost_usd=None,  # TODO: Calculate based on model pricing
            latency_ms=None,  # TODO: Track from LLM response
        )
        self._lg.debug(
            "model usage recorded",
            extra={"fact_id": fact_id, "model": model_name, "attempts": attempts},
        )

    def _record_training_metadata(
        self,
        fact_id: int,
        model_name: str,
        adapter: AdapterInfo | None,
    ) -> None:
        """Record training metadata.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
            adapter: Full adapter info from LLM response.
        """
        if adapter and adapter.actual and not adapter.fallback:
            self._record_finetuned_model_metadata(fact_id, model_name, adapter)
        else:
            self._record_base_model_metadata(fact_id, model_name, adapter)

    def _record_base_model_metadata(
        self,
        fact_id: int,
        model_name: str,
        adapter: AdapterInfo | None,
    ) -> None:
        """Record metadata for base (non-fine-tuned) model."""
        self._storage.insert(
            TrainingMetadata,
            fact_id=fact_id,
            base_model=model_name,
            adapter_version=None,
            training_iteration=None,
            training_date=None,
            training_data_size=None,
            is_base_model=True,
            # Legacy columns - no longer written, kept for historical data
            adapter_requested=None,
            adapter_fallback=False,
            # New: full adapter info as JSON
            adapter_info=asdict(adapter) if adapter else None,
        )
        self._lg.debug(
            "training metadata recorded (base model)",
            extra={"fact_id": fact_id, "model": model_name, "adapter": adapter},
        )

    def _record_finetuned_model_metadata(
        self, fact_id: int, model_name: str, adapter: AdapterInfo
    ) -> None:
        """Record metadata for fine-tuned model."""
        self._storage.insert(
            TrainingMetadata,
            fact_id=fact_id,
            base_model=model_name,
            adapter_version=adapter.actual,
            training_iteration=None,
            training_date=None,
            training_data_size=None,
            is_base_model=False,
            # Legacy columns - no longer written, kept for historical data
            adapter_requested=None,
            adapter_fallback=False,
            # New: full adapter info as JSON
            adapter_info=asdict(adapter),
        )
        self._lg.debug(
            "training metadata recorded (fine-tuned)",
            extra={"fact_id": fact_id, "model": model_name, "adapter": adapter},
        )
