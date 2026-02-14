"""Storage helper for jokester-p agent.

Handles joke persistence, model usage tracking, and training metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.log import Logger

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
        adapter_id: str | None = None,
        adapter_fallback: bool = False,
    ) -> int:
        """Save joke and record metadata.

        Args:
            joke: Generated joke.
            model_name: Actual model used.
            attempts: Number of generation attempts.
            adapter_id: LoRA adapter requested (if any).
            adapter_fallback: True if adapter was requested but not available.

        Returns:
            The fact_id of the saved joke.
        """
        fact_id = self._learn.learn.solutions.record(
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

        self.record_joke_metadata(fact_id, model_name, attempts, adapter_id, adapter_fallback)
        return fact_id

    def record_joke_metadata(
        self,
        fact_id: int,
        model_name: str,
        attempts: int,
        adapter_id: str | None = None,
        adapter_fallback: bool = False,
    ) -> None:
        """Record model usage and training metadata for a joke.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Actual model used (from LLM response).
            attempts: Number of generation attempts needed.
            adapter_id: LoRA adapter requested (if any).
            adapter_fallback: True if adapter was requested but not available.

        Raises:
            Exception: If metadata recording fails critically (re-raises after logging).
        """
        # If adapter was requested but fell back, record as base model
        actual_adapter = None if adapter_fallback else adapter_id

        usage_failed = self._try_record_model_usage(fact_id, model_name, attempts)
        training_failed = self._try_record_training_metadata(
            fact_id, model_name, actual_adapter, adapter_id, adapter_fallback
        )
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
        actual_adapter: str | None,
        requested_adapter: str | None,
        adapter_fallback: bool,
    ) -> bool:
        """Try to record training metadata, return True if failed."""
        try:
            self._record_training_metadata(
                fact_id, model_name, actual_adapter, requested_adapter, adapter_fallback
            )
            return False
        except Exception as e:
            self._lg.warning(
                "training metadata recording failed",
                extra={
                    "exception": e,
                    "fact_id": fact_id,
                    "model": model_name,
                    "actual_adapter": actual_adapter,
                    "requested_adapter": requested_adapter,
                    "adapter_fallback": adapter_fallback,
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
        actual_adapter: str | None,
        requested_adapter: str | None,
        adapter_fallback: bool,
    ) -> None:
        """Record training metadata.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
            actual_adapter: LoRA adapter actually used (None if base model or fallback).
            requested_adapter: LoRA adapter that was requested.
            adapter_fallback: True if adapter was requested but not available.
        """
        if actual_adapter:
            self._record_finetuned_model_metadata(fact_id, model_name, actual_adapter)
        else:
            self._record_base_model_metadata(
                fact_id, model_name, requested_adapter, adapter_fallback
            )

    def _record_base_model_metadata(
        self,
        fact_id: int,
        model_name: str,
        requested_adapter: str | None = None,
        adapter_fallback: bool = False,
    ) -> None:
        """Record metadata for base (non-fine-tuned) model.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
            requested_adapter: Adapter that was requested but not used.
            adapter_fallback: True if this is a fallback (adapter requested but unavailable).

        Raises:
            Exception: If database insert fails.
        """
        self._storage.insert(
            TrainingMetadata,
            fact_id=fact_id,
            base_model=model_name,
            adapter_version=None,
            training_iteration=None,
            training_date=None,
            training_data_size=None,
            is_base_model=True,
            adapter_requested=requested_adapter,
            adapter_fallback=adapter_fallback,
        )
        log_extra = {"fact_id": fact_id, "model": model_name}
        if adapter_fallback:
            log_extra["adapter_requested"] = requested_adapter
            log_extra["fallback"] = True
        self._lg.debug("training metadata recorded (base model)", extra=log_extra)

    def _record_finetuned_model_metadata(
        self, fact_id: int, model_name: str, adapter_id: str
    ) -> None:
        """Record metadata for fine-tuned model.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the base model used.
            adapter_id: LoRA adapter ID.

        Raises:
            Exception: If database insert fails.
        """
        self._storage.insert(
            TrainingMetadata,
            fact_id=fact_id,
            base_model=model_name,
            adapter_version=adapter_id,
            training_iteration=None,  # TODO: Get from config
            training_date=None,  # TODO: Get from adapter metadata
            training_data_size=None,  # TODO: Get from training logs
            is_base_model=False,
        )
        self._lg.debug(
            "training metadata recorded (fine-tuned)",
            extra={
                "fact_id": fact_id,
                "base_model": model_name,
                "adapter_id": adapter_id,
            },
        )
