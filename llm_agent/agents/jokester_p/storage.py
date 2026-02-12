"""Storage helper for jokester-p agent.

Handles model usage tracking and training metadata recording.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.log import Logger

from .schema import ModelUsage, TrainingMetadata


if TYPE_CHECKING:
    from ...core.traits.builtin.llm import LLMTrait
    from ...core.traits.builtin.storage import StorageTrait


class Storage:
    """Manages storage operations.

    Tracks:
    - Model usage (which model generated each joke, tokens, cost, latency)
    - Training metadata (training iterations, adapter versions)
    """

    def __init__(self, lg: Logger, storage_trait: StorageTrait, llm_trait: LLMTrait) -> None:
        """Initialize storage helper.

        Args:
            lg: Logger instance.
            storage_trait: StorageTrait for database access.
            llm_trait: LLMTrait for model information.
        """
        self._lg = lg
        self._storage = storage_trait.storage
        self._llm_trait = llm_trait

    def record_joke_metadata(self, fact_id: int) -> None:
        """Record model usage and training metadata for a joke.

        Args:
            fact_id: ID of the fact in atomic_facts.
        """
        model_name = self._get_model_name()
        self._record_model_usage(fact_id, model_name)
        self._record_training_metadata(fact_id, model_name)

    def _get_model_name(self) -> str:
        """Get the model name currently being used.

        Returns:
            Model name (e.g., 'claude-sonnet-4-5', 'llama-3.3-70b-instruct').
        """
        model_name = "unknown"
        if hasattr(self._llm_trait, "config") and hasattr(self._llm_trait.config, "default"):
            default_backend = self._llm_trait.config.default
            backends = getattr(self._llm_trait.config, "backends", {})
            if default_backend in backends:
                backend_config = backends[default_backend]
                model_name = getattr(backend_config, "model", "unknown")
        return model_name

    def _record_model_usage(self, fact_id: int, model_name: str) -> None:
        """Record model usage metadata in agent-specific table.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
        """
        try:
            self._storage.insert(
                ModelUsage,
                fact_id=fact_id,
                model_name=model_name,
                model_role="sole",  # Single model for now, extensible for multi-model
                tokens_in=None,  # TODO: Track from LLM response
                tokens_out=None,  # TODO: Track from LLM response
                cost_usd=None,  # TODO: Calculate based on model pricing
                latency_ms=None,  # TODO: Track from LLM response
            )
            self._lg.debug("model usage recorded", extra={"fact_id": fact_id, "model": model_name})
        except Exception as e:
            self._lg.warning(
                "failed to record model usage",
                extra={"exception": e, "fact_id": fact_id},
            )

    def _record_training_metadata(self, fact_id: int, model_name: str) -> None:
        """Record training metadata if using a fine-tuned model.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
        """
        is_finetuned = "jokester" in model_name.lower()

        if is_finetuned:
            self._record_finetuned_model_metadata(fact_id, model_name)
        else:
            self._record_base_model_metadata(fact_id, model_name)

    def _record_base_model_metadata(self, fact_id: int, model_name: str) -> None:
        """Record metadata for base (non-fine-tuned) model."""
        try:
            self._storage.insert(
                TrainingMetadata,
                fact_id=fact_id,
                base_model=model_name,
                adapter_version=None,
                training_iteration=None,
                training_date=None,
                training_data_size=None,
                is_base_model=True,
            )
            self._lg.debug(
                "training metadata recorded (base model)",
                extra={"fact_id": fact_id, "model": model_name},
            )
        except Exception as e:
            self._lg.warning(
                "failed to record training metadata",
                extra={"exception": e, "fact_id": fact_id},
            )

    def _record_finetuned_model_metadata(self, fact_id: int, model_name: str) -> None:
        """Record metadata for fine-tuned model."""
        # TODO: Parse adapter version from model_name or configuration
        # For now, this is a placeholder for future fine-tuned versions
        try:
            self._storage.insert(
                TrainingMetadata,
                fact_id=fact_id,
                base_model="llama-3.3-70b-instruct",  # TODO: Extract from config
                adapter_version=None,  # TODO: Parse from model_name
                training_iteration=None,  # TODO: Get from config
                training_date=None,  # TODO: Get from adapter metadata
                training_data_size=None,  # TODO: Get from training logs
                is_base_model=False,
            )
            self._lg.debug(
                "training metadata recorded (fine-tuned)",
                extra={"fact_id": fact_id, "model": model_name},
            )
        except Exception as e:
            self._lg.warning(
                "failed to record training metadata",
                extra={"exception": e, "fact_id": fact_id},
            )
