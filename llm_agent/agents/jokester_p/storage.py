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

        Raises:
            Exception: If metadata recording fails critically (re-raises after logging).
        """
        model_name = self._get_model_name()
        usage_failed = self._try_record_model_usage(fact_id, model_name)
        training_failed = self._try_record_training_metadata(fact_id, model_name)
        self._log_metadata_failures(fact_id, usage_failed, training_failed)

    def _try_record_model_usage(self, fact_id: int, model_name: str) -> bool:
        """Try to record model usage, return True if failed."""
        try:
            self._record_model_usage(fact_id, model_name)
            return False
        except Exception as e:
            self._lg.warning(
                "model usage recording failed", extra={"exception": e, "fact_id": fact_id}
            )
            return True

    def _try_record_training_metadata(self, fact_id: int, model_name: str) -> bool:
        """Try to record training metadata, return True if failed."""
        try:
            self._record_training_metadata(fact_id, model_name)
            return False
        except Exception as e:
            self._lg.warning(
                "training metadata recording failed", extra={"exception": e, "fact_id": fact_id}
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

    def _get_model_name(self) -> str:
        """Get the model name currently being used.

        Returns:
            Model name (e.g., 'claude-sonnet-4-5', 'llama-3.3-70b-instruct').
        """
        model_name = "unknown"

        if hasattr(self._llm_trait, "config"):
            config = self._llm_trait.config

            # Try multi-backend format (config.backends[config.default].model)
            if hasattr(config, "default") and hasattr(config, "backends"):
                default_backend = config.default
                backends = config.backends
                if default_backend in backends:
                    backend_config = backends[default_backend]
                    model_name = getattr(backend_config, "model", "unknown")
            # Try single-backend format (config.model)
            elif hasattr(config, "model"):
                model_name = config.model

        if model_name == "unknown":
            self._lg.warning("unable to extract model name from LLM config")

        return model_name

    def _record_model_usage(self, fact_id: int, model_name: str) -> None:
        """Record model usage metadata in agent-specific table.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.

        Raises:
            Exception: If database insert fails.
        """
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

    def _record_training_metadata(self, fact_id: int, model_name: str) -> None:
        """Record training metadata if using a fine-tuned model.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
        """
        # Detect fine-tuned models by common adapter naming patterns
        # Fine-tuned models typically have format: base-model-name-adapter-suffix
        # e.g., "llama-70b-jokester-v4" or "llama-3.3-70b-instruct-joke-teller"
        is_finetuned = any(
            keyword in model_name.lower()
            for keyword in ["-jokester", "-joke-teller", "-adapter", "-lora", "-qlora"]
        )

        if is_finetuned:
            self._record_finetuned_model_metadata(fact_id, model_name)
        else:
            self._record_base_model_metadata(fact_id, model_name)

    def _record_base_model_metadata(self, fact_id: int, model_name: str) -> None:
        """Record metadata for base (non-fine-tuned) model.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.

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
        )
        self._lg.debug(
            "training metadata recorded (base model)",
            extra={"fact_id": fact_id, "model": model_name},
        )

    def _record_finetuned_model_metadata(self, fact_id: int, model_name: str) -> None:
        """Record metadata for fine-tuned model.

        Extracts base model name by removing adapter suffixes from model_name.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.

        Raises:
            Exception: If database insert fails.
        """
        base_model = self._extract_base_model(model_name)
        adapter_version = self._extract_adapter_version(model_name)

        self._storage.insert(
            TrainingMetadata,
            fact_id=fact_id,
            base_model=base_model,
            adapter_version=adapter_version,
            training_iteration=None,  # TODO: Get from config
            training_date=None,  # TODO: Get from adapter metadata
            training_data_size=None,  # TODO: Get from training logs
            is_base_model=False,
        )
        self._lg.debug(
            "training metadata recorded (fine-tuned)",
            extra={
                "fact_id": fact_id,
                "model": model_name,
                "base_model": base_model,
                "adapter_version": adapter_version,
            },
        )

    def _extract_base_model(self, model_name: str) -> str:
        """Extract base model name by removing trailing adapter suffixes.

        Only strips adapter keywords when they appear as trailing segments,
        not in the middle of the model name.
        """
        base_model = model_name
        lower_model = model_name.lower()
        for suffix in ["-jokester", "-joke-teller", "-adapter", "-lora", "-qlora"]:
            # Check if suffix appears at the end or followed by version separator
            if lower_model.endswith(suffix) or (suffix + "-v") in lower_model:
                # Find last occurrence of the suffix
                idx = lower_model.rfind(suffix)
                base_model = base_model[:idx]
                break
        return base_model

    def _extract_adapter_version(self, model_name: str) -> str | None:
        """Extract adapter version from model name.

        Supports both integer versions (v1, v4) and dotted versions (v1.2, v2.0.3).
        Examples: 'llama-jokester-v4' -> 'v4', 'model-adapter-v1.2.3' -> 'v1.2.3'
        """
        parts = model_name.split("-")
        for part in reversed(parts):
            if part.startswith("v") and len(part) > 1:
                # Check if it's a valid version (digits with optional dots)
                version_part = part[1:]
                # Valid if all characters are digits or dots, and has at least one digit
                if all(c.isdigit() or c == "." for c in version_part) and any(
                    c.isdigit() for c in version_part
                ):
                    return part
        return None
