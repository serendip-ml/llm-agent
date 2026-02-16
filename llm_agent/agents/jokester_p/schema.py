"""Database schema for jokester agent storage.

Tracks model usage and training metadata for joke generation.
"""

from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Boolean, Date, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from llm_agent.storage import AgentTable


class ModelUsage(AgentTable):
    """Tracks which model(s) generated each joke and their resource usage.

    Enables:
    - Attribution: Which model generated which joke
    - Cost analysis: Total cost per model
    - Performance tracking: Latency/throughput per model
    - Multi-model support: Track when multiple models collaborate

    Links to atomic_facts.id via fact_id.
    """

    __tablename__ = "agent_jokester_model_usage"

    # Link to atomic_facts
    fact_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Foreign key to atomic_facts.id",
    )

    # Model identification
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Model name (e.g., 'claude-sonnet-3.5', 'llama-70b-jokester-v4')",
    )

    model_role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Role in generation: 'sole', 'generator', 'validator', 'refiner'",
    )

    # Generation metrics
    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Number of generation attempts needed (for novelty checking)",
    )

    # Resource usage
    tokens_in: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Input tokens consumed"
    )

    tokens_out: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Output tokens generated"
    )

    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True, comment="Cost in USD")

    latency_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Generation latency in milliseconds"
    )


class TrainingMetadata(AgentTable):
    """Tracks training iterations for fine-tuned models.

    Enables:
    - Training progression: Track improvements across iterations
    - Version management: Map jokes to specific adapter versions
    - Data provenance: Know which training data produced which version

    Links to atomic_facts.id via fact_id.
    """

    __tablename__ = "agent_jokester_training"

    # Link to atomic_facts
    fact_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Foreign key to atomic_facts.id",
    )

    # Model info
    base_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Base model name (e.g., 'llama-3.3-70b-instruct')",
    )

    adapter_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Adapter version (e.g., 'v1', 'v2', 'v3', 'v4')",
    )

    # Training iteration tracking
    training_iteration: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="Training iteration number (1, 2, 3, 4)",
    )

    training_date: Mapped[Date | None] = mapped_column(
        Date,
        nullable=True,
        comment="Date when this version was trained",
    )

    training_data_size: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of ratings used for training this version",
    )

    # Whether this is base model (no fine-tuning) or fine-tuned
    is_base_model: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True if base model without fine-tuning",
    )

    # Adapter fallback tracking
    adapter_requested: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Adapter that was requested (even if not used due to fallback)",
    )

    adapter_fallback: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True if adapter was requested but unavailable (fell back to base)",
    )

    # Full adapter info as JSON (md5, mtime, actual, etc.)
    adapter_info: Mapped[dict[str, object] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Full AdapterInfo from LLM response (md5, mtime, actual, requested, fallback)",
    )
