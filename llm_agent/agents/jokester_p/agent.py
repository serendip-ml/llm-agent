"""Programmatic joke-teller agent with guaranteed novelty checking.

This agent demonstrates the difference between prompt-based and programmatic agents:
- Prompt-based: Relies on LLM to follow instructions (probabilistic)
- Programmatic: Code enforces constraints (deterministic)

For joke-telling, the key constraint is "never repeat" across thousands of jokes,
which cannot be reliably enforced through prompts alone due to context window limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger
from llm_infer.client.exceptions import BackendUnavailableError
from pydantic import BaseModel

from ...core.agent import Agent, ExecutionResult, Identity
from ...core.llm.backend import StructuredOutputError
from ...core.traits.builtin.directive import DirectiveTrait
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.llm import LLMTrait
from ...core.traits.builtin.storage import StorageTrait
from .schema import ModelUsage, TrainingMetadata


class Joke(BaseModel):
    """Structured joke output."""

    text: str
    style: str  # pun, one-liner, observational, absurdist, wordplay, dark, etc.


@dataclass
class NoveltyCheck:
    """Result of novelty checking against existing jokes."""

    is_novel: bool
    max_similarity: float
    similar_joke: str | None


class JokesterAgent(Agent):
    """Programmatic agent that tells jokes with guaranteed novelty checking.

    Unlike prompt-based agents that rely on LLM instructions, this agent uses
    code to enforce the "never repeat" constraint across thousands of jokes.

    Flow:
        1. Recall recent jokes for style context
        2. LLM generates candidate joke
        3. Code validates novelty via embedding similarity (guaranteed check)
        4. If too similar, retry with feedback (code-controlled loop)
        5. Save novel joke to memory
    """

    def __init__(
        self,
        lg: Logger,
        identity: Identity,
        max_retries: int = 3,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize joke-teller agent.

        Args:
            lg: Logger instance.
            identity: Agent identity.
            max_retries: Maximum attempts to generate novel joke.
            similarity_threshold: Minimum similarity to consider a duplicate (0.0-1.0).
        """
        super().__init__(lg)
        self._identity = identity
        self._max_retries = max_retries
        self._similarity_threshold = similarity_threshold
        self._cycle_count = 0
        self._recent_results: list[ExecutionResult] = []
        self._jokes_generated_this_session = 0  # Track jokes generated since start

    @property
    def name(self) -> str:
        """Agent name from identity."""
        return self._identity.name

    @property
    def identity(self) -> Identity:
        """Agent identity."""
        return self._identity

    @property
    def cycle_count(self) -> int:
        """Number of execution cycles completed."""
        return self._cycle_count

    def start(self) -> None:
        """Start agent and traits."""
        self._start_traits()

        # Verify embedder is available (required for novelty checking)
        learn_trait = self.require_trait(LearnTrait)
        if not learn_trait.has_embedder:
            self._lg.error(
                "embedder not configured - required for novelty checking",
                extra={"agent": self.name},
            )
            # Clean up started traits before raising
            self._stop_traits()
            raise RuntimeError(
                "JokesterAgent requires embedder for guaranteed novelty checking. "
                "Configure embedder_url in learn section."
            )

        # Register agent-specific tables
        storage_trait = self.require_trait(StorageTrait)
        storage_trait.storage.register_table(ModelUsage)
        storage_trait.storage.register_table(TrainingMetadata)

        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop agent and traits."""
        if not self._started:
            return
        self._stop_traits()
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self.name})

    def run_once(self) -> ExecutionResult:
        """Execute one joke-telling cycle with novelty checking.

        Returns:
            ExecutionResult with the joke or error.
        """
        self._cycle_count += 1

        try:
            llm_trait = self.require_trait(LLMTrait)
            learn_trait = self.require_trait(LearnTrait)
            recent_jokes = self._get_recent_jokes(learn_trait, limit=5)

            # Generate novel joke with retry loop
            joke, attempts, max_similarity, similar_joke = self._generate_novel_joke(
                llm_trait, learn_trait, recent_jokes
            )

            if joke is None:
                return ExecutionResult(
                    success=False,
                    content=f"Failed to generate novel joke after {attempts} attempts",
                    iterations=attempts,
                )

            return self._complete_cycle(learn_trait, joke, attempts, max_similarity, similar_joke)

        except StructuredOutputError as e:
            self._lg.warning("joke generation failed", extra={"error": str(e)})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)
        except BackendUnavailableError as e:
            self._lg.warning("llm unavailable - will retry next cycle", extra={"error": str(e)})
            return ExecutionResult(success=False, content="LLM service unavailable", iterations=1)
        except Exception as e:
            self._lg.warning("joke generation failed", extra={"exception": e})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)

    def _complete_cycle(
        self,
        learn_trait: LearnTrait,
        joke: Joke,
        attempts: int,
        max_similarity: float,
        similar_joke: str | None,
    ) -> ExecutionResult:
        """Complete joke generation cycle with save and logging.

        Args:
            learn_trait: Learn trait for saving joke.
            joke: Generated joke.
            attempts: Number of generation attempts.
            max_similarity: Similarity score to closest existing joke (0.0-1.0).
            similar_joke: The closest existing joke text (if any).

        Returns:
            ExecutionResult with success status.
        """
        self._save_joke(learn_trait, joke)
        self._jokes_generated_this_session += 1

        result = ExecutionResult(
            success=True,
            content=f"{joke.text}\n(Style: {joke.style})",
            iterations=attempts,
        )
        self._recent_results.append(result)

        log_extra = {
            "agent": self.name,
            "success": result.success,
            "attempts": attempts,
            "style": joke.style,
            "joke": joke.text,
            "session_count": self._jokes_generated_this_session,
            "closest": {
                "similarity": max_similarity,
                "joke": similar_joke if similar_joke else "",
            },
        }

        self._lg.info("joke generation completed", extra=log_extra)

        return result

    def _generate_novel_joke(
        self, llm_trait: LLMTrait, learn_trait: LearnTrait, context: list[str]
    ) -> tuple[Joke | None, int, float, str | None]:
        """Generate novel joke with retry loop.

        Attempts joke generation with retries on failure.
        max_retries=0 means 1 attempt (no retries).
        max_retries=3 means 4 attempts (1 initial + 3 retries).

        Returns:
            Tuple of (joke, attempts, max_similarity, similar_joke) where joke is None if failed.
            max_similarity is the similarity to closest existing joke (0.0-1.0).
            similar_joke is the text of the closest existing joke (if any).
        """
        max_attempts = self._max_retries + 1  # retries + initial attempt
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._lg.debug(
                    "retrying joke generation...",
                    extra={"attempt": attempt, "max_retries": self._max_retries},
                )

            retry_feedback = "" if attempt == 1 else "Try a completely different style."
            joke = self._generate_joke(llm_trait, context, retry_feedback)

            if joke is None:
                self._lg.warning("LLM failed to generate joke", extra={"attempt": attempt})
                continue

            # Check novelty if embedder available
            if learn_trait.has_embedder:
                novelty = self._check_novelty(learn_trait, joke.text)
                if not novelty.is_novel:
                    continue
                return joke, attempt, novelty.max_similarity, novelty.similar_joke

            # No embedder - assume novel with 0.0 similarity
            return joke, attempt, 0.0, None

        self._lg.debug(
            "failed to generate novel joke after all attempts",
            extra={"max_retries": self._max_retries, "total_attempts": max_attempts},
        )
        return None, max_attempts, 0.0, None

    def ask(self, question: str) -> str:
        """Answer question (not supported)."""
        return "JokesterAgent does not support questions. Use run_once() to get a joke."

    def record_feedback(self, message: str) -> None:
        """Record feedback about a joke."""
        self._lg.info("feedback received", extra={"feedback": message})

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results."""
        return self._recent_results[-limit:]

    def _get_recent_jokes(self, learn_trait: LearnTrait, limit: int) -> list[str]:
        """Fetch recent jokes chronologically for style inspiration."""
        try:
            # Fetch recent facts and filter by category
            all_facts = learn_trait.learn.facts.list(limit=limit * 2)
            jokes = [f.content for f in all_facts if f.category == "joke"]
            return jokes[:limit]
        except Exception as e:
            self._lg.debug("failed to fetch recent jokes", extra={"exception": e})
            return []

    def _generate_joke(
        self, llm_trait: LLMTrait, context: list[str], retry_feedback: str
    ) -> Joke | None:
        """Generate a joke using LLM with structured output."""
        from ...core.llm.types import Message

        # Build prompt
        directive_trait = self.get_trait(DirectiveTrait)
        directive = directive_trait.directive.prompt if directive_trait else ""

        context_text = ""
        if context:
            context_text = "Recent jokes you've told:\n" + "\n".join(f"- {j}" for j in context)

        retry_text = f"\n\n{retry_feedback}" if retry_feedback else ""

        prompt = f"""{directive}

{context_text}

Tell one short, original joke (1-4 lines max). Choose a style you haven't used recently.{retry_text}

Return your joke in JSON format with 'text' and 'style' fields."""

        messages = [Message(role="user", content=prompt)]
        result = llm_trait.complete(messages, output_schema=Joke)

        if result.parsed is None:
            self._lg.warning("LLM failed to generate structured joke")
            return None

        return result.parsed  # type: ignore[no-any-return]

    def _check_novelty(self, learn_trait: LearnTrait, joke_text: str) -> NoveltyCheck:
        """Check if joke is novel using embedding similarity (RAG)."""
        self._lg.debug("checking joke novelty...", extra={"joke": joke_text})
        try:
            # Query without min_similarity to get closest joke regardless
            similar_facts = learn_trait.recall(
                query=joke_text,
                top_k=1,
                categories=["joke"],
            )

            if not similar_facts:
                self._lg.debug("joke is novel (no existing jokes)", extra={"joke": joke_text})
                return NoveltyCheck(is_novel=True, max_similarity=0.0, similar_joke=None)

            return self._evaluate_similarity(joke_text, similar_facts[0])

        except Exception as e:
            # Fail closed: reject joke when novelty check system fails to maintain
            # the "never repeat" guarantee. Caller will retry with a new joke.
            self._lg.warning(
                "novelty check failed, rejecting joke to maintain never-repeat guarantee",
                extra={"exception": e, "joke": joke_text},
            )
            return NoveltyCheck(is_novel=False, max_similarity=1.0, similar_joke=None)

    def _evaluate_similarity(self, joke_text: str, closest: Any) -> NoveltyCheck:
        """Evaluate similarity of joke against closest match.

        Args:
            joke_text: The candidate joke text.
            closest: The most similar existing joke from recall.

        Returns:
            NoveltyCheck indicating if joke is novel.
        """
        # Check if similarity exceeds threshold
        if closest.score >= self._similarity_threshold:
            self._log_similarity_result(joke_text, closest, is_novel=False)
            return NoveltyCheck(
                is_novel=False,
                max_similarity=closest.score,
                similar_joke=closest.entity.content,
            )

        # Novel, but log closest match for context
        self._log_similarity_result(joke_text, closest, is_novel=True)
        return NoveltyCheck(
            is_novel=True, max_similarity=closest.score, similar_joke=closest.entity.content
        )

    def _log_similarity_result(self, candidate: str, similar: Any, is_novel: bool) -> None:
        """Log similarity check result.

        Args:
            candidate: The generated joke being checked.
            similar: The most similar existing joke from recall.
            is_novel: Whether the joke passed the novelty threshold.
        """
        if is_novel:
            self._lg.debug(
                "joke is novel",
                extra={
                    "similarity": similar.score,
                    "closest_existing": similar.entity.content,
                    # "candidate": candidate,
                },
            )
        else:
            self._lg.debug(
                "joke too similar",
                extra={
                    "similarity": similar.score,
                    "existing": similar.entity.content,
                    # "candidate": candidate,
                },
            )

    def _save_joke(self, learn_trait: LearnTrait, joke: Joke) -> None:
        """Save joke as a solution to the task of telling a joke.

        Stores joke as type=solution with task context and metadata.
        Embeddings are created automatically by solutions.record().

        Also records model usage and training metadata in agent-specific tables.

        Raises:
            Exception: If joke save fails (caller should handle to mark run as failed).
        """
        # Get model info (for now hardcoded, extensible for multi-model later)
        model_name = self._get_model_name()

        # Record joke as solution - auto-creates embedding from answer_text
        fact_id = learn_trait.learn.solutions.record(
            agent_name=self.name,
            problem="Tell one short, original joke",
            problem_context={"style_preference": "varied"},
            answer={"text": joke.text, "style": joke.style},
            answer_text=joke.text,
            tokens_used=0,  # Not tracked at joke level yet
            latency_ms=0,  # Not tracked at joke level yet
            category="joke",
            source=model_name,  # Now uses actual model name instead of "agent"
        )
        self._lg.debug("joke saved as solution", extra={"fact_id": fact_id, "style": joke.style})

        # Record model usage in agent-specific table
        self._record_model_usage(fact_id, model_name)

        # Record training metadata if using fine-tuned model
        self._record_training_metadata(fact_id, model_name)

    def _get_model_name(self) -> str:
        """Get the model name currently being used.

        For now, returns the LLM backend model. In future, this could support
        tracking multiple models or fine-tuned adapters.

        Returns:
            Model name (e.g., 'claude-sonnet-4-5', 'llama-3.3-70b-instruct').
        """
        llm_trait = self.require_trait(LLMTrait)
        # Get model from LLM trait backend
        model_name = llm_trait.llm.model_name if hasattr(llm_trait.llm, 'model_name') else "unknown"
        return model_name

    def _record_model_usage(self, fact_id: int, model_name: str) -> None:
        """Record model usage metadata in agent-specific table.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
        """
        try:
            storage = self.require_trait(StorageTrait).storage
            storage.insert(
                ModelUsage,
                fact_id=fact_id,
                model_name=model_name,
                model_role="sole",  # Single model for now, extensible for multi-model
                tokens_in=None,  # TODO: Track from LLM response
                tokens_out=None,  # TODO: Track from LLM response
                cost_usd=None,  # TODO: Calculate based on model pricing
                latency_ms=None,  # TODO: Track from LLM response
            )
            self._lg.debug(
                "model usage recorded",
                extra={"fact_id": fact_id, "model": model_name}
            )
        except Exception as e:
            self._lg.warning(
                "failed to record model usage",
                extra={"exception": e, "fact_id": fact_id}
            )

    def _record_training_metadata(self, fact_id: int, model_name: str) -> None:
        """Record training metadata if using a fine-tuned model.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Name of the model used.
        """
        # Check if this is a fine-tuned model (contains version suffix)
        # Example: "llama-70b-jokester-v4" -> adapter_version="v4"
        is_finetuned = "jokester" in model_name.lower()

        if not is_finetuned:
            # Base model, record as such
            try:
                storage = self.require_trait(StorageTrait).storage
                storage.insert(
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
                    extra={"fact_id": fact_id, "model": model_name}
                )
            except Exception as e:
                self._lg.warning(
                    "failed to record training metadata",
                    extra={"exception": e, "fact_id": fact_id}
                )
        else:
            # Fine-tuned model - extract version info
            # TODO: Parse adapter version from model_name or configuration
            # For now, this is a placeholder for future fine-tuned versions
            try:
                storage = self.require_trait(StorageTrait).storage
                storage.insert(
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
                    extra={"fact_id": fact_id, "model": model_name}
                )
            except Exception as e:
                self._lg.warning(
                    "failed to record training metadata",
                    extra={"exception": e, "fact_id": fact_id}
                )
