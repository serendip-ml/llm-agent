"""Programmatic joke-teller agent with guaranteed novelty checking.

This agent demonstrates the difference between prompt-based and programmatic agents:
- Prompt-based: Relies on LLM to follow instructions (probabilistic)
- Programmatic: Code enforces constraints (deterministic)

For joke-telling, the key constraint is "never repeat" across thousands of jokes,
which cannot be reliably enforced through prompts alone due to context window limits.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger
from llm_infer.client.exceptions import BackendUnavailableError
from pydantic import BaseModel

from ...core.agent import Agent, ExecutionResult
from ...core.llm.backend import StructuredOutputError
from ...core.traits.builtin.directive import DirectiveTrait
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.llm import LLMTrait
from ...core.traits.builtin.storage import StorageTrait
from .schema import ModelUsage, TrainingMetadata
from .storage import Storage


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
        config: DotDict,
    ) -> None:
        """Initialize joke-teller agent.

        Args:
            lg: Logger instance.
            config: Agent configuration with keys:
                - identity.name: Agent name (required)
                - max_retries: Maximum attempts to generate novel joke (default: 3)
                - similarity_threshold: Min similarity for duplicate (0.0-1.0, default: 0.85)
                - denylist: List of words/phrases to filter (case-insensitive, default: [])
        """
        super().__init__(lg, config)
        self._max_retries = self.config.get("max_retries", 3)
        self._similarity_threshold = self.config.get("similarity_threshold", 0.85)
        denylist = self.config.get("denylist", [])
        self._denylist = [term.lower() for term in denylist] if denylist else []
        self._jokes_generated_this_session = 0  # Track jokes generated since start
        self._cumulative_attempts = 0  # Track total attempts until successful save
        self._recent_failed_jokes: deque[str] = deque(
            maxlen=10
        )  # Track recent failed attempts to avoid similar jokes
        self._recent_results: deque[ExecutionResult] = deque(
            maxlen=100
        )  # Bounded to prevent memory leak
        self._storage: Storage | None = None  # Set in start(), None before agent is started

    def start(self) -> None:
        """Start agent and traits."""
        self._start_traits()

        try:
            self._verify_embedder_available()
            self._setup_storage()
        except Exception:
            self._stop_traits()
            raise

        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def _verify_embedder_available(self) -> None:
        """Verify embedder is configured for novelty checking.

        Raises:
            RuntimeError: If embedder is not configured.
        """
        learn_trait = self.require_trait(LearnTrait)
        if not learn_trait.has_embedder:
            self._lg.error(
                "embedder not configured - required for novelty checking",
                extra={"agent": self.name},
            )
            raise RuntimeError(
                "Jokester agent requires embedder for guaranteed novelty checking. "
                "Configure embedder_url in learn section."
            )

    def _setup_storage(self) -> None:
        """Register tables and initialize storage helper.

        Raises:
            Exception: If storage setup fails.
        """
        storage_trait = self.require_trait(StorageTrait)
        storage_trait.storage.register_table(ModelUsage)
        storage_trait.storage.register_table(TrainingMetadata)

        self._storage = Storage(self._lg, storage_trait)

    def stop(self) -> None:
        """Stop agent and traits."""
        if not self._started:
            return
        self._stop_traits()
        self._storage = None
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self.name})

    def run_once(self) -> ExecutionResult:
        """Execute one joke-telling cycle with novelty checking.

        Returns:
            ExecutionResult with the joke or error.
        """
        self._cycle_count += 1

        try:
            return self._execute_joke_cycle()
        except StructuredOutputError as e:
            self._lg.warning("joke generation failed", extra={"error": str(e)})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)
        except BackendUnavailableError as e:
            self._lg.warning("llm unavailable - will retry next cycle", extra={"error": str(e)})
            return ExecutionResult(success=False, content="LLM service unavailable", iterations=1)
        except Exception as e:
            self._lg.warning("joke generation failed", extra={"exception": e})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)

    def _execute_joke_cycle(self) -> ExecutionResult:
        """Execute the core joke generation and novelty checking logic.

        Returns:
            ExecutionResult with the joke or failure message.
        """
        llm_trait = self.require_trait(LLMTrait)
        learn_trait = self.require_trait(LearnTrait)
        recent_jokes = self._get_recent_jokes(learn_trait, limit=5)

        (
            joke,
            run_attempts,
            cumulative_attempts,
            max_similarity,
            similar_joke,
            model_name,
        ) = self._generate_novel_joke(llm_trait, learn_trait, recent_jokes)

        if joke is None:
            return ExecutionResult(
                success=False,
                content=f"Failed to generate novel joke after {cumulative_attempts} attempts",
                iterations=cumulative_attempts,
            )

        return self._complete_cycle(
            learn_trait,
            joke,
            run_attempts,
            cumulative_attempts,
            max_similarity,
            similar_joke,
            model_name,
        )

    def _complete_cycle(
        self,
        learn_trait: LearnTrait,
        joke: Joke,
        run_attempts: int,
        cumulative_attempts: int,
        max_similarity: float,
        similar_joke: str | None,
        model_name: str,
    ) -> ExecutionResult:
        """Complete joke generation cycle with save and logging.

        Args:
            learn_trait: Learn trait for saving joke.
            joke: Generated joke.
            run_attempts: Number of attempts in this specific run.
            cumulative_attempts: Total attempts across all runs until success.
            max_similarity: Similarity score to closest existing joke (0.0-1.0).
            similar_joke: The closest existing joke text (if any).
            model_name: Actual model used (from LLM response).

        Returns:
            ExecutionResult with success status.
        """
        fact_id = self._save_joke(learn_trait, joke, model_name, cumulative_attempts)
        self._jokes_generated_this_session += 1

        # Auto-rate if enabled (uses rating conductors, not generation LLM)
        if self.config.get("rating", {}).get("auto", False):
            self._perform_inline_rating(fact_id, joke.text)

        result = ExecutionResult(
            success=True,
            content=f"{joke.text}\n(Style: {joke.style})",
            iterations=cumulative_attempts,
        )
        self._recent_results.append(result)

        log_extra = self._build_joke_log_extra(
            result, joke, run_attempts, cumulative_attempts, max_similarity, similar_joke
        )
        self._lg.info("found new joke", extra=log_extra)

        return result

    def _build_joke_log_extra(
        self,
        result: ExecutionResult,
        joke: Joke,
        run_attempts: int,
        cumulative_attempts: int,
        max_similarity: float,
        similar_joke: str | None,
    ) -> dict[str, Any]:
        """Build log extra dict for joke generation."""
        return {
            "agent": self.name,
            "success": result.success,
            "attempts": {
                "run": run_attempts,
                "cumulative": cumulative_attempts,
            },
            "style": joke.style,
            "joke": joke.text,
            "session_count": self._jokes_generated_this_session,
            "closest": {
                "similarity": max_similarity,
                "joke": similar_joke if similar_joke else "",
            },
        }

    def _perform_inline_rating(self, fact_id: int, joke_text: str) -> None:
        """Perform inline rating of a joke using configured rating conductors.

        Args:
            fact_id: ID of the saved joke fact.
            joke_text: Text of the joke to rate.
        """
        from ...core.traits.builtin.rating import RatingTrait

        rating_trait = self.get_trait(RatingTrait)
        if not rating_trait:
            return

        try:
            results = rating_trait.rate_fact_with_all_providers(
                fact_id=fact_id, content=joke_text, fact_type="solution"
            )
            # Log summary of ratings
            for result in results:
                stars_visual = "★" * result.stars + "☆" * (5 - result.stars)
                reasoning_preview = (
                    result.reasoning[:100] + "..."
                    if len(result.reasoning) > 100
                    else result.reasoning
                )
                self._lg.info(
                    f"inline rating: {stars_visual}",
                    extra={
                        "fact_id": fact_id,
                        "stars": result.stars,
                        "model": result.model,
                        "reasoning": reasoning_preview,
                    },
                )
        except Exception as e:
            self._lg.warning("inline rating failed", extra={"exception": e, "fact_id": fact_id})

    def _generate_novel_joke(
        self, llm_trait: LLMTrait, learn_trait: LearnTrait, context: list[str]
    ) -> tuple[Joke | None, int, int, float, str | None, str]:
        """Generate novel joke with retry loop.

        Returns:
            Tuple of (joke, run_attempts, cumulative_attempts, max_similarity, similar_joke, model_name).
            run_attempts is the count for this specific run (1 to max_retries+1).
            cumulative_attempts is the total count across all runs since last successful save.
            model_name is the actual model used from the last attempt, even on failure.
        """
        max_attempts = self._max_retries + 1
        last_model_name = "unknown"

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._lg.debug("retrying joke generation...", extra={"attempt": attempt})

            result = self._try_single_joke_attempt(llm_trait, learn_trait, context, attempt)
            if result is not None:
                joke, last_model_name, novelty = result
                self._recent_failed_jokes.clear()
                sim = novelty.max_similarity if novelty else 0.0
                similar = novelty.similar_joke if novelty else None
                return joke, attempt, self._cumulative_attempts, sim, similar, last_model_name

        self._lg.debug("failed to generate novel joke", extra={"attempts": max_attempts})
        return None, max_attempts, self._cumulative_attempts, 0.0, None, last_model_name

    def _try_single_joke_attempt(
        self, llm_trait: LLMTrait, learn_trait: LearnTrait, context: list[str], attempt: int
    ) -> tuple[Joke, str, NoveltyCheck | None] | None:
        """Try to generate a single novel joke.

        Returns:
            Tuple of (joke, model_name, novelty_check) if successful, None if failed.
        """
        self._cumulative_attempts += 1
        retry_feedback = "" if attempt == 1 else "Try a completely different style."
        joke, model_name = self._generate_joke(
            llm_trait, context, retry_feedback, list(self._recent_failed_jokes)
        )

        if joke is None:
            self._lg.warning("LLM failed to generate joke", extra={"attempt": attempt})
            return None

        return self._validate_joke(joke, model_name, learn_trait, attempt)

    def _validate_joke(
        self, joke: Joke, model_name: str, learn_trait: LearnTrait, attempt: int
    ) -> tuple[Joke, str, NoveltyCheck | None] | None:
        """Validate joke against denylist and novelty checks.

        Returns:
            Tuple of (joke, model_name, novelty_check) if valid, None if invalid.
        """
        # Check denylist
        contains_denied, denied_term = self._contains_denied_content(joke.text)
        if contains_denied:
            self._lg.debug("joke denied", extra={"term": denied_term, "attempt": attempt})
            self._recent_failed_jokes.append(joke.text)
            return None

        # Check novelty if embedder available
        if learn_trait.has_embedder:
            novelty = self._check_novelty(learn_trait, joke.text)
            if not novelty.is_novel:
                self._recent_failed_jokes.append(joke.text)
                return None
            return joke, model_name, novelty

        return joke, model_name, None

    def ask(self, question: str) -> str:
        """Answer question (not supported)."""
        return "Jokester agent does not support questions. Use run_once() to get a joke."

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
        self, llm_trait: LLMTrait, context: list[str], retry_feedback: str, avoid_jokes: list[str]
    ) -> tuple[Joke | None, str]:
        """Generate a joke using LLM with structured output.

        Args:
            llm_trait: LLM trait for generation.
            context: Recent successful jokes for style inspiration.
            retry_feedback: Feedback for retry attempts.
            avoid_jokes: Recent failed attempts to avoid generating similar jokes.

        Returns:
            Tuple of (joke, model_name) where joke is None if generation failed.
        """
        from ...core.llm.types import Message

        prompt = self._build_joke_prompt(context, retry_feedback, avoid_jokes)
        messages = [Message(role="user", content=prompt)]
        result = llm_trait.complete(messages, output_schema=Joke)

        if result.parsed is None:
            self._lg.warning("LLM failed to generate structured joke")
            return None, result.model

        return result.parsed, result.model

    def _build_joke_prompt(
        self, context: list[str], retry_feedback: str, avoid_jokes: list[str]
    ) -> str:
        """Build prompt for joke generation with context and constraints.

        Args:
            context: Recent successful jokes for style inspiration.
            retry_feedback: Feedback for retry attempts.
            avoid_jokes: Recent failed attempts to avoid generating similar jokes.

        Returns:
            Formatted prompt string.
        """
        directive_trait = self.get_trait(DirectiveTrait)
        directive = directive_trait.directive.prompt if directive_trait else ""

        context_text = ""
        if context:
            context_text = "Recent jokes you've told:\n" + "\n".join(f"- {j}" for j in context)

        avoid_text = ""
        if avoid_jokes:
            recent_failures = avoid_jokes[-5:]
            avoid_text = (
                "\n\nDO NOT generate anything similar to these recent attempts "
                "(too similar to existing jokes):\n" + "\n".join(f"- {j}" for j in recent_failures)
            )

        retry_text = f"\n\n{retry_feedback}" if retry_feedback else ""

        return f"""{directive}

{context_text}{avoid_text}

Tell one short, original joke (1-4 lines max).

IMPORTANT: Vary your joke structure and opening. Don't always start with the same patterns, such as "I've been trying". Mix it up - use questions, observations, one-liners, puns, different formats.{retry_text}

Return your joke in JSON format with 'text' and 'style' fields."""

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

    def _contains_denied_content(self, joke_text: str) -> tuple[bool, str | None]:
        """Check if joke contains denied words/phrases.

        Args:
            joke_text: The joke text to check.

        Returns:
            Tuple of (contains_denied, denied_term) where denied_term is the matched term if found.
        """
        if not self._denylist:
            return False, None

        joke_lower = joke_text.lower()
        for term in self._denylist:
            if term in joke_lower:
                return True, term
        return False, None

    def _save_joke(
        self, learn_trait: LearnTrait, joke: Joke, model_name: str, attempts: int
    ) -> int:
        """Save joke as a solution to the task of telling a joke.

        Stores joke as type=solution with task context and metadata.
        Embeddings are created automatically by solutions.record().

        Also records model usage and training metadata in agent-specific tables.

        Args:
            learn_trait: Learn trait for saving joke.
            joke: Generated joke.
            model_name: Actual model used (from LLM response).
            attempts: Number of generation attempts needed.

        Returns:
            The fact_id of the saved joke.

        Raises:
            Exception: If joke save fails (caller should handle to mark run as failed).
        """
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
            source="agent",  # Generic source, model tracked in metadata table
        )
        self._lg.debug("joke saved as solution", extra={"fact_id": fact_id, "style": joke.style})

        # Record model usage and training metadata via storage helper
        # Storage is always initialized in start(), so it should never be None here
        if self._storage is None:
            raise RuntimeError("Storage not initialized - call agent.start() first")
        self._storage.record_joke_metadata(fact_id, model_name, attempts)

        # Reset cumulative attempts counter after successful save
        self._cumulative_attempts = 0

        return fact_id

    def record_feedback(self, message: str) -> None:
        """Record feedback about a joke.

        Args:
            message: Feedback message from user.
        """
        # Log at debug level to avoid exposing PII in production logs
        self._lg.debug("feedback received", extra={"feedback": message})

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent execution results.
        """
        return list(self._recent_results)[-limit:]
