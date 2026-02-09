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
from pydantic import BaseModel

from ...core.agent import Agent, ExecutionResult, Identity
from ...core.traits.builtin.directive import DirectiveTrait
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.llm import LLMTrait


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


class JokeTellerAgent(Agent):
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
            joke, attempts = self._generate_novel_joke(llm_trait, learn_trait, recent_jokes)

            if joke is None:
                return ExecutionResult(
                    success=False,
                    content=f"Failed to generate novel joke after {attempts} attempts",
                    iterations=attempts,
                )

            return self._complete_cycle(learn_trait, joke, attempts)

        except Exception as e:
            from ...core.llm.backend import StructuredOutputError

            if isinstance(e, StructuredOutputError):
                self._lg.warning("joke generation failed", extra={"error": str(e)})
            else:
                self._lg.warning("joke generation failed", extra={"exception": e})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)

    def _complete_cycle(
        self, learn_trait: LearnTrait, joke: Joke, attempts: int
    ) -> ExecutionResult:
        """Complete joke generation cycle with save and logging.

        Args:
            learn_trait: Learn trait for saving joke.
            joke: Generated joke.
            attempts: Number of generation attempts.

        Returns:
            ExecutionResult with success status.
        """
        self._save_joke(learn_trait, joke)
        result = ExecutionResult(
            success=True,
            content=f"{joke.text}\n(Style: {joke.style})",
            iterations=attempts,
        )
        self._recent_results.append(result)

        self._lg.info(
            "joke generation completed",
            extra={
                "agent": self.name,
                "success": result.success,
                "attempts": attempts,
                "style": joke.style,
                "joke": joke.text,
            },
        )

        return result

    def _generate_novel_joke(
        self, llm_trait: LLMTrait, learn_trait: LearnTrait, context: list[str]
    ) -> tuple[Joke | None, int]:
        """Generate novel joke with retry loop.

        Returns:
            Tuple of (joke, attempts) where joke is None if failed.
        """
        for attempt in range(1, self._max_retries + 1):
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

            return joke, attempt

        self._lg.warning(
            "failed to generate novel joke after all retries",
            extra={"max_retries": self._max_retries},
        )
        return None, self._max_retries

    def ask(self, question: str) -> str:
        """Answer question (not supported)."""
        return "JokeTellerAgent does not support questions. Use run_once() to get a joke."

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
            self._lg.warning("novelty check failed, assuming novel", extra={"exception": e})
            return NoveltyCheck(is_novel=True, max_similarity=0.0, similar_joke=None)

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
        return NoveltyCheck(is_novel=True, max_similarity=closest.score, similar_joke=None)

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
        """Save joke to memory with embedding for future novelty checking."""
        try:
            # remember() automatically creates embeddings if embedder available
            fact_id = learn_trait.remember(
                fact=joke.text,
                category="joke",
                source="system",
                confidence=1.0,
            )
            self._lg.debug("joke saved", extra={"fact_id": fact_id, "style": joke.style})
        except Exception as e:
            self._lg.error("failed to save joke", extra={"exception": e})
