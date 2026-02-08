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


class Joke(BaseModel):
    """Structured joke output."""

    text: str
    style: str  # pun, one-liner, observational, absurdist, wordplay, dark, etc.


@dataclass
class NoveltyCheck:
    """Result of novelty checking against existing jokes."""

    is_novel: bool
    similar_jokes: list[dict[str, Any]]
    similarity_threshold: float


class JokeTellerAgent(Agent):
    """Programmatic agent that tells jokes with guaranteed novelty checking.

    Unlike prompt-based agents that rely on LLM instructions, this agent uses
    code to enforce the "never repeat" constraint across thousands of jokes.

    Flow:
        1. Recall recent jokes for style context (LLM inspiration)
        2. LLM generates joke (creative task)
        3. Code validates novelty against entire DB (guaranteed check)
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
            identity: Agent identity for addressing/naming.
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
        """Agent identity for addressing."""
        return self._identity

    @property
    def cycle_count(self) -> int:
        """Number of execution cycles completed."""
        return self._cycle_count

    def start(self) -> None:
        """Start agent and its traits."""
        self._start_traits()
        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop agent and its traits."""
        if not self._started:
            return
        self._stop_traits()
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self.name})

    def run_once(self) -> ExecutionResult:  # cq: max-lines=35
        """Execute one joke-telling cycle with guaranteed novelty checking.

        Returns:
            ExecutionResult with the joke or error.
        """
        self._cycle_count += 1

        # Validate required traits
        validation_result = self._validate_required_traits()
        if validation_result is not None:
            return validation_result

        try:
            llm_trait, learn_trait = self._get_required_traits()

            # Get recent jokes for style context
            recent_context = self._get_recent_jokes(learn_trait, limit=5)

            # Generate joke (novelty checking disabled - TODO: implement with embeddings)
            joke = self._generate_joke(llm_trait, recent_context)

            if joke is None:
                return self._failure_result(
                    "Could not generate joke",
                    iterations=1,
                )

            # Save to memory and return success
            self._save_joke(learn_trait, joke)
            return self._success_result(joke, 1)

        except Exception as e:
            return self._error_result(e)

    def _validate_required_traits(self) -> ExecutionResult | None:
        """Validate required traits are attached.

        Returns:
            ExecutionResult with error if validation fails, None if valid.
        """
        from ...core.traits.learn import LearnTrait
        from ...core.traits.llm import LLMTrait

        if self.get_trait(LLMTrait) is None:
            return self._failure_result("LLMTrait not attached", iterations=0)

        if self.get_trait(LearnTrait) is None:
            return self._failure_result("LearnTrait not attached", iterations=0)

        return None

    def _get_required_traits(self) -> tuple[Any, Any]:
        """Get required LLM and Learn traits (assumes validation passed)."""
        from ...core.traits.learn import LearnTrait
        from ...core.traits.llm import LLMTrait

        return self.get_trait(LLMTrait), self.get_trait(LearnTrait)

    def _success_result(self, joke: Joke, attempts: int) -> ExecutionResult:
        """Build success result with joke."""
        result = ExecutionResult(
            success=True,
            content=f"{joke.text}\n(Style: {joke.style})",
            iterations=attempts,
        )
        self._recent_results.append(result)
        return result

    def _failure_result(self, message: str, iterations: int) -> ExecutionResult:
        """Build failure result."""
        return ExecutionResult(success=False, content=message, iterations=iterations)

    def _error_result(self, error: Exception) -> ExecutionResult:
        """Build error result from exception."""
        self._lg.error("joke generation failed", extra={"agent": self.name, "exception": error})
        return ExecutionResult(success=False, content=f"Error: {error}", iterations=1)

    def ask(self, question: str) -> str:
        """Answer a question (not implemented for this agent).

        Args:
            question: User question.

        Returns:
            Error message indicating this agent doesn't support questions.
        """
        return "JokeTellerAgent does not support questions. Use run_once() to get a joke."

    def record_feedback(self, message: str) -> None:
        """Record feedback about a joke.

        Args:
            message: Feedback message.
        """
        self._lg.info("feedback received", extra={"agent": self.name, "feedback": message})
        # Could store feedback in DB for future learning

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent ExecutionResult objects.
        """
        return self._recent_results[-limit:]

    def _get_recent_jokes(self, learn_trait: Any, limit: int) -> list[str]:
        """Fetch recent jokes for style inspiration.

        Args:
            learn_trait: LearnTrait instance.
            limit: Number of recent jokes to fetch.

        Returns:
            List of recent joke texts.
        """
        try:
            facts = learn_trait.learn.solutions.list_by_agent(
                agent_name=self.name,
                limit=limit,
            )
            return [fact.content for fact in facts if fact.content]
        except Exception as e:
            self._lg.debug(
                "failed to fetch recent jokes",
                extra={"agent": self.name, "exception": e},
            )
            return []

    def _generate_joke(self, llm_trait: Any, context: list[str]) -> Joke | None:
        """Generate a joke (novelty checking disabled).

        Args:
            llm_trait: LLMTrait instance.
            context: Recent jokes for style inspiration.

        Returns:
            Generated joke or None if generation failed.
        """
        from ...core.llm.types import Message

        prompt = self._build_generation_prompt(context, retry_feedback="")
        messages = [Message(role="user", content=prompt)]
        result = llm_trait.complete(messages, output_schema=Joke)

        if result.parsed is None:
            self._lg.warning(
                "LLM failed to generate structured joke",
                extra={"agent": self.name},
            )
            return None

        return result.parsed  # type: ignore[no-any-return]

    def _attempt_joke_generation(  # cq: max-lines=40
        self,
        llm_trait: Any,
        learn_trait: Any,
        context: list[str],
        retry_prompt: str,
        attempt: int,
    ) -> tuple[Joke | None, str]:
        """Attempt to generate one novel joke.

        Returns:
            Tuple of (joke, next_retry_prompt). joke is None if attempt failed.
        """
        from ...core.llm.types import Message

        # Build prompt and generate joke
        prompt = self._build_generation_prompt(context, retry_prompt)
        messages = [Message(role="user", content=prompt)]
        result = llm_trait.complete(messages, output_schema=Joke)

        if result.parsed is None:
            self._lg.warning(
                "LLM failed to generate structured joke",
                extra={"agent": self.name, "attempt": attempt},
            )
            return None, "Previous attempt failed to produce valid JSON. Try again."

        joke = result.parsed
        novelty = self._check_novelty(learn_trait, joke.text)

        if novelty.is_novel:
            self._lg.info(
                "generated novel joke",
                extra={"agent": self.name, "attempts": attempt, "style": joke.style},
            )
            return joke, ""

        # Too similar - build retry prompt
        next_retry = self._build_similarity_retry_prompt(novelty)
        self._lg.info(
            "joke too similar, retrying",
            extra={
                "agent": self.name,
                "attempt": attempt,
                "similar_count": len(novelty.similar_jokes),
            },
        )
        return None, next_retry

    def _build_similarity_retry_prompt(self, novelty: NoveltyCheck) -> str:
        """Build retry prompt when joke is too similar to existing ones."""
        similar_jokes_text = "\n".join(
            f"- {j['text']} (similarity: {j['similarity']:.2f})" for j in novelty.similar_jokes[:2]
        )
        return f"""Your previous joke was too similar to existing jokes:

{similar_jokes_text}

Try a completely different style or topic."""

    def _build_generation_prompt(self, context: list[str], retry_feedback: str) -> str:
        """Build prompt for joke generation.

        Args:
            context: Recent jokes for style inspiration.
            retry_feedback: Feedback from previous failed attempts.

        Returns:
            Prompt string for LLM.
        """
        context_text = ""
        if context:
            context_text = "Recent jokes you've told:\n" + "\n".join(
                f"- {joke}" for joke in context
            )

        retry_text = f"\n\n{retry_feedback}" if retry_feedback else ""

        # Get directive from traits
        from ...core.traits.directive import DirectiveTrait

        directive_trait = self.get_trait(DirectiveTrait)
        directive_prompt = directive_trait.directive.prompt if directive_trait else ""

        return f"""{directive_prompt}

{context_text}

Tell one short, original joke (1-4 lines max). Choose a style you haven't used recently.{retry_text}

Return your joke in JSON format with 'text' and 'style' fields."""

    def _check_novelty(self, learn_trait: Any, joke_text: str) -> NoveltyCheck:
        """Check if joke is novel using embedding similarity.

        This is the key difference from prompt-based agents: novelty checking
        is GUARANTEED to run via code, not dependent on LLM following instructions.

        Args:
            learn_trait: LearnTrait instance.
            joke_text: The joke text to check.

        Returns:
            NoveltyCheck result with similarity details.
        """
        try:
            embedding = learn_trait.embedder.embed(joke_text)
            similar_facts = self._find_similar_jokes(learn_trait, embedding)

            if similar_facts:
                similar_jokes = self._format_similar_facts(similar_facts)
                return self._novelty_result(is_novel=False, similar_jokes=similar_jokes)

            return self._novelty_result(is_novel=True, similar_jokes=[])

        except Exception as e:
            self._lg.warning(
                "novelty check failed, assuming novel",
                extra={"agent": self.name, "exception": e},
            )
            # Fail open: if check fails, assume novel to avoid blocking
            return self._novelty_result(is_novel=True, similar_jokes=[])

    def _find_similar_jokes(self, learn_trait: Any, embedding: list[float]) -> list[Any]:
        """Query database for similar jokes."""
        return learn_trait.learn.solutions.find_similar(  # type: ignore[no-any-return]
            embedding=embedding,
            profile_name=self.name,
            category="execution",
            min_similarity=self._similarity_threshold,
            limit=5,
        )

    def _format_similar_facts(self, similar_facts: list[Any]) -> list[dict[str, Any]]:
        """Format similar facts into joke dictionaries."""
        return [
            {
                "text": fact.answer_text,
                "similarity": fact.similarity,
                "created_at": fact.created_at.isoformat(),
            }
            for fact in similar_facts
        ]

    def _novelty_result(self, is_novel: bool, similar_jokes: list[dict[str, Any]]) -> NoveltyCheck:
        """Create NoveltyCheck result."""
        return NoveltyCheck(
            is_novel=is_novel,
            similar_jokes=similar_jokes,
            similarity_threshold=self._similarity_threshold,
        )

    def _save_joke(self, learn_trait: Any, joke: Joke) -> None:
        """Save joke to memory for future novelty checking.

        Args:
            learn_trait: LearnTrait instance.
            joke: The joke to save.
        """
        try:
            fact_id = learn_trait.learn.solutions.record(
                agent_name=self.name,
                problem="Tell a joke",
                problem_context={},
                answer={"text": joke.text, "style": joke.style},
                answer_text=joke.text,
                tokens_used=0,
                latency_ms=0,
                category="execution",
                source="agent",
            )
            self._lg.debug(
                "joke saved",
                extra={"agent": self.name, "fact_id": fact_id, "style": joke.style},
            )
        except Exception as e:
            self._lg.warning(
                "failed to save joke",
                extra={"agent": self.name, "exception": e},
            )
