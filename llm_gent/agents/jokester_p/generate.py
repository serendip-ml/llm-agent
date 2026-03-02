"""Joke generation with novelty checking and retry logic.

Handles the complete joke generation cycle including LLM calls,
novelty validation, denylist filtering, and retry logic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_infer.client.types import AdapterInfo
from pydantic import AliasChoices, BaseModel, Field

from .novelty import NoveltyCheck, NoveltyChecker


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ...core.traits.builtin.directive import DirectiveTrait
    from ...core.traits.builtin.llm import LLMTrait


class Joke(BaseModel):
    """Structured joke output."""

    text: str = Field(validation_alias=AliasChoices("text", "joke"))
    style: str  # pun, one-liner, observational, absurdist, wordplay, dark, etc.


@dataclass
class GenerationAttempt:
    """Result of a joke generation cycle with all metadata."""

    joke: Joke | None
    run_attempts: int
    cumulative_attempts: int
    model_name: str
    adapter: AdapterInfo | None  # Full adapter info from LLM response
    novelty: NoveltyCheck | None = None

    @property
    def success(self) -> bool:
        """Whether a novel joke was generated."""
        return self.joke is not None

    @property
    def adapter_fallback(self) -> bool:
        """Whether adapter fell back to base model."""
        return self.adapter.fallback if self.adapter else False

    @property
    def max_similarity(self) -> float:
        """Similarity to closest existing joke (0.0-1.0)."""
        return self.novelty.max_similarity if self.novelty else 0.0

    @property
    def similar_joke(self) -> str | None:
        """The closest existing joke text."""
        return self.novelty.similar_joke if self.novelty else None


class JokeGenerator:
    """Generates novel jokes with retry logic and validation.

    Handles the complete generation cycle:
    1. Build prompt with context and constraints
    2. Call LLM for structured joke output
    3. Validate against denylist
    4. Check novelty via embedding similarity
    5. Retry on failure up to max_retries
    """

    def __init__(
        self,
        lg: Logger,
        llm_trait: LLMTrait,
        novelty_checker: NoveltyChecker,
        directive_trait: DirectiveTrait | None = None,
        max_retries: int = 3,
        denylist: list[str] | None = None,
    ) -> None:
        """Initialize joke generator.

        Args:
            lg: Logger instance.
            llm_trait: LLM trait for generation.
            novelty_checker: Checker for joke novelty.
            directive_trait: Optional directive trait for system prompt.
            max_retries: Maximum retry attempts (default: 3).
            denylist: Words/phrases to filter (case-insensitive).
        """
        self._lg = lg
        self._llm = llm_trait
        self._novelty = novelty_checker
        self._directive = directive_trait
        self._max_retries = max_retries
        self._denylist = [term.lower() for term in (denylist or [])]
        self._cumulative_attempts = 0
        # Track (generated_joke, similar_existing_joke) pairs for smarter retries
        self._recent_failed: deque[tuple[str, str | None]] = deque(maxlen=10)

    def generate(self, context: list[str]) -> GenerationAttempt:
        """Generate a novel joke with retry loop.

        Args:
            context: Recent successful jokes for style inspiration.

        Returns:
            GenerationAttempt with result and metadata.
        """
        max_attempts = self._max_retries + 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._lg.debug("retrying joke generation...", extra={"attempt": attempt})

            result = self._try_single_attempt(context, attempt)
            if result is not None:
                return self._success_attempt(attempt, *result)

        self._lg.debug("failed to generate novel joke", extra={"attempts": max_attempts})
        return GenerationAttempt(
            joke=None,
            run_attempts=max_attempts,
            cumulative_attempts=self._cumulative_attempts,
            model_name="unknown",
            adapter=None,
        )

    def _success_attempt(
        self,
        attempt: int,
        joke: Joke,
        model: str,
        adapter: AdapterInfo | None,
        novelty: NoveltyCheck | None,
    ) -> GenerationAttempt:
        """Build successful generation attempt and reset state."""
        self._recent_failed.clear()
        cumulative = self._cumulative_attempts
        self._cumulative_attempts = 0
        return GenerationAttempt(
            joke=joke,
            run_attempts=attempt,
            cumulative_attempts=cumulative,
            model_name=model,
            adapter=adapter,
            novelty=novelty,
        )

    def _try_single_attempt(
        self, context: list[str], attempt: int
    ) -> tuple[Joke, str, AdapterInfo | None, NoveltyCheck | None] | None:
        """Try to generate a single novel joke."""
        self._cumulative_attempts += 1
        retry_feedback = "" if attempt == 1 else "Try a completely different style."

        joke, model_name, adapter = self._call_llm(context, retry_feedback)

        if joke is None:
            self._lg.warning("LLM failed to generate joke", extra={"attempt": attempt})
            return None

        return self._validate(joke, model_name, adapter, attempt)

    def _call_llm(
        self, context: list[str], retry_feedback: str
    ) -> tuple[Joke | None, str, AdapterInfo | None]:
        """Generate joke via LLM with structured output."""
        import appinfra.time
        from llm_infer.client.exceptions import BackendRequestError, BackendTimeoutError

        from ...core.llm.types import Message

        prompt = self._build_prompt(context, retry_feedback)
        self._lg.debug("sending LLM request...")
        start_t = appinfra.time.start()
        try:
            result = self._llm.complete([Message(role="user", content=prompt)], output_schema=Joke)
        except (BackendTimeoutError, BackendRequestError) as e:
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self._lg.error(
                "LLM request failed",
                extra={
                    "after": appinfra.time.since(start_t),
                    "prompt": prompt_preview,
                    "error": str(e),
                },
            )
            return None, "unknown", None
        self._lg.debug("LLM request completed", extra={"after": appinfra.time.since(start_t)})

        if result.parsed is None:
            self._lg.warning("LLM failed to generate structured joke")
            return None, result.model, result.adapter

        return result.parsed, result.model, result.adapter

    def _validate(
        self, joke: Joke, model_name: str, adapter: AdapterInfo | None, attempt: int
    ) -> tuple[Joke, str, AdapterInfo | None, NoveltyCheck | None] | None:
        """Validate joke against denylist, completeness, and novelty checks."""
        if self._is_incomplete(joke.text):
            self._lg.debug(
                "joke incomplete (question without punchline)", extra={"attempt": attempt}
            )
            self._recent_failed.append((joke.text, None))
            return None

        if self._contains_denied(joke.text):
            self._lg.debug("joke denied by denylist", extra={"attempt": attempt})
            self._recent_failed.append((joke.text, None))
            return None

        novelty = self._novelty.check(joke.text)
        if not novelty.is_novel:
            self._recent_failed.append((joke.text, novelty.similar_joke))
            return None

        return joke, model_name, adapter, novelty

    def _is_incomplete(self, text: str) -> bool:
        """Check if joke is incomplete (question without punchline)."""
        text = text.strip()
        # Incomplete if ends with ? and has no content after the question.
        # Note: This heuristic has false positives for jokes where the question mark
        # IS the punchline (e.g., "She looked surprised?"). Accept this tradeoff
        # since incomplete jokes (setup only) are more common failure modes.
        if not text.endswith("?"):
            return False
        # Check if there's any text after the last question mark (punchline)
        last_q = text.rfind("?")
        after_q = text[last_q + 1 :].strip()
        return len(after_q) == 0

    def _contains_denied(self, text: str) -> bool:
        """Check if text contains denied words/phrases."""
        if not self._denylist:
            return False
        text_lower = text.lower()
        return any(term in text_lower for term in self._denylist)

    def _build_prompt(self, context: list[str], retry_feedback: str) -> str:
        """Build prompt for joke generation."""
        directive = self._directive.directive.prompt if self._directive else ""

        context_text = ""
        if context:
            context_text = "Recent jokes you've told:\n" + "\n".join(f"- {j}" for j in context)

        avoid_text = ""
        if self._recent_failed:
            recent = list(self._recent_failed)[-5:]
            avoid_text = self._format_avoid_section(recent)

        retry_text = f"\n\n{retry_feedback}" if retry_feedback else ""

        return f"""{directive}

{context_text}{avoid_text}{retry_text}

Return your joke in JSON format with 'text' and 'style' fields."""

    def _format_avoid_section(self, recent: list[tuple[str, str | None]]) -> str:
        """Format the avoid section with detailed similarity info."""
        lines = [
            "\n\nDO NOT generate jokes similar to these - they are too close to existing jokes:"
        ]
        for generated, existing in recent:
            if existing:
                lines.append(f'- You tried: "{generated}"')
                lines.append(f'  Too similar to existing: "{existing}"')
            else:
                lines.append(f'- Rejected: "{generated}"')
        lines.append(
            "\nGenerate something with a COMPLETELY different topic, setup, and punchline."
        )
        return "\n".join(lines)
