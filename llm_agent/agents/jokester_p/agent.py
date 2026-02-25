"""Programmatic joke-teller agent with guaranteed novelty checking.

This agent demonstrates the difference between prompt-based and programmatic agents:
- Prompt-based: Relies on LLM to follow instructions (probabilistic)
- Programmatic: Code enforces constraints (deterministic)

For joke-telling, the key constraint is "never repeat" across thousands of jokes,
which cannot be reliably enforced through prompts alone due to context window limits.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger
from appinfra.time import since
from llm_infer.client.exceptions import BackendUnavailableError

from ...core.agent import Agent, ExecutionResult
from ...core.llm.backend import StructuredOutputError
from ...core.traits.builtin.learn import LearnTrait
from .generate import GenerationAttempt, JokeGenerator
from .rating import BatchRater
from .storage import Storage


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
        self._jokes_generated_this_session = 0
        self._recent_results: deque[ExecutionResult] = deque(maxlen=100)
        self._storage: Storage | None = None
        self._generator: JokeGenerator | None = None
        self._rater: BatchRater | None = None
        self._last_joke_time: float = time.monotonic()

    def start(self) -> None:
        """Start agent and traits."""
        from .factory import Factory

        self._start_traits()

        try:
            components = Factory.create_components(self)
            self._generator = components.generator
            self._storage = components.storage
            self._rater = components.rater
        except Exception:
            self._stop_traits()
            raise

        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop agent and traits."""
        if not self._started:
            return
        self._stop_traits()
        self._storage = None
        self._generator = None
        self._rater = None
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
        """Execute the core joke generation and novelty checking logic."""
        assert self._generator is not None
        learn_trait = self.require_trait(LearnTrait)
        recent_jokes = self._get_recent_jokes(learn_trait, limit=5)

        attempt = self._generator.generate(recent_jokes)

        if not attempt.success:
            return ExecutionResult(
                success=False,
                content=f"Failed to generate novel joke after {attempt.cumulative_attempts} attempts",
                iterations=attempt.cumulative_attempts,
            )

        return self._complete_cycle(attempt)

    def _complete_cycle(self, attempt: GenerationAttempt) -> ExecutionResult:
        """Complete joke generation cycle with save and logging."""
        assert attempt.joke is not None
        assert self._storage is not None

        fact_id = self._storage.save_joke(
            joke=attempt.joke,
            model_name=attempt.model_name,
            attempts=attempt.cumulative_attempts,
            adapter=attempt.adapter,
        )
        self._jokes_generated_this_session += 1

        if self._rater:
            self._rater.queue(fact_id, attempt.joke.text)

        result = ExecutionResult(
            success=True,
            content=f"{attempt.joke.text}\n(Style: {attempt.joke.style})",
            iterations=attempt.cumulative_attempts,
        )
        self._recent_results.append(result)

        self._lg.info("found new joke", extra=self._build_joke_log_extra(attempt, fact_id))
        self._last_joke_time = time.monotonic()
        if attempt.similar_joke:
            self._lg.debug("closest existing joke", extra={"joke": attempt.similar_joke})
        return result

    def _build_joke_log_extra(self, attempt: GenerationAttempt, fact_id: int) -> dict[str, Any]:
        """Build log extra dict for joke generation."""
        assert attempt.joke is not None
        return {
            "after": since(self._last_joke_time),
            "fact": fact_id,
            "agent": self.name,
            "joke": attempt.joke.text,
            "chars": len(attempt.joke.text),
            "model": attempt.model_name,
            "style": attempt.joke.style,
            "closest": round(attempt.max_similarity, 2),
            "attempts": {"run": attempt.run_attempts, "cumulative": attempt.cumulative_attempts},
            "sess_count": self._jokes_generated_this_session,
        }

    def ask(self, question: str) -> str:
        """Answer question (not supported)."""
        return "Jokester agent does not support questions. Use run_once() to get a joke."

    def _get_recent_jokes(self, learn_trait: LearnTrait, limit: int) -> list[str]:
        """Fetch recent jokes chronologically for style inspiration."""
        try:
            facts = learn_trait.kelt.atomic.solutions.list_by_category("joke", limit=limit)
            return [f.content for f in facts if f.content]
        except Exception as e:
            self._lg.debug("failed to fetch recent jokes", extra={"exception": e})
            return []

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
