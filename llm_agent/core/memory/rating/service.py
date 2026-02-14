"""Generic rating service for LLM-based content evaluation.

This service is backend-agnostic and only handles LLM interaction.
Use with a backend (e.g., AtomicFactsRatingBackend) for persistence.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from appinfra.log import Logger

from llm_agent.core.llm.json_cleaner import JSONCleaner

from .models import BatchItem, BatchRequest, Criteria, ProviderType, Request, Result


if TYPE_CHECKING:
    from llm_infer.client import LLMClient, LLMRouter


def stars_to_signal(stars: int) -> tuple[Literal["positive", "negative", "dismiss"], float]:
    """Convert star rating to signal and strength.

    Args:
        stars: Star rating (1-5).

    Returns:
        Tuple of (signal, strength).
    """
    mapping: dict[int, tuple[Literal["positive", "negative", "dismiss"], float]] = {
        1: ("negative", 1.0),
        2: ("negative", 0.5),
        3: ("dismiss", 1.0),
        4: ("positive", 0.5),
        5: ("positive", 1.0),
    }
    return mapping.get(stars, ("dismiss", 1.0))


class Service:
    """Generic service for rating content using LLM.

    Backend-agnostic - handles prompt building, LLM calls, and response parsing.
    Does NOT handle persistence - use a backend for saving ratings.

    Example:
        from llm_agent.core.memory.rating import Service, AtomicFactsBackend
        from llm_agent.core.memory.rating import Request, Criteria

        service = Service(lg, llm_client)
        backend = AtomicFactsBackend(lg, database)

        # Rate content with LLM
        request = Request(
            fact=123,  # For atomic facts, this is the fact_id
            content="Why did the chicken cross the road?",
            prompt_template="You are rating a joke...",
            criteria=[Criteria(name="humor", description="Is it funny?", weight=1.0)],
            model="auto",
            provider="llm_primary",
        )
        result = service.rate_content(request)

        # Save using backend
        backend.save_rating(result, source="llm_rater")
    """

    def __init__(self, lg: Logger, llm_client: LLMClient | LLMRouter) -> None:
        """Initialize rating service.

        Args:
            lg: Logger instance.
            llm_client: LLM client or router for rating.
        """
        self._lg = lg
        self._llm_client = llm_client

    def rate_content(self, request: Request) -> Result:
        """Rate content using LLM.

        Args:
            request: Rating request with content and parameters.

        Returns:
            Rating result with scores and reasoning.
        """
        prompt = self._build_prompt(request.content, request.prompt_template, request.criteria)
        response = self._call_llm(prompt, request.model, request.temperature)
        actual_model = response.model or request.model
        return self._parse_response(
            request.fact, response.content, request.criteria, actual_model, request.provider
        )

    def _build_prompt(
        self,
        content: str,
        prompt_template: str,
        criteria: list[Criteria],
    ) -> str:
        """Build rating prompt from template and criteria."""
        criteria_desc = "\n".join(
            f"- {c.name}: {c.description} (weight: {c.weight})" for c in criteria
        )

        return f"""{prompt_template}

Criteria:
{criteria_desc}

Content to rate:
{content}

Provide your rating in JSON format:
{{
  "stars": <1-5>,
  "criteria_scores": {{
    "criterion_name": <0.0-1.0>,
    ...
  }},
  "reasoning": "<brief explanation>"
}}

Rating scale:
- 5 stars (★★★★★): Excellent - exceeds expectations on all criteria
- 4 stars (★★★★☆): Good - meets expectations well
- 3 stars (★★★☆☆): Neutral - acceptable but unremarkable
- 2 stars (★★☆☆☆): Below expectations - has issues
- 1 star (★☆☆☆☆): Poor - fails to meet criteria

Respond only with the JSON, no additional text."""

    def _call_llm(self, prompt: str, model: str, temperature: float) -> Any:
        """Call LLM with rating prompt."""
        return self._llm_client.chat_full(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
        )

    def _parse_response(
        self,
        fact: Any,
        response: str,
        criteria: list[Criteria],
        model: str,
        provider: str,
    ) -> Result:
        """Parse LLM response into Result."""
        data = self._extract_json(response, criteria)
        # Clamp stars to valid 1-5 range
        raw_stars = data.get("stars", 3)
        stars = max(1, min(5, int(raw_stars)))
        signal, strength = self._stars_to_signal(stars)

        return Result(
            fact=fact,
            signal=signal,
            strength=strength,
            stars=stars,
            criteria_scores=data.get("criteria_scores", {}),
            reasoning=data.get("reasoning", ""),
            provider_type=ProviderType.LLM,
            model=model,
            provider=provider,
        )

    def _extract_json(self, response: str, criteria: list[Criteria]) -> dict[str, Any]:
        """Extract and parse JSON from LLM response.

        Uses JSONCleaner to handle code fences, auto-close braces,
        and extract only the first JSON object if multiple are returned.
        """
        try:
            cleaner = JSONCleaner()
            json_str = cleaner.extract_first_object(response)
            return cast(dict[str, Any], json.loads(json_str))
        except json.JSONDecodeError as e:
            self._lg.warning(
                "failed to parse rating JSON",
                extra={"exception": e, "response": response},
            )
            return self._fallback_rating(criteria)

    def _fallback_rating(self, criteria: list[Criteria]) -> dict[str, Any]:
        """Return neutral rating when parsing fails."""
        return {
            "stars": 3,
            "criteria_scores": {c.name: 0.5 for c in criteria},
            "reasoning": "Failed to parse rating response",
        }

    def _stars_to_signal(
        self, stars: int
    ) -> tuple[Literal["positive", "negative", "dismiss"], float]:
        """Convert star rating to signal and strength."""
        return stars_to_signal(stars)

    # =========================================================================
    # Batch rating
    # =========================================================================

    def rate_batch(self, request: BatchRequest) -> list[Result]:
        """Rate multiple items in a single LLM call.

        Batches items together to reduce API costs by sharing prompt overhead.

        Args:
            request: Batch rating request with items and parameters.

        Returns:
            List of rating results (may be fewer than items if parsing fails).
        """
        if not request.items:
            return []

        prompt = self._build_batch_prompt(request.items, request.prompt_template)
        response = self._call_llm(prompt, request.model, request.temperature)
        actual_model = response.model or request.model

        return self._parse_batch_response(
            request.items, response.content, actual_model, request.provider
        )

    def _build_batch_prompt(self, items: list[BatchItem], prompt_template: str) -> str:
        """Build batch rating prompt."""
        items_text = "\n\n".join(
            f"ITEM {i + 1} (ID: {item.fact}):\n{item.content}" for i, item in enumerate(items)
        )

        return f"""{prompt_template}

{items_text}

Rate each item 1-5 stars. Respond with a JSON array:
```json
[
  {{"id": <item_id>, "stars": <1-5>, "reasoning": "<brief explanation>"}},
  ...
]
```

Be strict. Respond only with the JSON array."""

    def _parse_batch_response(
        self,
        items: list[BatchItem],
        response: str,
        model: str,
        provider: str,
    ) -> list[Result]:
        """Parse batch response into individual results."""
        results = []
        item_map = {item.fact: item for item in items}

        try:
            parsed = self._extract_batch_json(response)
            for rating_data in parsed:
                result = self._parse_batch_item(rating_data, item_map, model, provider)
                if result:
                    results.append(result)
        except Exception as e:
            self._lg.warning(
                "failed to parse batch response",
                extra={"exception": e, "response": response[:500]},
            )

        return results

    def _extract_batch_json(self, response: str) -> list[dict[str, Any]]:
        """Extract JSON array from batch response."""
        # Handle markdown code fences
        json_str = response
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()

        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            # Single item wrapped in object
            return [parsed]
        return cast(list[dict[str, Any]], parsed)

    def _parse_batch_item(
        self,
        data: dict[str, Any],
        item_map: dict[Any, BatchItem],
        model: str,
        provider: str,
    ) -> Result | None:
        """Parse a single item from batch response."""
        item_id = data.get("id")
        stars_raw = data.get("stars")
        reasoning = data.get("reasoning", "")

        if item_id not in item_map:
            self._lg.warning("unknown item ID in batch response", extra={"id": item_id})
            return None

        if not isinstance(stars_raw, int) or not 1 <= stars_raw <= 5:
            self._lg.warning("invalid stars in batch response", extra={"stars": stars_raw})
            return None

        stars = max(1, min(5, int(stars_raw)))
        signal, strength = self._stars_to_signal(stars)

        return Result(
            fact=item_id,
            signal=signal,
            strength=strength,
            stars=stars,
            criteria_scores={},
            reasoning=str(reasoning),
            provider_type=ProviderType.LLM,
            model=model,
            provider=provider,
        )
