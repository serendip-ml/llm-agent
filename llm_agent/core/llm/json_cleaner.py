"""JSON cleanup utilities for LLM outputs.

LLMs often return JSON wrapped in markdown or with extra formatting.
This module provides utilities to clean up these common patterns.
"""

from __future__ import annotations


class JSONCleaner:
    """Cleans LLM-generated JSON strings for parsing.

    Handles common issues:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - (Future) Other common LLM formatting quirks

    Example:
        cleaner = JSONCleaner()
        raw = '```json\\n{"key": "value"}\\n```'
        cleaned = cleaner.clean(raw)  # '{"key": "value"}'
    """

    def clean(self, content: str) -> str:
        """Clean LLM JSON output for parsing.

        Args:
            content: Raw JSON string from LLM, possibly with markdown formatting.

        Returns:
            Cleaned JSON string ready for json.loads().
        """
        cleaned = content.strip()

        # Remove markdown code fences
        cleaned = self._strip_code_fences(cleaned)

        # Auto-close unclosed braces/brackets
        cleaned = self._auto_close_json(cleaned)

        return cleaned

    def _strip_code_fences(self, content: str) -> str:
        """Remove markdown code fences from JSON.

        Handles patterns like:
            ```json
            {"key": "value"}
            ```

        Or:
            ```
            {"key": "value"}
            ```
        """
        if not content.startswith("```"):
            return content

        lines = content.split("\n")

        # Remove opening fence (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def _auto_close_json(self, content: str) -> str:
        """Auto-close unclosed braces and brackets in JSON.

        Common LLM issue: incomplete JSON like {"key": "value" (missing closing brace).
        This adds missing closing characters based on the opening count.
        """
        if not content:
            return content

        # Count unclosed braces and brackets
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Add missing closing characters
        result = content
        if open_brackets > 0:
            result += "]" * open_brackets
        if open_braces > 0:
            result += "}" * open_braces

        return result
