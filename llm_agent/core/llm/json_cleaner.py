"""JSON cleanup utilities for LLM outputs.

LLMs often return JSON wrapped in markdown or with extra formatting.
This module provides utilities to clean up these common patterns.
"""

from __future__ import annotations

import json


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

        Only applies the fix if JSON is actually invalid - prevents corrupting valid JSON
        that contains braces/brackets within string values (e.g., {"text": "I {love} this"}).
        """
        if not content:
            return content

        # First check if JSON is already valid - don't touch it if so
        try:
            json.loads(content)
            return content  # Valid JSON, no fix needed
        except (json.JSONDecodeError, ValueError):
            pass  # JSON is invalid, proceed with auto-close attempt

        # Scan for unclosed brackets/braces, then append closers in correct order
        unclosed_stack = self._scan_json_structure(content)
        return content + "".join(reversed(unclosed_stack))

    def _scan_json_structure(self, content: str) -> list[str]:
        """Scan JSON content and return stack of unclosed brackets/braces.

        Uses string-aware scanning to ignore braces inside string literals.
        """
        stack: list[str] = []
        in_string = False
        escape = False

        for char in content:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char in ("}", "]") and stack and stack[-1] == char:
                    stack.pop()

        return stack
