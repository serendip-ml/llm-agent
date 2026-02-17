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
        cleaned = self._basic_clean(content)

        # If multiple objects (e.g., LLM echoed schema), extract last one
        cleaned = self._extract_last_object(cleaned)

        return cleaned

    def _basic_clean(self, content: str) -> str:
        """Basic cleaning without multi-object handling."""
        cleaned = content.strip()
        cleaned = self._strip_code_fences(cleaned)
        cleaned = self._auto_close_json(cleaned)
        return cleaned

    def _strip_code_fences(self, content: str) -> str:
        """Extract JSON from markdown code fences.

        Handles patterns like:
            ```json
            {"key": "value"}
            ```

        Also handles preamble text before the code fence:
            Some thinking text...
            ```json
            {"key": "value"}
            ```
        """
        # Find code fence anywhere in content
        fence_start = content.find("```")
        if fence_start == -1:
            return content

        # Find end of opening fence line
        newline_after_fence = content.find("\n", fence_start)
        if newline_after_fence == -1:
            return content

        # Find closing fence
        fence_end = content.find("```", newline_after_fence)
        if fence_end == -1:
            # No closing fence - extract from opening fence to end
            return content[newline_after_fence + 1 :].strip()

        # Extract content between fences
        return content[newline_after_fence + 1 : fence_end].strip()

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
            if char == "\\" and in_string:
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

    def extract_first_object(self, content: str) -> str:
        """Extract only the first JSON object from content.

        LLMs sometimes return multiple JSON objects. This extracts just the first one.

        Args:
            content: String potentially containing multiple JSON objects.

        Returns:
            String containing only the first JSON object.
        """
        cleaned = self._basic_clean(content)
        if not cleaned:
            return cleaned

        # Use raw_decode to extract just the first JSON object
        try:
            decoder = json.JSONDecoder()
            _, end_idx = decoder.raw_decode(cleaned)
            return cleaned[:end_idx]
        except json.JSONDecodeError:
            # If parsing fails, return the cleaned content for downstream error handling
            return cleaned

    def _extract_last_object(self, content: str) -> str:
        """Extract only the last JSON object from content.

        Useful when LLM echoes schema definition before actual data.
        Skips objects that look like JSON schema definitions.
        """
        if not content:
            return content

        # Find all JSON objects using raw_decode
        decoder = json.JSONDecoder()
        idx = 0
        last_obj = content

        try:
            while idx < len(content):
                obj, end_idx = decoder.raw_decode(content[idx:])
                candidate = content[idx : idx + end_idx]
                # Skip JSON schema definitions
                if not self._looks_like_schema(obj):
                    last_obj = candidate
                idx += end_idx
                # Skip whitespace and commas between objects
                while idx < len(content) and content[idx] in " \t\n,":
                    idx += 1
        except json.JSONDecodeError:
            pass

        return last_obj

    def _looks_like_schema(self, obj: object) -> bool:
        """Check if parsed object looks like a JSON schema definition."""
        if not isinstance(obj, dict):
            return False
        return "properties" in obj and obj.get("type") == "object"

    def clean_parsed(
        self, data: dict[str, object], expected_fields: set[str] | None = None
    ) -> dict[str, object]:
        """Clean parsed JSON dict by unwrapping nested structures.

        LLMs sometimes nest the actual data inside wrapper keys. This method
        attempts to extract the real data when the structure doesn't match
        expected fields.

        Handles patterns:
        - Single wrapper key: {"joke": {"text": "...", "style": "..."}}
          → {"text": "...", "style": "..."}
        - Field containing nested object: {"text": {"text": "...", "style": "..."}}
          → {"text": "...", "style": "..."}

        Args:
            data: Parsed JSON dict from LLM.
            expected_fields: Set of field names we expect. If None, attempts
                heuristic unwrapping based on structure patterns.

        Returns:
            Cleaned dict, possibly unwrapped from nested structure.
        """
        if not isinstance(data, dict):
            return data

        # If expected fields provided, try targeted unwrapping
        if expected_fields:
            return self._unwrap_with_expected(data, expected_fields)

        # Heuristic: single-key dict with nested dict value → unwrap
        return self._unwrap_single_wrapper(data)

    def _unwrap_with_expected(
        self, data: dict[str, object], expected: set[str]
    ) -> dict[str, object]:
        """Unwrap nested structure using expected field names as guide.

        Tries to find and extract a nested dict that contains the expected fields.
        """
        # Already has expected fields → return as-is
        if expected.issubset(data.keys()):
            return data

        # Case 1: {"text": {"text": "...", "style": "..."}} - field contains nested obj
        for field in expected:
            if field in data and isinstance(data[field], dict):
                inner = data[field]
                if isinstance(inner, dict) and expected.issubset(inner.keys()):
                    return inner

        # Case 2: {"joke": {"text": "...", "style": "..."}} - single wrapper key
        if len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict) and expected.issubset(inner.keys()):
                return inner

        # Case 3: {"text": {"joke": "...", "style": "..."}} - expected field wraps aliased content
        # Unwrap if single-key and inner is a dict (let Pydantic aliases handle field names)
        if len(data) == 1:
            key = next(iter(data.keys()))
            inner = data[key]
            if key in expected and isinstance(inner, dict) and len(inner) >= len(expected):
                return inner

        return data

    def _unwrap_single_wrapper(self, data: dict[str, object]) -> dict[str, object]:
        """Heuristically unwrap single-key wrapper dicts.

        If dict has exactly one key and the value is a dict, unwrap it.
        Skip if the inner dict looks like a schema definition.
        """
        if len(data) != 1:
            return data

        inner = next(iter(data.values()))
        if not isinstance(inner, dict):
            return data

        # Don't unwrap schema definitions
        if self._looks_like_schema(inner):
            return data

        return inner
