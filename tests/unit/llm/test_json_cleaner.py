"""Tests for JSON cleaner utilities."""

from llm_agent.core.llm.json_cleaner import JSONCleaner


class TestJSONCleanerBasic:
    """Test basic cleaning functionality."""

    def test_clean_plain_json(self) -> None:
        """Clean JSON without formatting passes through."""
        cleaner = JSONCleaner()
        result = cleaner.clean('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_clean_strips_whitespace(self) -> None:
        """Whitespace is stripped."""
        cleaner = JSONCleaner()
        result = cleaner.clean('  {"key": "value"}  \n')
        assert result == '{"key": "value"}'

    def test_clean_strips_code_fences(self) -> None:
        """Markdown code fences are removed."""
        cleaner = JSONCleaner()
        result = cleaner.clean('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_clean_strips_code_fences_no_language(self) -> None:
        """Code fences without language tag are removed."""
        cleaner = JSONCleaner()
        result = cleaner.clean('```\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'


class TestJSONCleanerAutoClose:
    """Test auto-closing of unclosed braces/brackets."""

    def test_auto_close_missing_brace(self) -> None:
        """Missing closing brace is added."""
        cleaner = JSONCleaner()
        result = cleaner.clean('{"key": "value"')
        assert result == '{"key": "value"}'

    def test_auto_close_missing_bracket(self) -> None:
        """Missing closing bracket is added."""
        cleaner = JSONCleaner()
        result = cleaner.clean('["a", "b"')
        assert result == '["a", "b"]'

    def test_auto_close_nested(self) -> None:
        """Nested unclosed structures are closed correctly."""
        cleaner = JSONCleaner()
        result = cleaner.clean('{"items": ["a", "b"')
        assert result == '{"items": ["a", "b"]}'

    def test_no_auto_close_valid_json(self) -> None:
        """Valid JSON is not modified."""
        cleaner = JSONCleaner()
        result = cleaner.clean('{"text": "I {love} this"}')
        assert result == '{"text": "I {love} this"}'


class TestJSONCleanerMultipleObjects:
    """Test handling of multiple JSON objects (schema echo issue)."""

    def test_extract_last_single_object(self) -> None:
        """Single object is returned as-is."""
        cleaner = JSONCleaner()
        result = cleaner.clean('{"text": "hello", "style": "greeting"}')
        assert result == '{"text": "hello", "style": "greeting"}'

    def test_extract_last_two_objects(self) -> None:
        """When two objects present, last one is extracted."""
        cleaner = JSONCleaner()
        content = '{"schema": true}, {"text": "hello", "style": "greeting"}'
        result = cleaner.clean(content)
        assert result == '{"text": "hello", "style": "greeting"}'

    def test_extract_last_schema_then_data(self) -> None:
        """Schema echo followed by data - extracts data."""
        cleaner = JSONCleaner()
        content = """```json
{
  "description": "Structured output",
  "properties": {"text": {"type": "string"}},
  "type": "object"
},
{
  "text": "actual data",
  "style": "test"
}
```"""
        result = cleaner.clean(content)
        # Should get the last object (the actual data)
        assert '"text": "actual data"' in result
        assert '"properties"' not in result

    def test_extract_last_with_whitespace(self) -> None:
        """Multiple objects with various whitespace."""
        cleaner = JSONCleaner()
        content = '{"first": 1}\n\n,\n{"second": 2}'
        result = cleaner.clean(content)
        assert result == '{"second": 2}'

    def test_skip_schema_only(self) -> None:
        """When only a schema is returned, skip it."""
        cleaner = JSONCleaner()
        content = '{"properties": {"text": {}}, "type": "object"}'
        result = cleaner.clean(content)
        # Schema is skipped, returns original (no valid data found)
        assert result == content

    def test_skip_schema_keep_data(self) -> None:
        """Schema followed by data - schema skipped, data kept."""
        cleaner = JSONCleaner()
        content = '{"properties": {}, "type": "object"}, {"text": "hello"}'
        result = cleaner.clean(content)
        assert result == '{"text": "hello"}'


class TestExtractFirstObject:
    """Test extract_first_object method."""

    def test_extract_first_single(self) -> None:
        """Single object returns itself."""
        cleaner = JSONCleaner()
        result = cleaner.extract_first_object('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_extract_first_multiple(self) -> None:
        """Multiple objects - first is extracted."""
        cleaner = JSONCleaner()
        result = cleaner.extract_first_object('{"first": 1}, {"second": 2}')
        assert result == '{"first": 1}'

    def test_extract_first_with_code_fences(self) -> None:
        """Code fences are stripped before extraction."""
        cleaner = JSONCleaner()
        result = cleaner.extract_first_object('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'
