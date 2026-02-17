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

    def test_clean_strips_code_fences_with_preamble(self) -> None:
        """Code fences with preamble text are handled."""
        cleaner = JSONCleaner()
        content = """Looking at my previous jokes, I see the wordplay ones got good reactions.

```json
{"text": "joke here", "style": "pun"}
```"""
        result = cleaner.clean(content)
        assert result == '{"text": "joke here", "style": "pun"}'

    def test_clean_strips_code_fences_with_long_preamble(self) -> None:
        """Code fences with multi-line preamble are handled."""
        cleaner = JSONCleaner()
        content = """I'll analyze the patterns here.
The puns work well.
Let me try something new.

```json
{"text": "test", "style": "observational"}
```"""
        result = cleaner.clean(content)
        assert result == '{"text": "test", "style": "observational"}'


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

    def test_skip_wrapped_schema(self) -> None:
        """Schema wrapped in {"schema": ...} is detected and skipped."""
        cleaner = JSONCleaner()
        content = '{"schema": {"type": "object", "properties": {"text": {}}}}'
        result = cleaner.clean(content)
        # Wrapped schema is skipped, returns original (no valid data found)
        assert result == content

    def test_skip_wrapped_schema_keep_data(self) -> None:
        """Wrapped schema followed by data - schema skipped, data kept."""
        cleaner = JSONCleaner()
        content = '{"schema": {"type": "object", "properties": {}}}, {"text": "hello"}'
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


class TestCleanParsed:
    """Test post-parse cleaning for nested structure unwrapping."""

    def test_clean_parsed_already_correct(self) -> None:
        """Data with expected fields passes through unchanged."""
        cleaner = JSONCleaner()
        data = {"text": "hello", "style": "greeting"}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"text": "hello", "style": "greeting"}

    def test_clean_parsed_single_wrapper_key(self) -> None:
        """Single wrapper key containing expected fields is unwrapped.

        e.g., {"joke": {"text": "...", "style": "..."}} -> {"text": "...", "style": "..."}
        """
        cleaner = JSONCleaner()
        data = {"joke": {"text": "hello", "style": "pun"}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_field_contains_nested(self) -> None:
        """Expected field containing nested object with expected fields is unwrapped.

        e.g., {"text": {"text": "...", "style": "..."}} -> {"text": "...", "style": "..."}
        """
        cleaner = JSONCleaner()
        data = {"text": {"text": "hello", "style": "pun"}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_no_expected_fields_single_wrapper(self) -> None:
        """Without expected fields, single wrapper key is still unwrapped."""
        cleaner = JSONCleaner()
        data = {"wrapper": {"text": "hello", "style": "pun"}}
        result = cleaner.clean_parsed(data, None)
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_no_expected_fields_multiple_keys(self) -> None:
        """Without expected fields, multiple keys means no unwrapping."""
        cleaner = JSONCleaner()
        data = {"text": "hello", "style": "pun"}
        result = cleaner.clean_parsed(data, None)
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_schema_not_unwrapped(self) -> None:
        """Schema-like structure is not unwrapped without expected fields."""
        cleaner = JSONCleaner()
        data = {"schema": {"type": "object", "properties": {"text": {}}}}
        result = cleaner.clean_parsed(data, None)
        # Inner is a schema, so don't unwrap
        assert result == {"schema": {"type": "object", "properties": {"text": {}}}}

    def test_clean_parsed_non_dict_passthrough(self) -> None:
        """Non-dict data passes through unchanged."""
        cleaner = JSONCleaner()
        result = cleaner.clean_parsed(["a", "b"], {"text", "style"})  # type: ignore
        assert result == ["a", "b"]

    def test_clean_parsed_partial_match_unwraps_for_pydantic(self) -> None:
        """Single-key wrapper is unwrapped even if expected fields are partial.

        The cleaner aggressively unwraps single-key wrappers (unless schema).
        If inner is missing expected fields, Pydantic will catch it with a
        clear error rather than us returning a confusing nested structure.
        """
        cleaner = JSONCleaner()
        data = {"wrapper": {"text": "hello"}}  # Missing "style"
        result = cleaner.clean_parsed(data, {"text", "style"})
        # Unwraps to let Pydantic validate - will fail on missing "style"
        assert result == {"text": "hello"}

    def test_clean_parsed_expected_field_wraps_aliased_content(self) -> None:
        """Expected field containing aliased fields is unwrapped for Pydantic aliases.

        e.g., {"text": {"joke": "...", "style": "..."}} -> {"joke": "...", "style": "..."}
        This lets Pydantic's AliasChoices handle the field name mapping.
        """
        cleaner = JSONCleaner()
        data = {"text": {"joke": "hello", "style": "pun"}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"joke": "hello", "style": "pun"}

    def test_clean_parsed_double_nested_unwrap(self) -> None:
        """Double-nested structure is fully unwrapped.

        Handles confused LLM output like:
        {"text": {"joke": {"text": "...", "style": "..."}}}
        """
        cleaner = JSONCleaner()
        data = {"text": {"joke": {"text": "hello", "style": "pun"}}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_triple_nested_unwrap(self) -> None:
        """Triple-nested structure is fully unwrapped."""
        cleaner = JSONCleaner()
        data = {"response": {"data": {"joke": {"text": "hello", "style": "pun"}}}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_alias_wrapper(self) -> None:
        """Alias wrapper key (not in expected fields) is unwrapped.

        e.g., {"joke": {"text": "...", "style": "..."}} -> {"text": "...", "style": "..."}
        where "joke" is a Pydantic alias for "text" but not in expected fields.
        """
        cleaner = JSONCleaner()
        data = {"joke": {"text": "hello", "style": "pun"}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"text": "hello", "style": "pun"}

    def test_clean_parsed_alias_wrapper_with_aliased_content(self) -> None:
        """Alias wrapper with aliased inner content is unwrapped.

        e.g., {"joke": {"joke": "...", "style": "..."}} -> {"joke": "...", "style": "..."}
        Both wrapper and inner use alias - let Pydantic handle resolution.
        """
        cleaner = JSONCleaner()
        data = {"joke": {"joke": "hello", "style": "pun"}}
        result = cleaner.clean_parsed(data, {"text", "style"})
        assert result == {"joke": "hello", "style": "pun"}
