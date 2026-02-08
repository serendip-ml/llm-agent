"""Tests for traits."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llm_agent import (
    BaseTrait,
    Identity,
    IdentityTrait,
    MethodTrait,
    StructuredOutputError,
)
from llm_agent.core.llm.types import Message
from llm_agent.core.traits.llm import LLMTrait


pytestmark = pytest.mark.unit


class TestBaseTrait:
    """Tests for BaseTrait."""

    def test_init(self):
        trait = BaseTrait()

        assert trait._agent is None

    def test_agent_property_raises_if_not_attached(self):
        trait = BaseTrait()

        with pytest.raises(RuntimeError, match="not attached to agent"):
            _ = trait.agent

    def test_attach(self):
        trait = BaseTrait()
        mock_agent = MagicMock()

        trait.attach(mock_agent)

        assert trait._agent == mock_agent

    def test_agent_property_returns_agent_after_attach(self):
        trait = BaseTrait()
        mock_agent = MagicMock()
        trait.attach(mock_agent)

        assert trait.agent == mock_agent

    def test_lifecycle_methods_exist(self):
        trait = BaseTrait()

        # Should not raise - these are no-op by default
        trait.on_start()
        trait.on_stop()


class TestCustomTrait:
    """Tests for custom traits using BaseTrait."""

    def test_custom_trait_inherits_behavior(self):
        class MyTrait(BaseTrait):
            def __init__(self, value: str) -> None:
                super().__init__()
                self.value = value

            def get_value(self) -> str:
                return f"{self.agent.name}: {self.value}"

        trait = MyTrait("test")
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        trait.attach(mock_agent)

        assert trait.get_value() == "test-agent: test"


class TestIdentity:
    """Tests for Identity."""

    def test_identity_with_prompt(self):
        identity = Identity(prompt="You are a code reviewer.")

        assert identity.prompt == "You are a code reviewer."
        assert identity.extensions == {}

    def test_identity_with_extensions(self):
        identity = Identity(
            prompt="You are a code reviewer.",
            extensions={"custom_field": "value"},
        )

        assert identity.prompt == "You are a code reviewer."
        assert identity.extensions == {"custom_field": "value"}


class TestIdentityTrait:
    """Tests for IdentityTrait."""

    def test_init_with_identity_object(self):
        identity = Identity(prompt="Test identity")
        trait = IdentityTrait(identity)

        assert trait.identity == identity

    def test_init_with_string(self):
        """IdentityTrait can be initialized with a string."""
        trait = IdentityTrait("You are a code reviewer.")

        assert trait.identity.prompt == "You are a code reviewer."

    def test_attach(self):
        identity = Identity(prompt="Test identity")
        trait = IdentityTrait(identity)
        mock_agent = MagicMock()

        trait.attach(mock_agent)

        assert trait._agent == mock_agent

    def test_build_prompt(self):
        identity = Identity(prompt="You are a code reviewer. Be critical.")
        trait = IdentityTrait(identity)

        result = trait.build_prompt("Base system prompt.")

        # Identity is prepended
        assert result.startswith("You are a code reviewer.")
        assert "Base system prompt." in result


class TestMethodTrait:
    """Tests for MethodTrait."""

    def test_init(self):
        trait = MethodTrait("- Step 1\n- Step 2")

        assert trait.method == "- Step 1\n- Step 2"

    def test_build_prompt(self):
        trait = MethodTrait("- Step 1\n- Step 2")

        result = trait.build_prompt("Base prompt.")

        assert "Base prompt." in result
        assert "## Method" in result
        assert "- Step 1\n- Step 2" in result

    def test_update(self):
        trait = MethodTrait("Original method")

        trait.update("Updated method")

        assert trait.method == "Updated method"


class TestAgentTraits:
    """Tests for Agent trait support."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create a test agent."""
        from llm_agent.agents.default import Agent as DefaultAgent
        from llm_agent.core.agent import Identity

        identity = Identity.from_name("test")
        return DefaultAgent(lg=mock_logger, identity=identity, default_prompt="")

    def test_add_trait(self, agent):
        identity = Identity(prompt="Test identity")
        trait = IdentityTrait(identity)
        agent.add_trait(trait)

        assert agent.has_trait(IdentityTrait)
        assert agent.get_trait(IdentityTrait) == trait

    def test_add_trait_attaches(self, agent):
        identity = Identity(prompt="Test identity")
        trait = IdentityTrait(identity)
        agent.add_trait(trait)

        assert trait._agent == agent

    def test_add_duplicate_trait_raises(self, agent):
        identity = Identity(prompt="Test identity")
        agent.add_trait(IdentityTrait(identity))

        with pytest.raises(ValueError, match="already added"):
            agent.add_trait(IdentityTrait(identity))

    def test_get_trait_returns_none_if_not_added(self, agent):
        assert agent.get_trait(IdentityTrait) is None

    def test_has_trait_returns_false_if_not_added(self, agent):
        assert not agent.has_trait(IdentityTrait)


class TestLLMTraitStructuredOutput:
    """Tests for LLMTrait structured output support."""

    class Answer(BaseModel):
        """Test schema for structured output."""

        answer: str
        confidence: float

    @pytest.fixture
    def trait(self):
        """Create LLMTrait with mocked client."""
        trait = LLMTrait(MagicMock(), {})
        trait._client = MagicMock()
        return trait

    def test_structured_output_basic(self, trait):
        """Test successful structured output parsing."""
        # Mock response with valid JSON
        mock_response = MagicMock()
        mock_response.content = '{"answer": "42", "confidence": 0.95}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="What is the meaning of life?")]
        result = trait.complete(messages, output_schema=self.Answer)

        assert result.parsed is not None
        assert isinstance(result.parsed, self.Answer)
        assert result.parsed.answer == "42"
        assert result.parsed.confidence == 0.95

    def test_structured_output_injects_schema_prompt(self, trait):
        """Test that schema prompt is injected into system message."""
        mock_response = MagicMock()
        mock_response.content = '{"answer": "test", "confidence": 1.0}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Question?"),
        ]
        trait.complete(messages, output_schema=self.Answer)

        # Check the messages passed to chat_full
        call_args = trait._client.chat_full.call_args
        sent_messages = call_args.kwargs["messages"]

        # Schema prompt should be appended to system message
        assert "You are helpful." in sent_messages[0]["content"]
        assert "json" in sent_messages[0]["content"].lower()
        assert "answer" in sent_messages[0]["content"]

    def test_structured_output_creates_system_message_if_none(self, trait):
        """Test that schema prompt creates system message if none exists."""
        mock_response = MagicMock()
        mock_response.content = '{"answer": "test", "confidence": 1.0}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="Question?")]
        trait.complete(messages, output_schema=self.Answer)

        call_args = trait._client.chat_full.call_args
        sent_messages = call_args.kwargs["messages"]

        # First message should be a system message with schema
        assert sent_messages[0]["role"] == "system"
        assert "json" in sent_messages[0]["content"].lower()

    def test_structured_output_enables_json_mode(self, trait):
        """Test that JSON mode is enabled via extra_body."""
        mock_response = MagicMock()
        mock_response.content = '{"answer": "test", "confidence": 1.0}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="Question?")]
        trait.complete(messages, output_schema=self.Answer)

        call_args = trait._client.chat_full.call_args
        extra_body = call_args.kwargs.get("extra_body")

        assert extra_body is not None
        assert extra_body == {"response_format": {"type": "json_object"}}

    def test_structured_output_invalid_json_raises(self, trait):
        """Test that invalid JSON raises StructuredOutputError."""
        mock_response = MagicMock()
        mock_response.content = "not valid json {"
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="Question?")]

        with pytest.raises(StructuredOutputError, match="Invalid JSON"):
            trait.complete(messages, output_schema=self.Answer)

    def test_structured_output_schema_mismatch_raises(self, trait):
        """Test that schema mismatch raises StructuredOutputError."""
        mock_response = MagicMock()
        # Missing required 'confidence' field
        mock_response.content = '{"answer": "test"}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="Question?")]

        with pytest.raises(StructuredOutputError, match="doesn't match schema"):
            trait.complete(messages, output_schema=self.Answer)

    def test_structured_output_wrong_type_raises(self, trait):
        """Test that wrong field type raises StructuredOutputError."""
        mock_response = MagicMock()
        # confidence should be float, not string
        mock_response.content = '{"answer": "test", "confidence": "high"}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="Question?")]

        with pytest.raises(StructuredOutputError, match="doesn't match schema"):
            trait.complete(messages, output_schema=self.Answer)

    def test_no_schema_parsed_is_none(self, trait):
        """Test backward compatibility - no schema means parsed is None."""
        mock_response = MagicMock()
        mock_response.content = "Just plain text response"
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        trait._client.chat_full.return_value = mock_response

        messages = [Message(role="user", content="Question?")]
        result = trait.complete(messages)

        assert result.parsed is None
        assert result.content == "Just plain text response"

    def test_tools_and_schema_raises(self, trait):
        """Test that using both tools and output_schema raises ValueError."""
        messages = [Message(role="user", content="Question?")]
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]

        with pytest.raises(ValueError, match="Cannot use both tools and output_schema"):
            trait.complete(messages, tools=tools, output_schema=self.Answer)
