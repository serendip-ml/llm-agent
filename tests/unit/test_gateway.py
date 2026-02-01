"""Tests for server configuration models.

Note: The AgentGateway class has been replaced by AgentRegistry.
See test_registry.py for registry tests.
This file tests configuration models that are still in use.
"""

import pytest

from llm_agent.core.traits.identity import Identity
from llm_agent.runtime.server.config import (
    AgentConfigYAML,
    AgentServerConfig,
    LearnBackendConfig,
    ScheduleConfigYAML,
    TaskConfigYAML,
)


pytestmark = pytest.mark.unit


class TestAgentServerConfig:
    """Tests for AgentServerConfig."""

    def test_agent_config_yaml(self):
        """Parse agent config from dict."""
        config = AgentConfigYAML(
            identity=Identity(prompt="Test prompt"),
            task=TaskConfigYAML(description="Do things"),
            tools={"shell": {"allowed_commands": ["ls"]}},
            schedule=ScheduleConfigYAML(interval=300),
        )
        assert config.class_ == "prompt"
        assert config.identity.prompt == "Test prompt"
        assert config.schedule.interval == 300

    def test_agent_server_config_from_dict(self):
        """Create AgentServerConfig from raw dict."""
        raw = {
            "server": {"port": 9000},
            "llm": {"base_url": "http://localhost:8000/v1"},
            "agents": {
                "test-agent": {
                    "class": "prompt",
                    "identity": {"prompt": "Be helpful"},
                    "task": {"description": "Help users"},
                }
            },
        }
        config = AgentServerConfig.from_dict(raw)

        assert config.server.port == 9000
        assert config.llm["base_url"] == "http://localhost:8000/v1"
        assert "test-agent" in config.agents
        assert config.agents["test-agent"].identity.prompt == "Be helpful"


class TestScheduleConfigYAML:
    """Tests for ScheduleConfigYAML."""

    def test_interval_required(self):
        """ScheduleConfigYAML requires interval."""
        config = ScheduleConfigYAML(interval=60)
        assert config.interval == 60


class TestLearnBackendConfig:
    """Tests for LearnBackendConfig."""

    def test_defaults(self):
        """LearnBackendConfig has defaults."""
        config = LearnBackendConfig(profile_id=1)
        assert config.profile_id == 1
        assert config.db_config_path == "etc/infra.yaml"
        assert config.db_key == "main"
        assert config.embedder_model == "default"

    def test_custom_values(self):
        """LearnBackendConfig accepts custom values."""
        config = LearnBackendConfig(
            profile_id=42,
            db_config_path="/custom/path.yaml",
            db_key="custom",
            embedder_url="http://embedder:8000",
            embedder_model="custom-model",
        )
        assert config.profile_id == 42
        assert config.db_config_path == "/custom/path.yaml"
        assert config.db_key == "custom"
        assert config.embedder_url == "http://embedder:8000"
        assert config.embedder_model == "custom-model"


class TestTaskConfigYAML:
    """Tests for TaskConfigYAML."""

    def test_description_required(self):
        """TaskConfigYAML requires description."""
        config = TaskConfigYAML(description="Test task")
        assert config.description == "Test task"
        assert config.output_schema is None

    def test_with_output_schema(self):
        """TaskConfigYAML accepts output_schema."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        config = TaskConfigYAML(description="Test task", output_schema=schema)
        assert config.output_schema == schema


class TestAgentConfigYAML:
    """Tests for AgentConfigYAML."""

    def test_default_class(self):
        """AgentConfigYAML defaults to 'prompt' class."""
        config = AgentConfigYAML(
            identity="Test prompt",
            task=TaskConfigYAML(description="Test"),
        )
        assert config.class_ == "prompt"

    def test_programmatic_class(self):
        """AgentConfigYAML accepts 'programmatic' class."""
        raw = {
            "class": "programmatic",
            "identity": "Test prompt",
            "task": {"description": "Test"},
        }
        config = AgentConfigYAML(**raw)
        assert config.class_ == "programmatic"

    def test_tools_default_empty(self):
        """AgentConfigYAML defaults tools to empty dict."""
        config = AgentConfigYAML(
            identity="Test prompt",
            task=TaskConfigYAML(description="Test"),
        )
        assert config.tools == {}

    def test_schedule_optional(self):
        """AgentConfigYAML schedule is optional."""
        config = AgentConfigYAML(
            identity="Test prompt",
            task=TaskConfigYAML(description="Test"),
        )
        assert config.schedule is None
