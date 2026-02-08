"""Agent identity - wraps llm-learn ProfileIdentity with convenient constructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm_learn.core.identity import IdentityResolver, ProfileIdentity


@dataclass(frozen=True)
class Identity(ProfileIdentity):  # type: ignore[misc]
    """Agent identity with convenient constructors.

    Wraps ProfileIdentity with agent-friendly API and room for future
    agent-specific enrichments (version, capabilities, etc.).

    Examples:
        # Simple name
        identity = Identity.from_name("joke-teller")

        # With workspace
        identity = Identity.from_name("joke-teller", workspace="production")

        # From YAML config
        config = {"domain": "acme", "workspace": "prod", "name": "reviewer"}
        identity = Identity.from_config(config)

        # With defaults (for factory)
        identity = Identity.from_config(profile_config, defaults={"name": agent_name})
    """

    @classmethod
    def from_name(
        cls, name: str, workspace: str = "default", domain: str | None = None
    ) -> Identity:
        """Create identity from simple name.

        Args:
            name: Agent name (unique within workspace).
            workspace: Workspace name (default: "default").
            domain: Optional domain name.

        Returns:
            Identity instance with resolved IDs.

        Example:
            identity = Identity.from_name("joke-teller")
        """
        config = {"name": name, "workspace": workspace}
        if domain is not None:
            config["domain"] = domain

        identity = IdentityResolver.resolve(config)
        return cls(**asdict(identity))

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | None = None, defaults: dict[str, Any] | None = None
    ) -> Identity:
        """Create identity from configuration dict.

        Args:
            config: Configuration dict with name/workspace/domain/IDs.
            defaults: Default values if not in config.

        Returns:
            Identity instance with resolved IDs.

        Example:
            config = {"name": "reviewer", "workspace": "production"}
            identity = Identity.from_config(config)
        """
        identity = IdentityResolver.resolve(config, defaults)
        return cls(**asdict(identity))
