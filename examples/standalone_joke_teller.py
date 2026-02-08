"""Standalone joke teller - using llm-agent without runtime.

This example demonstrates how to use llm-agent agents in your own Python scripts
without the multi-agent runtime infrastructure. You only need:
- llm_agent.core (platform resources)
- The agent you want to use (joke_teller)

No runtime, no HTTP server, no subprocess management - just agents.
"""

from typing import Any

from appinfra.log import LogConfig, Logger, LoggerFactory

from llm_agent.agents.joke_teller_p import Factory
from llm_agent.core import PlatformContext


def _setup_logging() -> Logger:
    """Configure and create root logger."""
    log_config = LogConfig.from_params(  # type: ignore[call-arg]
        level="info",
        handlers={
            "console": {
                "type": "console",
                "enabled": True,
                "level": "info",
                "format": "text",
                "stream": "stdout",
                "colors": True,
            }
        },
    )
    return LoggerFactory.create_root(log_config)


def _build_platform_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build LLM and learn configurations."""
    llm_config = {
        "default": "local",
        "backends": {
            "local": {
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "model": "default",
                "api_key": "not-needed",
            }
        },
    }

    learn_config = {
        "db": {
            "url": "postgresql://postgres:postgres@127.0.0.1:7632/learn",
            "extensions": ["vector"],
        },
        "embedder_url": "http://localhost:8001",
        "embedder_model": "default",
        "profile_config": {
            "domain": None,
            "workspace": "examples",
            "name": "standalone-joker",
        },
    }

    return llm_config, learn_config


def _build_agent_config() -> dict[str, Any]:
    """Build agent configuration."""
    return {
        "profile": {
            "domain": None,
            "workspace": "examples",
            "name": "standalone-joker",
        },
        "identity": """You are a joke teller who develops a personal sense of humor over time.

You learn what lands and what doesn't. You recall previous jokes and the
reactions they got, and you use that to refine your style. You never repeat
a joke. You have range — puns, one-liners, observational humor, absurdist
bits — but you gravitate toward whatever gets the best response.

You keep it short. A joke should fit in a text message.""",
        "config": {
            "max_retries": 3,
            "similarity_threshold": 0.85,
        },
    }


def _display_result(logger: Logger, result) -> None:
    """Display agent execution result."""
    if result.success:
        logger.info("agent completed successfully")
        print("\n" + "=" * 60)
        print("JOKE:")
        print(result.content)
        print("=" * 60 + "\n")
        print(f"Iterations: {result.iterations}")
    else:
        logger.error("agent failed", extra={"error": result.content})


def main() -> None:  # cq: max-lines=45
    """Run joke teller agent standalone."""
    # Setup logging
    logger = _setup_logging()
    logger.info("standalone joke teller starting")

    # Configure platform resources
    llm_config, learn_config = _build_platform_config()

    # Create platform context (central resource manager)
    platform = PlatformContext.from_config(
        lg=logger,
        llm_config=llm_config,
        learn_config=learn_config,
    )

    logger.info("platform context created", extra={"traits": platform.traits.count()})

    # Start platform traits (DB connections, etc.)
    platform.start_traits()

    try:
        # Create agent via factory
        factory = Factory(platform)
        agent = factory.create(_build_agent_config())

        logger.info("agent created", extra={"agent": agent.name})

        # Start and run agent
        agent.start()
        logger.info("running agent...")
        result = agent.run_once()

        # Display result
        _display_result(logger, result)

        # Stop agent
        agent.stop()

    finally:
        # Cleanup platform resources
        platform.cleanup()
        logger.info("standalone joke teller finished")


if __name__ == "__main__":
    main()
