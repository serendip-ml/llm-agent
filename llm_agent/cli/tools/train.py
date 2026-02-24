"""Train tool - create training manifests for agents."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal, Protocol

from appinfra import DotDict
from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_learn.training import Factory as TrainFactory


class AdapterNotFoundError(Exception):
    """Raised when a specified adapter cannot be found."""

    def __init__(self, md5: str) -> None:
        self.md5 = md5
        super().__init__(f"Adapter not found: {md5}")


class TrainingDataProvider(Protocol):
    """Protocol for agents to provide training data."""

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add agent-specific arguments to the parser."""
        ...

    def get_sft_examples(self, args: argparse.Namespace) -> list[dict[str, str]]:
        """Get SFT training examples.

        Returns:
            List of {"instruction": str, "output": str} dicts.
        """
        ...

    def get_dpo_pairs(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        """Get DPO preference pairs.

        Returns:
            List of {"chosen": str, "rejected": str, "prompt": str} dicts.
        """
        ...

    def get_context_key(self) -> str:
        """Get context key for this agent."""
        ...

    def get_description(self, method: str, count: int) -> str:
        """Get description for the training manifest."""
        ...


class TrainTool(Tool):
    """Create training manifests for agents."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="train", help_text="Create training manifests for agents")
        super().__init__(parent, config)
        self._pg: PG | None = None

    def configure(self) -> None:
        """Set up database connection."""
        if not hasattr(self.app, "config"):
            raise RuntimeError("App does not have config")

        learn_config = getattr(self.app.config, "learn", None)
        if learn_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        db_config = getattr(learn_config, "db", None)
        if db_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        self._pg = PG(self.lg, db_config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="method", help="Training method")

        # SFT subcommand
        sft_parser = subparsers.add_parser("sft", help="Supervised fine-tuning")
        self._add_common_args(sft_parser)
        self._add_agent_subcommand(sft_parser)

        # DPO subcommand
        dpo_parser = subparsers.add_parser("dpo", help="Direct preference optimization")
        self._add_common_args(dpo_parser)
        self._add_agent_subcommand(dpo_parser)

    def _add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common training arguments."""
        parser.add_argument(
            "--model",
            type=str,
            help="Base model for training (default: from llm-learn config)",
        )
        parser.add_argument(
            "--adapter",
            type=str,
            help="Name for the trained adapter (default: <agent>-<method>)",
        )
        parser.add_argument(
            "--from-adapter",
            type=str,
            help="Existing adapter (md5) to train on top of",
        )
        parser.add_argument(
            "--registry-path",
            type=str,
            help="Path to adapter registry (default: from config or ~/.llm-learn/adapters)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without saving",
        )

    def _add_agent_subcommand(self, parser: argparse.ArgumentParser) -> None:
        """Add 'agent' subcommand that takes agent name and args."""
        subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand")
        agent_parser = subparsers.add_parser("agent", help="Train a specific agent")
        agent_parser.add_argument(
            "agent_name",
            metavar="name",
            help="Agent name (e.g., jokester-p)",
        )
        # Remaining args will be parsed by agent's data provider
        agent_parser.add_argument(
            "agent_args",
            nargs=argparse.REMAINDER,
            help="Agent-specific arguments",
        )

    def run(self, **kwargs: Any) -> int:
        method = self._validate_method()
        if method is None:
            return 1

        provider = self._get_validated_provider(method)
        if provider is None:
            return 1

        try:
            if method == "sft":
                return self._run_sft(provider)
            elif method == "dpo":
                return self._run_dpo(provider)
            else:
                print(f"Unknown method: {method}")
                return 1
        except AdapterNotFoundError as e:
            print(f"Error: {e}")
            return 1

    def _validate_method(self) -> str | None:
        """Validate method and subcommand args."""
        method = self.args.method
        if method is None:
            print("Usage: train <sft|dpo> [options] agent <name> [agent-options]")
            return None

        subcommand = getattr(self.args, "subcommand", None)
        if subcommand != "agent":
            print(f"Usage: train {method} [options] agent <name> [agent-options]")
            return None

        return str(method)

    def _get_validated_provider(self, method: str) -> TrainingDataProvider | None:
        """Get provider and parse agent-specific args."""
        agent_name = self.args.agent_name
        provider = self._get_data_provider(agent_name)
        if provider is None:
            return None

        # Parse agent-specific args and merge into self.args
        agent_parser = argparse.ArgumentParser(prog=f"train {method} agent {agent_name}")
        provider.add_args(agent_parser)
        agent_args = agent_parser.parse_args(self.args.agent_args)
        for key, value in vars(agent_args).items():
            setattr(self.args, key, value)

        return provider

    def _get_data_provider(self, agent_name: str) -> TrainingDataProvider | None:
        """Get training data provider for an agent."""
        # Load agent config
        if not hasattr(self.app.config, "agents"):
            print("Error: No agents configured")
            return None

        agents = self.app.config.agents
        if agent_name not in agents:
            print(f"Error: Agent '{agent_name}' not found")
            print(f"Available agents: {', '.join(agents.keys())}")
            return None

        agent_config = agents[agent_name]

        # Get the agent's training data provider
        # For now, support jokester-p directly; later make this pluggable
        if agent_name == "jokester-p":
            return self._create_jokester_provider(agent_config)

        print(f"Error: Agent '{agent_name}' does not support training")
        return None

    def _create_jokester_provider(self, config: DotDict) -> TrainingDataProvider:
        """Create training data provider for jokester-p agent."""
        from llm_agent.agents.jokester_p.training import JokesterTrainingProvider

        assert self._pg is not None
        context_key = self._resolve_context_key(config)
        return JokesterTrainingProvider(self.lg, self._pg, context_key)

    def _resolve_context_key(self, config: DotDict) -> str:
        """Resolve context key from agent config."""
        if "identity" in config:
            identity = config["identity"]
            if isinstance(identity, dict) and "name" in identity:
                return str(identity["name"])
        return "unknown"

    def _run_sft(self, provider: TrainingDataProvider) -> int:
        """Run SFT training flow."""
        examples = provider.get_sft_examples(self.args)
        if not examples:
            print("\nNo examples available for SFT training.")
            return 0

        self._print_sft_summary(examples, provider)

        if self.args.dry_run:
            print("\nDry run complete. Use without --dry-run to submit manifest.")
            return 0

        factory = self._get_training_factory()
        return self._submit_manifest(factory, provider, "sft", examples)

    def _run_dpo(self, provider: TrainingDataProvider) -> int:
        """Run DPO training flow."""
        pairs = provider.get_dpo_pairs(self.args)
        if not pairs:
            print("\nNo pairs available for DPO training.")
            return 0

        self._print_dpo_summary(pairs, provider)

        if self.args.dry_run:
            print("\nDry run complete. Use without --dry-run to submit manifest.")
            return 0

        factory = self._get_training_factory()
        return self._submit_manifest(factory, provider, "dpo", pairs)

    def _get_training_factory(self) -> TrainFactory:
        """Get training factory with default profiles."""
        path = self._get_registry_path()
        profiles = self._get_default_profiles()
        return TrainFactory(self.lg, Path(path), profiles)

    def _get_registry_path(self) -> str:
        """Get adapter registry path from args, config, or default."""
        if self.args.registry_path:
            return str(self.args.registry_path)

        try:
            adapters_config = self.app.config.learn.get("adapters", {})
            lora_config = adapters_config.get("lora", {})
            if base_path := lora_config.get("base_path"):
                return os.path.expanduser(str(base_path))
        except (AttributeError, KeyError):
            pass

        return os.environ.get("ADAPTER_REGISTRY_PATH", os.path.expanduser("~/.llm-learn/adapters"))

    def _get_default_profiles(self) -> DotDict | None:
        """Get default training profiles from config."""
        try:
            training_config = self.app.config.learn.get("training", {})
            profiles = training_config.get("default_profiles")
            return DotDict(profiles) if profiles else None
        except (AttributeError, KeyError):
            return None

    def _get_adapter_name(self, provider: TrainingDataProvider, method: str) -> str:
        """Get adapter name from args or generate default."""
        if self.args.adapter:
            return str(self.args.adapter)
        context_key = provider.get_context_key()
        return f"{context_key}-{method}"

    def _submit_manifest(
        self,
        factory: TrainFactory,
        provider: TrainingDataProvider,
        method: Literal["sft", "dpo"],
        data: list[dict[str, Any]],
    ) -> int:
        """Create and submit training manifest."""
        adapter_name = self._get_adapter_name(provider, method)
        context_key = provider.get_context_key()
        description = provider.get_description(method, len(data))

        # Check for existing pending manifest
        existing = factory.manifest.get_pending(adapter_name)
        if existing:
            if not self._confirm_replace(adapter_name):
                print("Aborted.")
                return 1
            factory.manifest.remove_pending(adapter_name)

        parent = self._resolve_parent_adapter(factory)

        manifest = factory.manifest.create(
            adapter=adapter_name,
            method=method,
            model=self.args.model,
            parent=parent,
            data=data,
            context_key=context_key,
            description=description,
        )

        result = factory.manifest.submit(manifest)
        self._print_submit_result(manifest, method, len(data), Path(result.location))
        return 0

    def _confirm_replace(self, adapter_name: str) -> bool:
        """Confirm replacement of existing manifest."""
        print(f"\nPending manifest already exists for '{adapter_name}'")
        try:
            response = input("Replace it? [y/N] ").strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _resolve_parent_adapter(self, factory: TrainFactory) -> Any:
        """Resolve parent adapter from --from-adapter arg.

        Raises:
            AdapterNotFoundError: If the specified adapter doesn't exist.
        """
        from_adapter = getattr(self.args, "from_adapter", None)
        if not from_adapter:
            return None

        parent = factory.manifest.find_adapter(from_adapter)
        if parent is None:
            raise AdapterNotFoundError(from_adapter)
        return parent

    def _print_sft_summary(
        self, examples: list[dict[str, str]], provider: TrainingDataProvider
    ) -> None:
        """Print SFT training summary."""
        print(f"\n=== SFT Training: {provider.get_context_key()} ===")
        print(f"Examples: {len(examples)}")
        print("\nSample:")
        for i, ex in enumerate(examples[:3], 1):
            output = ex.get("output", "")[:60]
            suffix = "..." if len(ex.get("output", "")) > 60 else ""
            print(f"  {i}. {output}{suffix}")
        if len(examples) > 3:
            print(f"  ... and {len(examples) - 3} more")

    def _print_dpo_summary(
        self, pairs: list[dict[str, Any]], provider: TrainingDataProvider
    ) -> None:
        """Print DPO training summary."""
        print(f"\n=== DPO Training: {provider.get_context_key()} ===")
        print(f"Pairs: {len(pairs)}")

    def _print_submit_result(self, manifest: Any, method: str, count: int, path: Path) -> None:
        """Print manifest submission result."""
        print(f"\n✓ Submitted {method.upper()} manifest")
        print(f"  Adapter:   {manifest.adapter}")
        model = manifest.training.get("requested_model") or "(from config)"
        print(f"  Model:     {model}")
        print(f"  Records:   {count}")
        print(f"  Path:      {path}")
