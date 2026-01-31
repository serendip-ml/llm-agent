"""Stop tool - stop an agent."""

import argparse
from typing import Any

import httpx
from appinfra.app.tools import Tool, ToolConfig


class StopTool(Tool):
    """Stop an agent."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="stop", help_text="Stop an agent by name")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to stop",
        )
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def run(self, **kwargs: Any) -> int:
        server = self.args.server.rstrip("/")
        name = self.args.name

        try:
            response = httpx.post(f"{server}/agents/{name}/stop", timeout=30.0)

            if response.status_code == 404:
                print(f"Error: Agent '{name}' not found")
                return 1

            response.raise_for_status()
            data = response.json()

        except httpx.RequestError as e:
            self.lg.error("failed to connect to server", extra={"exception": e})
            print(f"Error: Could not connect to gateway at {server}")
            return 1

        status = data.get("status", "unknown")
        if status == "stopped":
            print(f"Agent '{name}' stopped successfully")
        elif status == "error":
            print(f"Agent '{name}' failed to stop: {data.get('error', 'Unknown error')}")
            return 1
        else:
            print(f"Agent '{name}' status: {status}")

        return 0
