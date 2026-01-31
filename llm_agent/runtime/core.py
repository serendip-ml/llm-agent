"""Runtime core - orchestrates agent subprocess lifecycle.

Core is the runtime orchestrator that manages:
- Subprocess spawning and termination
- Communication with running agents
- Log queue for subprocess logging
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, cast

from appinfra.log.mp import LogQueueListener

from llm_agent.runtime.handle import AgentHandle, AgentInfo
from llm_agent.runtime.state import AgentState, transition
from llm_agent.runtime.transport import Message, MessageType, Request, create_channel_pair


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.traits.learn import LearnTrait
    from llm_agent.core.traits.llm import LLMConfig
    from llm_agent.runtime.registry import AgentRegistry


class Core:
    """Runtime core - orchestrates agent subprocess lifecycle.

    Core manages the runtime aspects of agents:
    - Starting/stopping agent subprocesses
    - Communication with running agents (ask, feedback, insights)
    - Centralized log queue for subprocess logging
    - Graceful shutdown

    It works with AgentRegistry which holds the agent configurations.
    """

    def __init__(
        self,
        lg: Logger,
        registry: AgentRegistry,
        llm_config: LLMConfig,
        learn_trait: LearnTrait | None = None,
        variables: dict[str, str] | None = None,
    ) -> None:
        """Initialize the runtime core.

        Args:
            lg: Logger instance.
            registry: Agent registry for handle lookup.
            llm_config: LLM configuration for agents.
            learn_trait: Optional shared LearnTrait.
            variables: Variable substitutions for agent configs.
        """
        self._lg = lg
        self._registry = registry
        self._llm_config = llm_config
        self._learn_trait = learn_trait
        self._variables = variables or {}

        # Queue-based logging for subprocesses
        self._log_queue: Queue[Any] = Queue()
        self._log_listener = LogQueueListener(self._log_queue, lg)
        self._log_listener.start()

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry for read operations."""
        return self._registry

    def start(self, name: str) -> AgentInfo:
        """Start an agent process.

        Spawns a new subprocess running the agent.

        Args:
            name: Agent name.

        Returns:
            Updated AgentInfo.

        Raises:
            KeyError: If agent not found.
            InvalidTransitionError: If agent cannot be started (wrong state).
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        handle.state = transition(handle.state, AgentState.STARTING)
        handle.error = None

        try:
            self._spawn_process(handle)
            handle.state = transition(handle.state, AgentState.RUNNING)
            self._lg.info("agent started", extra={"agent": name})
        except Exception as e:
            handle.state = AgentState.ERROR
            handle.error = str(e)
            self._lg.warning("failed to start agent", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def stop(self, name: str) -> AgentInfo:
        """Stop an agent process.

        Sends shutdown message and terminates the subprocess.

        Args:
            name: Agent name.

        Returns:
            Updated AgentInfo.

        Raises:
            KeyError: If agent not found.
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        if handle.state != AgentState.RUNNING:
            return AgentInfo.from_handle(handle)

        handle.state = transition(handle.state, AgentState.STOPPING)

        try:
            self._terminate_process(handle)
            handle.state = transition(handle.state, AgentState.STOPPED)
            self._lg.info("agent stopped", extra={"agent": name})
        except Exception as e:
            handle.state = AgentState.ERROR
            handle.error = str(e)
            self._lg.warning("error stopping agent", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def ask(self, name: str, question: str, timeout: float = 60.0) -> str:
        """Ask an agent a question.

        Args:
            name: Agent name.
            question: Question to ask.
            timeout: Response timeout in seconds.

        Returns:
            Agent's response.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running.
            TimeoutError: If no response within timeout.
        """
        handle = self._get_running_handle(name)

        self._lg.debug("ask request", extra={"agent": name, "question_len": len(question)})

        request = Request(type=MessageType.ASK, payload={"question": question})
        response = handle.channel.request(request, timeout=timeout)  # type: ignore[union-attr]

        if not response.success:
            self._lg.warning("ask failed", extra={"agent": name, "error": response.error})
            raise RuntimeError(response.error or "Ask failed")

        self._lg.debug("ask completed", extra={"agent": name})
        return str(response.payload.get("response", ""))

    def feedback(self, name: str, message: str, timeout: float = 30.0) -> None:
        """Send feedback to an agent.

        Args:
            name: Agent name.
            message: Feedback message.
            timeout: Response timeout in seconds.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running or feedback failed.
        """
        handle = self._get_running_handle(name)

        self._lg.debug("feedback request", extra={"agent": name})

        request = Request(type=MessageType.FEEDBACK, payload={"message": message})
        response = handle.channel.request(request, timeout=timeout)  # type: ignore[union-attr]

        if not response.success:
            self._lg.warning("feedback failed", extra={"agent": name, "error": response.error})
            raise RuntimeError(response.error or "Feedback failed")

        self._lg.debug("feedback sent", extra={"agent": name})

    def get_insights(
        self, name: str, limit: int = 10, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Get recent insights from an agent.

        Args:
            name: Agent name.
            limit: Maximum insights to return.
            timeout: Response timeout in seconds.

        Returns:
            List of insight dictionaries.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running.
        """
        handle = self._get_running_handle(name)

        self._lg.debug("get_insights request", extra={"agent": name, "limit": limit})

        request = Request(type=MessageType.GET_INSIGHTS, payload={"limit": limit})
        response = handle.channel.request(request, timeout=timeout)  # type: ignore[union-attr]

        if not response.success:
            self._lg.warning("get_insights failed", extra={"agent": name, "error": response.error})
            raise RuntimeError(response.error or "Get insights failed")

        insights = cast(list[dict[str, Any]], response.payload.get("insights", []))
        self._lg.debug("get_insights completed", extra={"agent": name, "count": len(insights)})
        return insights

    def shutdown(self) -> None:
        """Shutdown all agents and clean up.

        Stops all running agents gracefully.
        """
        self._lg.info("shutting down core")
        for handle in self._registry.handles():
            if handle.state == AgentState.RUNNING:
                try:
                    self.stop(handle.name)
                except Exception as e:
                    self._lg.warning(
                        "error stopping agent during shutdown",
                        extra={"agent": handle.name, "exception": e},
                    )
        self._log_listener.stop()

    def _get_running_handle(self, name: str) -> AgentHandle:
        """Get handle for a running agent.

        Args:
            name: Agent name.

        Returns:
            AgentHandle for the running agent.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running.
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        if handle.state != AgentState.RUNNING or handle.channel is None:
            raise RuntimeError(f"Agent not running: {name}")

        return handle

    def _spawn_process(self, handle: AgentHandle) -> None:
        """Spawn subprocess for agent."""
        self._lg.debug("spawning process", extra={"agent": handle.name})

        main_channel, subprocess_channel = create_channel_pair()
        handle.channel = main_channel

        handle.process = mp.Process(
            target=_subprocess_entry,
            args=(
                handle.name,
                self._build_runner_config(handle),
                subprocess_channel,
                asdict(self._llm_config),
                self._variables,
                self._log_queue,
            ),
            name=f"agent-{handle.name}",
            daemon=True,
        )
        handle.process.start()
        self._wait_for_started(main_channel)

    def _wait_for_started(self, channel: Any) -> None:
        """Wait for subprocess to signal it has started."""
        msg = channel.recv(timeout=30.0)
        if msg is None:
            raise RuntimeError("Agent process did not start within timeout")
        if msg.type == MessageType.ERROR:
            raise RuntimeError(msg.payload.get("error", "Unknown startup error"))

    def _build_runner_config(self, handle: AgentHandle) -> dict[str, Any]:
        """Build configuration dict for the runner."""
        config = dict(handle.config)
        config["name"] = handle.name
        return config

    def _terminate_process(self, handle: AgentHandle) -> None:
        """Terminate agent subprocess."""
        import contextlib

        self._lg.debug("terminating process", extra={"agent": handle.name})

        if handle.channel is not None:
            with contextlib.suppress(Exception):
                handle.channel.send(Message(type=MessageType.SHUTDOWN))

        if handle.process is not None:
            handle.process.join(timeout=5.0)
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=2.0)
            if handle.process.is_alive():
                handle.process.kill()

        handle.process = None
        handle.channel = None


def _subprocess_entry(
    name: str,
    config: dict[str, Any],
    channel: Any,
    llm_config_dict: dict[str, Any],
    variables: dict[str, str],
    log_queue: Queue[Any],
) -> None:
    """Entry point for agent subprocess.

    This runs in the subprocess - creates logger and runner, then runs.
    """
    from appinfra.log import Logger

    from llm_agent.core.traits.llm import LLMConfig
    from llm_agent.runtime.runner import AgentRunner

    lg = Logger.with_queue(log_queue, name=f"agent.{name}", level="debug")
    llm_config = LLMConfig(**llm_config_dict)

    runner = AgentRunner(
        name=name,
        config=config,
        channel=channel,
        lg=lg,
        llm_config=llm_config,
        learn_trait=None,  # LearnTrait can't be passed across processes
        variables=variables,
    )
    runner.run()
