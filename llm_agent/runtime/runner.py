"""Agent runner - runs a single agent in a subprocess.

The AgentRunner is the entry point for agent subprocesses. It:
- Creates and starts the ConversationalAgent
- Runs a sync main loop processing messages from the channel
- Handles scheduled execution if configured
- Responds to shutdown messages cleanly
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from llm_agent.runtime.transport import Message, MessageType, Response


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_agent.core.conversational import ConversationalAgent
    from llm_agent.core.traits.learn import LearnTrait
    from llm_agent.core.traits.llm import LLMConfig
    from llm_agent.runtime.transport import Channel


class AgentRunner:
    """Runs a single agent in a subprocess.

    The runner is responsible for:
    - Creating the agent from config
    - Running the main message loop
    - Handling scheduled execution
    - Processing commands from Core

    This class runs in the subprocess - it receives a channel for
    communication with the main process (Core).

    The main loop is intentionally sync and simple:
    1. Wait for message (with timeout if scheduled)
    2. Handle message or run scheduled cycle
    3. Repeat until shutdown
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        channel: Channel,
        lg: Logger,
        llm_config: LLMConfig,
        learn_trait: LearnTrait | None = None,
        variables: dict[str, str] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            name: Agent name.
            config: Agent configuration dictionary.
            channel: Communication channel to Core.
            lg: Logger instance.
            llm_config: LLM backend configuration.
            learn_trait: Optional LearnTrait for memory.
            variables: Variable substitutions for config.
        """
        self._name = name
        self._config = config
        self._channel = channel
        self._lg = lg
        self._llm_config = llm_config
        self._learn_trait = learn_trait
        self._variables = variables or {}

        self._agent: ConversationalAgent | None = None
        self._running = False
        self._schedule_interval: float | None = None

        # Extract schedule interval from config (must be positive)
        schedule = config.get("schedule")
        if schedule and isinstance(schedule, dict):
            interval = schedule.get("interval")
            if interval is not None:
                interval_float = float(interval)
                self._schedule_interval = interval_float if interval_float > 0 else None

    def run(self) -> None:
        """Main loop (blocking, sync). Called in subprocess.

        Creates the agent, starts it, then enters the message processing loop.
        Exits when shutdown message received or channel closes.
        """
        self._lg.debug("starting agent...", extra={"agent": self._name})

        try:
            self._agent = self._create_agent()
            self._agent.start()
            self._running = True

            # Notify Core we're running
            self._channel.send(Message(type=MessageType.STARTED, payload={"name": self._name}))

            self._run_loop()

        except KeyboardInterrupt:
            pass  # Clean shutdown, no traceback
        except Exception as e:
            self._lg.warning("runner error", extra={"agent": self._name, "exception": e})
            self._channel.send(
                Message(type=MessageType.ERROR, payload={"name": self._name, "error": str(e)})
            )
        finally:
            self._cleanup()

    def _run_loop(self) -> None:
        """Main message processing loop."""
        # Start at 0 to trigger immediate first run for scheduled agents
        last_cycle = 0.0
        self._lg.trace(
            "entering run loop",
            extra={"agent": self._name, "schedule_interval": self._schedule_interval},
        )

        while self._running:
            # Calculate timeout based on schedule
            timeout = self._calculate_timeout(last_cycle)
            msg = self._channel.recv(timeout=timeout)

            if msg is None:
                # Timeout - check if we should run scheduled cycle
                if self._should_run_cycle(last_cycle):
                    self._run_cycle()
                    last_cycle = time.time()
            else:
                self._handle_message(msg)

    def _calculate_timeout(self, last_cycle: float) -> float:
        """Calculate recv timeout based on schedule."""
        if self._schedule_interval is None:
            return 60.0  # Default poll interval

        elapsed = time.time() - last_cycle
        remaining = max(0.1, self._schedule_interval - elapsed)
        return min(remaining, 60.0)

    def _should_run_cycle(self, last_cycle: float) -> bool:
        """Check if scheduled cycle should run."""
        if self._schedule_interval is None:
            return False
        return time.time() - last_cycle >= self._schedule_interval

    def _run_cycle(self) -> None:
        """Run one scheduled execution cycle."""
        if self._agent is None:
            return

        self._lg.debug("running scheduled cycle", extra={"agent": self._name})
        try:
            result = self._agent.run_once()
            self._channel.send(
                Message(
                    type=MessageType.CYCLE_COMPLETE,
                    payload={
                        "name": self._name,
                        "success": result.success,
                        "iterations": result.iterations,
                    },
                )
            )
        except Exception as e:
            self._lg.warning("cycle failed", extra={"agent": self._name, "exception": e})
            self._channel.send(
                Message(
                    type=MessageType.CYCLE_ERROR,
                    payload={"name": self._name, "error": str(e)},
                )
            )

    def _handle_message(self, msg: Message) -> None:
        """Handle message from Core."""
        self._lg.debug("received message", extra={"agent": self._name, "type": msg.type})

        match msg.type:
            case MessageType.SHUTDOWN:
                self._running = False

            case MessageType.ASK:
                self._handle_ask(msg)

            case MessageType.FEEDBACK:
                self._handle_feedback(msg)

            case MessageType.GET_INSIGHTS:
                self._handle_get_insights(msg)

            case MessageType.RUN_CYCLE:
                self._run_cycle()

            case _:
                self._lg.warning(
                    "unknown message type", extra={"agent": self._name, "type": msg.type}
                )

    def _handle_ask(self, msg: Message) -> None:
        """Handle ask request."""
        if self._agent is None:
            self._send_error_response(msg, "Agent not initialized")
            return

        question = msg.payload.get("question", "")
        self._lg.debug("handling ask", extra={"agent": self._name, "question_len": len(question)})

        try:
            response_text = self._agent.ask(question)
            self._lg.debug(
                "ask completed", extra={"agent": self._name, "response_len": len(response_text)}
            )
            self._channel.send(
                Response(
                    id=uuid4().hex,
                    type=MessageType.ASK_RESPONSE,
                    request_id=msg.id,
                    success=True,
                    payload={"response": response_text},
                )
            )
        except Exception as e:
            self._lg.warning("ask failed", extra={"agent": self._name, "exception": e})
            self._send_error_response(msg, str(e))

    def _handle_feedback(self, msg: Message) -> None:
        """Handle feedback request."""
        if self._agent is None:
            self._send_error_response(msg, "Agent not initialized")
            return

        message = msg.payload.get("message", "")
        self._lg.debug("handling feedback", extra={"agent": self._name})

        try:
            self._agent.record_feedback(message)
            self._lg.debug("feedback recorded", extra={"agent": self._name})
            self._channel.send(
                Response(
                    id=uuid4().hex,
                    type=MessageType.FEEDBACK_RESPONSE,
                    request_id=msg.id,
                    success=True,
                )
            )
        except Exception as e:
            self._lg.warning("feedback failed", extra={"agent": self._name, "exception": e})
            self._send_error_response(msg, str(e))

    def _handle_get_insights(self, msg: Message) -> None:
        """Handle get_insights request."""
        if self._agent is None:
            self._send_error_response(msg, "Agent not initialized")
            return

        limit = msg.payload.get("limit", 10)
        self._lg.debug("handling get_insights", extra={"agent": self._name, "limit": limit})

        try:
            insights = self._build_insights(limit)
            self._lg.debug(
                "get_insights completed", extra={"agent": self._name, "count": len(insights)}
            )
            self._channel.send(
                Response(
                    id=uuid4().hex,
                    type=MessageType.INSIGHTS_RESPONSE,
                    request_id=msg.id,
                    success=True,
                    payload={
                        "insights": insights,
                        "cycle_count": self._agent.cycle_count,
                    },
                )
            )
        except Exception as e:
            self._lg.warning("get_insights failed", extra={"agent": self._name, "exception": e})
            self._send_error_response(msg, str(e))

    def _build_insights(self, limit: int) -> list[dict[str, Any]]:
        """Build insights list from recent results."""
        assert self._agent is not None  # Caller (_handle_get_insights) checks this
        results = self._agent.get_recent_results(limit)
        return [
            {
                "success": r.success,
                "content": r.content[:500],
                "parsed": r.parsed.model_dump() if r.parsed else None,
                "iterations": r.iterations,
            }
            for r in results
        ]

    def _send_error_response(self, msg: Message, error: str) -> None:
        """Send error response for a request."""
        # Map request types to response types
        response_type_map: dict[str, str] = {
            MessageType.ASK: MessageType.ASK_RESPONSE,
            MessageType.FEEDBACK: MessageType.FEEDBACK_RESPONSE,
            MessageType.GET_INSIGHTS: MessageType.INSIGHTS_RESPONSE,
        }
        response_type = response_type_map.get(msg.type, MessageType.ERROR)

        self._channel.send(
            Response(
                id=uuid4().hex,
                type=response_type,
                request_id=msg.id,
                success=False,
                error=error,
            )
        )

    def _create_agent(self) -> ConversationalAgent:
        """Create ConversationalAgent from config."""
        from llm_agent.core.factory import create_agent_from_config

        self._lg.trace("creating agent...", extra={"agent": self._name})

        agent = create_agent_from_config(
            lg=self._lg,
            config_dict=self._config,
            llm_config=self._llm_config,
            learn_trait=self._learn_trait,
            variables=self._variables,
        )

        self._lg.debug("agent created", extra={"agent": self._name})
        return agent

    def _cleanup(self) -> None:
        """Clean up on exit."""
        if self._agent is not None:
            try:
                self._agent.stop()
            except Exception as e:
                self._lg.warning("error stopping agent", extra={"exception": e})

        self._channel.close()
        self._lg.info("runner stopped", extra={"agent": self._name})
