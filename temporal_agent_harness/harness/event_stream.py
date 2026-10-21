# ABOUTME: The pluggable store an agent's events are published to and read from. One backend
# serves three places: the agent workflow (publishes lifecycle events), activities (publish
# deltas and tool events), and clients (read the log back). ``WorkflowStreamBackend`` is the
# default and keeps the log inside the agent workflow itself; another store (e.g. Redis) plugs in
# by implementing ``EventStreamBackend``.
#
# Offsets are part of the protocol (``AgentMessageReply.accepted_offset``, ``attach(from_offset)``,
# the subagent ``from_offset``), so every backend must keep a dense, 0-based, per-agent offset:
# the Nth event published by an agent has offset N.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, NamedTuple, Protocol

from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
)

from temporal_agent_harness.harness.agent_protocol import TURN_EVENTS_TOPIC, AgentEvent

# How often a WorkflowStream subscription re-polls when events arrive faster than one poll.
_POLL_COOLDOWN = timedelta(milliseconds=10)


class StoredEvent(NamedTuple):
    """One event read back from an agent's log, with its offset in that log.

    Named like the SDK's ``WorkflowStreamItem`` (``offset``, ``data``), so code written against
    a WorkflowStream subscription reads it unchanged.
    """

    offset: int
    data: AgentEvent


class EventSink(Protocol):
    """Something events can be published to."""

    def publish(self, event: AgentEvent, /) -> None: ...


class WorkflowEventLog(EventSink, Protocol):
    """The workflow-side view of an agent's log.

    Runs inside workflow code, so both methods must be deterministic and must not block.
    """

    def head_offset(self) -> int:
        """The offset the next published event will get, or a lower bound on it.

        Returned to senders as ``AgentMessageReply.accepted_offset``; the client merge only
        needs it to be at or before the message's ``turn_started``.
        """
        ...


class EventStreamBackend(Protocol):
    """Where an agent's events are stored, as seen from the workflow, activities and clients."""

    def workflow_log(self) -> WorkflowEventLog:
        """Open the log for the current agent workflow. Called once, in ``@agent.init``."""
        ...

    def activity_publisher(
        self, *, batch_interval: timedelta
    ) -> AbstractAsyncContextManager[EventSink]:
        """Open a publisher to the log of the workflow that scheduled the current activity.

        Everything published must be delivered by the time the context exits.
        ``batch_interval`` is how often buffered events are flushed, for backends that batch.
        """
        ...

    async def head_offset(self, client: Client, workflow_id: str) -> int:
        """The offset the next event published by ``workflow_id`` will get."""
        ...

    def subscribe(
        self, client: Client, workflow_id: str, *, from_offset: int
    ) -> AsyncIterator[StoredEvent]:
        """Read ``workflow_id``'s events from ``from_offset``, then follow new ones live.

        Ends only when the log can no longer grow (the agent workflow has finished).
        """
        ...

    def worker_lifetime(self) -> AbstractAsyncContextManager[None]:
        """Held open for as long as a worker publishing to this backend runs.

        ``AgentHarnessPlugin(event_stream=...)`` enters it around the worker, so a backend can
        run something worker-wide (e.g. a background writer) for its workflows.
        """
        ...


class WorkflowStreamBackend:
    """The default backend: the log lives in the agent workflow, as a Temporal ``WorkflowStream``.

    Activities deliver their events to the workflow as Signals, and clients read with Updates.
    """

    def workflow_log(self) -> WorkflowEventLog:
        return WorkflowStreamLog(WorkflowStream())

    @asynccontextmanager
    async def activity_publisher(self, *, batch_interval: timedelta) -> AsyncIterator[EventSink]:
        client = WorkflowStreamClient.from_within_activity(batch_interval=batch_interval)
        async with client:
            yield client.topic(TURN_EVENTS_TOPIC, type=AgentEvent)

    async def head_offset(self, client: Client, workflow_id: str) -> int:
        return await WorkflowStreamClient.create(client, workflow_id).get_offset()

    def worker_lifetime(self) -> AbstractAsyncContextManager[None]:
        return nullcontext()

    async def subscribe(
        self, client: Client, workflow_id: str, *, from_offset: int
    ) -> AsyncIterator[StoredEvent]:
        items = WorkflowStreamClient.create(client, workflow_id).subscribe(
            topics=[TURN_EVENTS_TOPIC],
            from_offset=from_offset,
            result_type=AgentEvent,
            poll_cooldown=_POLL_COOLDOWN,
        )
        try:
            async for item in items:
                yield StoredEvent(offset=item.offset, data=item.data)
        finally:
            # Closing this generator must also close the subscription: that releases the poll
            # Update it holds open against the workflow. (subscribe() returns an async
            # generator, but is typed as a plain iterator.)
            await items.aclose()  # type: ignore[attr-defined]


@dataclass(frozen=True)
class EventReader:
    """A backend's client side, bound to the Temporal client it reads with."""

    backend: EventStreamBackend
    client: Client

    def subscribe(self, workflow_id: str, *, from_offset: int) -> AsyncIterator[StoredEvent]:
        return self.backend.subscribe(self.client, workflow_id, from_offset=from_offset)


class WorkflowStreamLog:
    """``WorkflowEventLog`` over a ``WorkflowStream`` the agent workflow owns."""

    def __init__(self, stream: WorkflowStream) -> None:
        self._stream = stream
        self._events = stream.topic(TURN_EVENTS_TOPIC, type=AgentEvent)

    def publish(self, event: AgentEvent) -> None:
        self._events.publish(event)

    def head_offset(self) -> int:
        # The real log head rather than a count of our own publishes: events from activities
        # land in the same log (as Signals) and a count would miss them.
        return self._stream._on_offset()


# -- Choosing the backend activities publish to ---------------------------------------------

_activity_backend: ContextVar[EventStreamBackend | None] = ContextVar(
    "agent_event_stream_backend", default=None
)


def activity_backend() -> EventStreamBackend:
    """The backend the current activity publishes to: the worker's, else the default."""
    return _activity_backend.get() or WorkflowStreamBackend()


class EventStreamInterceptor(Interceptor):
    """Worker interceptor that points every activity at ``backend``.

    ``AgentHarnessPlugin(event_stream=...)`` installs it; there is no need to add it by hand.
    """

    def __init__(self, backend: EventStreamBackend) -> None:
        self._backend = backend

    def intercept_activity(self, next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:
        return _SetActivityBackend(next, self._backend)


class _SetActivityBackend(ActivityInboundInterceptor):
    def __init__(self, next: ActivityInboundInterceptor, backend: EventStreamBackend) -> None:
        super().__init__(next)
        self._backend = backend

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        token = _activity_backend.set(self._backend)
        try:
            return await super().execute_activity(input)
        finally:
            _activity_backend.reset(token)
