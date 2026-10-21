# ABOUTME: An EventStreamBackend that keeps each agent's events in a Redis Stream, so streaming
# costs no Temporal actions or history. Activities write to Redis directly, workflow code hands
# its events to a background writer on the worker, and clients read with blocking XREADs.
# Requires the ``redis`` extra.
#
# DESIGN — best effort. The stream is a live view, not the record: a message's result is read
# durably with ``await_message``, and pending approvals and callbacks are on ``agent_status``.
# Events are lost if a worker dies with events still queued, or Redis stays unreachable.
#
# DESIGN — offsets. A Lua script appends each event at offset = the stream's current length,
# under the stream ID ``0-<offset + 1>``, so offsets are dense and 0-based as EventStreamBackend
# requires, and a reader resumes from offset N by reading after ID ``0-N``.
#
# DESIGN — workflow events skip Temporal entirely. Workflow code can't do I/O, so its log only
# numbers each event and hands it, thread-safely, to the writer the worker runs. Nothing is
# handed over while history is replayed: those events were written when the code first ran.
# Each workflow event carries the id ``<run_id>:<n>``, so a workflow task that fails and runs
# again does not write its events twice. AgentHarnessPlugin starts the writer, and shares this
# module into the workflow sandbox so workflow code can reach it.

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import cast

from redis.asyncio import Redis
from temporalio import activity, workflow
from temporalio.client import Client

from temporal_agent_harness.harness.agent_protocol import AgentEvent
from temporal_agent_harness.harness.event_stream import (
    EventSink,
    StoredEvent,
    WorkflowEventLog,
)

_log = logging.getLogger(__name__)

# Appends ARGV, read as (event_json, dedupe_id) pairs, to the stream KEYS[1]. A pair whose
# dedupe_id is already in the set KEYS[2] is skipped; an empty dedupe_id is never skipped.
_APPEND_SCRIPT = """
local offset = redis.call('XLEN', KEYS[1])
for i = 1, #ARGV, 2 do
  local dedupe_id = ARGV[i + 1]
  if dedupe_id == '' or redis.call('SADD', KEYS[2], dedupe_id) == 1 then
    redis.call('XADD', KEYS[1], '0-' .. (offset + 1), 'event', ARGV[i])
    offset = offset + 1
  end
end
return offset
"""

# What XREAD returns: (stream key, [(entry id, fields)]) per stream read.
_XReadResponse = list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]

# The longest one blocking read waits before asking again (see _read_block_ms).
_READ_BLOCK = timedelta(seconds=5)
# The connection's socket timeout, unless the URL sets one: long enough for a blocking read.
_SOCKET_TIMEOUT = 2 * _READ_BLOCK
# Attempts at a write before its events are dropped.
_WRITE_ATTEMPTS = 3


class RedisEventStream:
    """Keeps agent events in Redis, one stream per agent workflow.

    Configure workers and clients with the Redis URL, and build the agent's runner with
    ``RedisEventStream()``::

        events = RedisEventStream("redis://localhost:6379")
        AgentHarnessPlugin(event_stream=events)                  # workers
        AgentClient(temporal, workflow_id, event_stream=events)  # clients, e.g. the web server

        @agent.init
        def __init__(self, config: AgentConfig) -> None:
            self._runner = AgentWorkflowRunner(config, stream=RedisEventStream(), ...)

    Workflow code needs no URL: its events go through the writer of the worker running it.
    ``key_prefix`` namespaces the Redis keys and must match on every worker and client.
    """

    def __init__(self, url: str | None = None, *, key_prefix: str = "agent-events") -> None:
        self._url = url
        self._key_prefix = key_prefix
        self._redis: Redis | None = None

    # -- EventStreamBackend --------------------------------------------------------------

    def workflow_log(self) -> WorkflowEventLog:
        return _WorkflowLog()

    @asynccontextmanager
    async def activity_publisher(self, *, batch_interval: timedelta) -> AsyncIterator[EventSink]:
        workflow_id = activity.info().workflow_id
        if workflow_id is None:
            raise RuntimeError("Agent events can only be published from a workflow's activity.")
        publisher = _BufferedPublisher(self, workflow_id)
        stop = asyncio.Event()
        flusher = asyncio.create_task(publisher.flush_until(stop, every=batch_interval))
        try:
            yield publisher
        finally:
            stop.set()
            await flusher

    async def head_offset(self, client: Client, workflow_id: str) -> int:
        return await self._connection().xlen(self._stream_key(workflow_id))

    async def subscribe(
        self, client: Client, workflow_id: str, *, from_offset: int
    ) -> AsyncIterator[StoredEvent]:
        redis = self._connection()
        key = self._stream_key(workflow_id)
        last_id = f"0-{from_offset}"
        block_ms = self._read_block_ms()
        while True:
            response = cast(_XReadResponse, await redis.xread({key: last_id}, block=block_ms))
            for _stream, entries in response:
                for entry_id, fields in entries:
                    last_id = entry_id
                    yield StoredEvent(
                        offset=_offset_of(entry_id),
                        data=AgentEvent.model_validate_json(fields[b"event"]),
                    )

    @asynccontextmanager
    async def worker_lifetime(self) -> AsyncIterator[None]:
        """Run the writer that workflow code on this worker hands its events to."""
        async with _shared_writer(self):
            yield

    # -- Writing -------------------------------------------------------------------------

    async def append(self, workflow_id: str, events: list[tuple[str, str]]) -> None:
        """Append ``(event_json, dedupe_id)`` pairs to ``workflow_id``'s stream, in order.

        Best effort: retried briefly, then dropped with a warning, never raised.
        """
        for attempt in range(1, _WRITE_ATTEMPTS + 1):
            try:
                await self._connection().eval(
                    _APPEND_SCRIPT,
                    2,
                    self._stream_key(workflow_id),
                    self._dedupe_key(workflow_id),
                    *(part for event in events for part in event),
                )
                return
            except Exception as e:  # noqa: BLE001 — a lost live view must never fail the agent
                if attempt == _WRITE_ATTEMPTS:
                    _log.warning(
                        "Dropping %d events for %s after %d failed writes: %r",
                        len(events),
                        workflow_id,
                        attempt,
                        e,
                    )
                    return
                await asyncio.sleep(0.1 * attempt)

    # -- Internal ------------------------------------------------------------------------

    def _connection(self) -> Redis:
        if self._redis is None:
            if self._url is None:
                raise RuntimeError(
                    "RedisEventStream() without a url can only be used as an agent's stream in "
                    "workflow code; workers and clients need RedisEventStream(url)."
                )
            self._redis = Redis.from_url(
                self._url, socket_timeout=_SOCKET_TIMEOUT.total_seconds()
            )
        return self._redis

    def _read_block_ms(self) -> int:
        """How long a blocking read may wait: inside the connection's socket timeout (ours, or
        the URL's ``socket_timeout``), or the read raises instead of returning nothing."""
        block = _READ_BLOCK.total_seconds()
        socket_timeout = self._connection().connection_pool.connection_kwargs.get("socket_timeout")
        if socket_timeout:
            block = min(block, socket_timeout / 2)
        return int(block * 1000)

    def _stream_key(self, workflow_id: str) -> str:
        return f"{self._key_prefix}:{workflow_id}"

    def _dedupe_key(self, workflow_id: str) -> str:
        return f"{self._key_prefix}:{workflow_id}:written"


def _offset_of(entry_id: bytes) -> int:
    """The offset stored in a stream ID ``0-<offset + 1>``."""
    return int(entry_id.split(b"-")[1]) - 1


class _BufferedPublisher:
    """Activity side: collects events and appends them in batches."""

    def __init__(self, stream: RedisEventStream, workflow_id: str) -> None:
        self._stream = stream
        self._workflow_id = workflow_id
        self._pending: list[str] = []

    def publish(self, event: AgentEvent, /) -> None:
        self._pending.append(event.model_dump_json())

    async def flush_until(self, stop: asyncio.Event, *, every: timedelta) -> None:
        """Flush on every tick until ``stop`` is set, then flush what is left."""
        stopped = False
        while not stopped:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), every.total_seconds())
            stopped = stop.is_set()
            await self._flush()

    async def _flush(self) -> None:
        if self._pending:
            events, self._pending = self._pending, []
            await self._stream.append(self._workflow_id, [(event, "") for event in events])


class _WorkflowLog:
    """Workflow side: numbers each event and hands it to the worker's writer."""

    def __init__(self) -> None:
        info = workflow.info()
        self._workflow_id = info.workflow_id
        self._run_id = info.run_id
        self._published = 0

    def publish(self, event: AgentEvent, /) -> None:
        self._published += 1
        if workflow.unsafe.is_replaying_history_events():
            return
        _running_writer().enqueue(
            self._workflow_id, event.model_dump_json(), f"{self._run_id}:{self._published}"
        )

    def head_offset(self) -> int:
        # Everything this workflow has published lands in the stream before its next event,
        # along with any activity events, so the count is a lower bound on the next offset.
        return self._published


class _Writer:
    """Appends events handed over by workflow code, from the worker's event loop."""

    def __init__(self, stream: RedisEventStream) -> None:
        self.stream = stream
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[tuple[str, str, str] | None] = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    def enqueue(self, workflow_id: str, event_json: str, dedupe_id: str) -> None:
        """Queue one event. Safe to call from workflow code, which runs on another thread."""
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, (workflow_id, event_json, dedupe_id)
        )

    def close(self) -> None:
        """Stop once everything already queued is written, without waiting for it.

        Not awaited because a worker run with ``async with Worker(...)`` is cancelled as soon as
        it has stopped, which would cancel the wait (and surface in the caller as a
        CancelledError). The drain finishes in the background.
        """
        self._queue.put_nowait(None)
        _draining.add(self._task)
        self._task.add_done_callback(_draining.discard)

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get()]
            while not self._queue.empty():
                batch.append(self._queue.get_nowait())
            by_workflow: dict[str, list[tuple[str, str]]] = {}
            for item in batch:
                if item is not None:
                    workflow_id, event_json, dedupe_id = item
                    by_workflow.setdefault(workflow_id, []).append((event_json, dedupe_id))
            for workflow_id, events in by_workflow.items():
                await self.stream.append(workflow_id, events)
            if None in batch:
                return


# The writer for this process, shared by every worker using the backend. Module state, because
# workflow code has no other way to reach the worker: AgentHarnessPlugin passes this module
# through the workflow sandbox, so workflow code sees this same variable.
_writer: _Writer | None = None
_writer_users = 0
# Writers still writing out their queue after their workers stopped, kept so they finish.
_draining: set[asyncio.Task[None]] = set()


@asynccontextmanager
async def _shared_writer(stream: RedisEventStream) -> AsyncIterator[None]:
    global _writer, _writer_users
    if _writer is None:
        _writer = _Writer(stream)
    elif _writer.stream is not stream:
        raise RuntimeError("Every worker in a process must share one RedisEventStream instance.")
    _writer_users += 1
    try:
        yield
    finally:
        _writer_users -= 1
        if _writer_users == 0:
            _writer.close()
            _writer = None


def _running_writer() -> _Writer:
    if _writer is None:
        raise RuntimeError(
            "No Redis event writer is running in this worker. Pass the RedisEventStream to "
            "AgentHarnessPlugin(event_stream=...) on the worker."
        )
    return _writer
