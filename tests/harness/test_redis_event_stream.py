# ABOUTME: Tests for RedisEventStream against a real redis-server (skipped when none is
# installed): the append script's offsets and de-duplication, and an agent whose whole turn —
# workflow events and activity deltas — streams through Redis with no Signals in its history.

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import uuid
from collections.abc import AsyncIterator

import pytest
from _redis_agent import RedisEchoAgent, stream_words
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from temporalio.api.enums.v1 import EventType
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_agent_harness.harness.agent_client import AgentClient
from temporal_agent_harness.harness.agent_protocol import (
    AgentConfig,
    AgentEvent,
    AgentEventType,
    ReplyDelta,
)
from temporal_agent_harness.harness.event_stream import StoredEvent
from temporal_agent_harness.harness.redis_event_stream import RedisEventStream
from temporal_agent_harness.plugin import AgentHarnessPlugin


@pytest.fixture
async def redis_url(tmp_path) -> AsyncIterator[str]:
    """A throwaway redis-server on a free port."""
    server = shutil.which("redis-server")
    if server is None:
        pytest.skip("redis-server is not installed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    process = subprocess.Popen(
        [server, "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(tmp_path)],
        stdout=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}"
    try:
        async with asyncio.timeout(10):
            while True:
                try:
                    await Redis.from_url(url).ping()
                    break
                except RedisConnectionError:
                    await asyncio.sleep(0.05)
        yield url
    finally:
        process.terminate()
        process.wait()


def _delta(text: str) -> AgentEvent:
    return AgentEvent(
        agent_id="a1b2c3", turn_id="t1", turn_number=1, timestamp=0.0, event=ReplyDelta(text=text)
    )


async def _read(stream: RedisEventStream, workflow_id: str, *, from_offset: int, count: int):
    """The first ``count`` events from ``from_offset`` (subscribe itself never ends)."""
    events: list[StoredEvent] = []
    async with asyncio.timeout(10):
        async for stored in stream.subscribe(None, workflow_id, from_offset=from_offset):
            events.append(stored)
            if len(events) == count:
                return events
    return events


async def test_offsets_are_dense_and_a_rewritten_event_is_skipped(redis_url):
    stream = RedisEventStream(redis_url)
    first, second = _delta("one").model_dump_json(), _delta("two").model_dump_json()

    # The workflow event with id "run:1" is handed over twice, as after a retried workflow task.
    await stream.append("wf", [(first, "run:1"), (second, "")])
    await stream.append("wf", [(first, "run:1")])

    assert await stream.head_offset(None, "wf") == 2
    events = await _read(stream, "wf", from_offset=0, count=2)
    assert [(e.offset, e.data.event.text) for e in events] == [(0, "one"), (1, "two")]
    resumed = await _read(stream, "wf", from_offset=1, count=1)
    assert [(e.offset, e.data.event.text) for e in resumed] == [(1, "two")]


@pytest.mark.parametrize("query", ["", "?socket_timeout=0.5"])
async def test_a_blocking_read_ends_inside_the_socket_timeout(redis_url, query):
    """With redis-py's defaults (a 5s socket timeout in 8.x) or the URL's own timeout."""
    stream = RedisEventStream(redis_url + query)
    pool = stream._connection().connection_pool
    connection = await pool.get_connection()
    try:
        assert stream._read_block_ms() < connection.socket_timeout * 1000
    finally:
        await pool.release(connection)


async def test_an_idle_subscription_outlasts_the_socket_timeout(redis_url):
    """A quiet stream must keep waiting, not raise once the connection's socket timeout passes."""
    stream = RedisEventStream(f"{redis_url}?socket_timeout=0.5")
    reading = asyncio.create_task(_read(stream, "wf", from_offset=0, count=1))
    await asyncio.sleep(1.5)
    await stream.append("wf", [(_delta("late").model_dump_json(), "")])

    events = await reading
    assert [e.data.event.text for e in events] == ["late"]


# max_cached_workflows=0 makes every workflow task replay the whole history first, so replayed
# events must be neither written again nor counted out of order.
@pytest.mark.parametrize("max_cached_workflows", [1000, 0])
async def test_an_agent_streams_its_turn_through_redis(redis_url, max_cached_workflows):
    """Workflow events and activity deltas both arrive, in order, and none of it went through
    Temporal: the agent's history has no Signals (WorkflowStream would add one per flush)."""
    events = RedisEventStream(redis_url)
    task_queue = f"redis-agent-{uuid.uuid4()}"

    async with await WorkflowEnvironment.start_time_skipping(
        plugins=[AgentHarnessPlugin(event_stream=events)]
    ) as env:
        # Sandboxed (the default runner): the plugin must share the Redis module with workflows.
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[RedisEchoAgent],
            activities=[stream_words],
            max_cached_workflows=max_cached_workflows,
        ):
            handle = await env.client.start_workflow(
                RedisEchoAgent.run, AgentConfig(), id=f"redis-{uuid.uuid4()}", task_queue=task_queue
            )
            client = AgentClient(env.client, handle.id, event_stream=events)
            turn = await client.send_message(
                "echo", {"text": "hello redis world"}, on_item=lambda item, _offset: item
            )
            streamed = [item async for item in turn]
            history = await handle.fetch_history()

    assert [e.event.type for e in streamed] == [
        AgentEventType.TURN_STARTED,
        AgentEventType.MESSAGE_HANDLER_START,
        AgentEventType.REPLY_DELTA,
        AgentEventType.REPLY_DELTA,
        AgentEventType.REPLY_DELTA,
        AgentEventType.MESSAGE_HANDLER_END,
        AgentEventType.TURN_END,
    ]
    assert [e.event.text for e in streamed if e.event.type == AgentEventType.REPLY_DELTA] == [
        "hello",
        "redis",
        "world",
    ]
    assert await events.head_offset(None, handle.id) == 8  # those 7 plus message_accepted
    signals = [
        e for e in history.events if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED
    ]
    assert signals == []
