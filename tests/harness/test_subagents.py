# ABOUTME: Unit tests for the subagent wrapper's pure, in-workflow bookkeeping — the
# per-subagent FIFO gate (call-order admission) and the subagent registry on _WorkflowStatus.
# These need no Temporal env; the activity dispatch + end-to-end parent→subagent flow are
# covered separately (they require a running child agent workflow).

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from pydantic import BaseModel, Field
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from temporal_agent_harness.harness import agent
from temporal_agent_harness.harness.agent_protocol import (
    AgentEvent,
    AgentMessageReply,
    MessageDisposition,
    MessageResult,
    RunSubagentTurnInput,
    SubagentMessageSent,
    SubagentStarted,
    SubagentStopped,
    ToolApprovalPolicy,
)
from temporal_agent_harness.harness.agent_workflow import _SubagentInstance, _WorkflowStatus
from temporal_agent_harness.harness import agent_workflow, subagent_activities
from temporal_agent_harness.harness.subagent_activities import (
    SubagentActivities,
    _TurnProgress,
)
from temporal_agent_harness.harness.stream_context import TurnStreamContext


# A minimal child agent for exercising the toolset generator. No @workflow.defn is needed —
# subagent_toolset reads the @agent.accepts handlers by pure reflection, and falls back to the
# class name for the workflow type. ``ask`` and ``summarize`` deliberately share one input model
# (name-routed dispatch allows it).
class _Question(BaseModel):
    """A question to research."""

    text: str = Field(description="The natural-language question to research.")


class _Answer(BaseModel):
    """An answer to a question."""

    text: str = Field(description="The answer text.")


class _SampleChildAgent:
    @agent.accepts
    async def ask(self, q: _Question) -> _Answer:
        """Answer a free-form question."""
        ...

    @agent.accepts
    async def summarize(self, q: _Question) -> _Answer:
        """Summarize the conversation so far."""
        ...


class _SampleChildAgentWithOperatorOnly:
    """A child that declares an operator-only handler alongside a model-callable one."""

    @agent.accepts
    async def ask(self, q: _Question) -> _Answer:
        """Answer a free-form question."""
        ...

    @agent.accepts(model_callable=False)
    async def set_config(self, q: _Question) -> _Answer:
        """Reconfigure the session — an operator surface, not for a parent's model."""
        ...


def _status() -> _WorkflowStatus:
    return _WorkflowStatus(
        agent_id="parent",
        approval_policy=ToolApprovalPolicy.allow_inherently_safe(),
    )


def test_gate_hands_out_tickets_in_call_order():
    inst = _SubagentInstance(handle="h", workflow_id="wf", agent_key="k")
    # Tickets are monotonic so gathered callers are ordered by the order they call take_ticket
    # (i.e. the model's call order), not by await scheduling.
    assert [inst.take_ticket() for _ in range(3)] == [0, 1, 2]


def test_gate_serves_one_ticket_at_a_time_in_order():
    inst = _SubagentInstance(handle="h", workflow_id="wf", agent_key="k")
    t0, t1, t2 = inst.take_ticket(), inst.take_ticket(), inst.take_ticket()

    # Only the first ticket is admitted initially.
    assert inst.is_serving(t0)
    assert not inst.is_serving(t1)
    assert not inst.is_serving(t2)

    # Releasing passes the gate to the next ticket, strictly in order.
    inst.release_gate()
    assert not inst.is_serving(t0)
    assert inst.is_serving(t1)
    assert not inst.is_serving(t2)

    inst.release_gate()
    assert inst.is_serving(t2)


def test_gate_sequential_take_serve_release_never_blocks():
    inst = _SubagentInstance(handle="h", workflow_id="wf", agent_key="k")
    # The common (non-concurrent) path: take → already serving → release, repeatedly. Each
    # ticket is served the moment it is taken (no waiting), since the prior one released.
    for expected in range(3):
        ticket = inst.take_ticket()
        assert ticket == expected
        assert inst.is_serving(ticket)
        inst.release_gate()
    # Three releases advanced _serving to 3, so the next ticket (3) is served immediately too.
    assert inst.take_ticket() == 3
    assert inst.is_serving(3)


def test_register_keys_by_handle_and_stores_workflow_id():
    st = _status()
    inst = st.register_subagent("a3f9c2", "sample-subagent-<uuid>", "sample")
    assert inst.handle == "a3f9c2"
    assert inst.workflow_id == "sample-subagent-<uuid>"  # the real child id, hidden from the model
    assert inst.agent_key == "sample"
    assert inst.last_consumed_offset == 0
    assert st.subagent("a3f9c2") is inst
    assert st.has_subagent("a3f9c2") and not st.has_subagent("nope")


def test_lookup_unknown_subagent_raises_typed_error():
    st = _status()
    with pytest.raises(ApplicationError) as excinfo:
        st.subagent("nope")
    err = excinfo.value
    assert err.type == "UnknownSubagent"
    assert err.non_retryable
    # details carry the offending handle + the known set for a useful model-facing message.
    assert err.details[0] == {"handle": "nope", "known": []}


def test_remove_subagent_is_idempotent_and_then_unknown():
    st = _status()
    st.register_subagent("a3f9c2", "wf", "sample")
    st.remove_subagent("a3f9c2")
    with pytest.raises(ApplicationError):
        st.subagent("a3f9c2")
    # Removing again is a no-op, not an error.
    st.remove_subagent("a3f9c2")


def test_agent_status_lists_subagents_without_gate_internals():
    st = _status()
    inst = st.register_subagent("a3f9c2", "sample-subagent-wf", "sample")
    # Hand out a couple of gate tickets so the internal counters are non-default.
    inst.take_ticket()
    inst.take_ticket()

    status = st.to_agent_status()
    assert len(status.subagents) == 1
    info = status.subagents[0]
    assert (info.subagent_id, info.agent_key, info.workflow_id) == (
        "a3f9c2",
        "sample",
        "sample-subagent-wf",
    )
    # The caller-side gate's ticket counters are an implementation detail and must NOT leak
    # into the status projection.
    fields = set(vars(info))
    assert "_next_ticket" not in fields
    assert "_serving" not in fields
    assert fields == {"subagent_id", "agent_key", "workflow_id"}


def test_subagent_lifecycle_events_carry_workflow_id_and_round_trip():
    # SubagentStarted/Stopped must survive the AgentEvent discriminated-union round-trip with
    # the child workflow_id intact — that field is what lets a consumer dynamically mount the
    # subagent's own stream for a consolidated view.
    for ev in (
        SubagentStarted(subagent_id="a3f9c2", agent_key="sample", workflow_id="sample-subagent-wf"),
        SubagentStopped(subagent_id="a3f9c2", agent_key="sample", workflow_id="sample-subagent-wf"),
    ):
        envelope = AgentEvent(
            event=ev, agent_id="parent-wf", turn_id="t1", turn_number=2, timestamp=0.0
        )
        back = AgentEvent.model_validate_json(envelope.model_dump_json())
        assert type(back.event) is type(ev)
        # The envelope agent_id is the PARENT; the payload subagent_id is the child being driven.
        assert back.agent_id == "parent-wf"
        assert back.event.workflow_id == "sample-subagent-wf"
        assert back.event.subagent_id == "a3f9c2"
        assert back.event.agent_key == "sample"


def test_subagent_message_sent_event_round_trips_with_dispatch_details():
    # SubagentMessageSent marks a dispatch to a specific subagent on the parent's stream. It
    # must round-trip through the discriminated union carrying enough to correlate with the
    # child's own stream: the handle/workflow_id, the target handler, and the child turn
    # number.
    ev = SubagentMessageSent(
        subagent_id="a3f9c2",
        agent_key="monty",
        workflow_id="monty-subagent-wf",
        handler="run_script",
        subagent_turn=3,
    )
    # Envelope turn_number (the parent's turn) is deliberately DIFFERENT from subagent_turn (the
    # child's turn) — the two must not be conflated, which is exactly why the payload field is
    # named subagent_turn rather than turn_number.
    envelope = AgentEvent(
        event=ev, agent_id="parent-wf", turn_id="t1", turn_number=5, timestamp=0.0
    )
    back = AgentEvent.model_validate_json(envelope.model_dump_json())
    assert type(back.event) is SubagentMessageSent
    assert back.event.subagent_id == "a3f9c2"
    assert back.event.agent_key == "monty"
    assert back.event.workflow_id == "monty-subagent-wf"
    assert back.event.handler == "run_script"
    assert back.event.subagent_turn == 3
    assert back.turn_number == 5  # the parent turn, on the envelope — not the subagent's


def test_toolset_emits_namespaced_start_send_stop_tools():
    tools = agent.subagent_toolset(_SampleChildAgent, key="sample", task_queue="sample-q")
    # start_<key>, one <key>_<fn> per handler (discovery order is alphabetical), then stop_<key>.
    assert [t.__name__ for t in tools] == ["start_sample", "sample_ask", "sample_summarize", "stop_sample"]
    # Every generated tool is a real inline tool_defn — so it inherits run_tool dispatch, the
    # approval gate, and tool lifecycle events with no extra wiring.
    assert all(getattr(t, "__agent_tool__", False) for t in tools)


def test_toolset_honors_model_callable_hints_by_default():
    """The default policy omits a handler the child marked ``model_callable=False``.

    This is the replacement for the old magic-name filter: what a parent's model can reach is
    decided by the policy, not by what a handler happens to be called."""
    tools = agent.subagent_toolset(
        _SampleChildAgentWithOperatorOnly, key="sample", task_queue="sample-q"
    )

    assert [t.__name__ for t in tools] == ["start_sample", "sample_ask", "stop_sample"]


def test_toolset_policy_can_narrow_below_the_hints():
    """``allow_only`` is authoritative even when the hints are more permissive."""
    tools = agent.subagent_toolset(
        _SampleChildAgent,
        key="sample",
        task_queue="sample-q",
        tools=agent.SubagentToolPolicy.allow_only("summarize"),
    )

    assert [t.__name__ for t in tools] == [
        "start_sample",
        "sample_summarize",
        "stop_sample",
    ]


def test_toolset_policy_can_override_a_non_model_callable_hint():
    """A parent naming a handler explicitly overrides its ``model_callable=False`` hint —
    naming it IS the override. The hint is intent, never an access decision."""
    tools = agent.subagent_toolset(
        _SampleChildAgentWithOperatorOnly,
        key="sample",
        task_queue="sample-q",
        tools=agent.SubagentToolPolicy.allow_only("set_config"),
    )

    assert [t.__name__ for t in tools] == [
        "start_sample",
        "sample_set_config",
        "stop_sample",
    ]


def test_toolset_dangerously_allow_all_ignores_the_hints():
    tools = agent.subagent_toolset(
        _SampleChildAgentWithOperatorOnly,
        key="sample",
        task_queue="sample-q",
        tools=agent.SubagentToolPolicy.dangerously_allow_all(),
    )

    assert [t.__name__ for t in tools] == [
        "start_sample",
        "sample_ask",
        "sample_set_config",
        "stop_sample",
    ]


def test_toolset_rejects_a_policy_that_selects_nothing():
    """A child whose whole surface is operator-only has no callable surface to wire, and the
    error names what it declared so the mistake is diagnosable."""

    class _OperatorOnlyChild:
        @agent.accepts(model_callable=False)
        async def set_config(self, q: _Question) -> _Answer:
            """Operator-only."""
            ...

    with pytest.raises(TypeError, match="no handlers under the given SubagentToolPolicy"):
        agent.subagent_toolset(_OperatorOnlyChild, key="x", task_queue="q")


def test_send_tool_signature_uses_the_childs_real_models():
    tools = {t.__name__: t for t in agent.subagent_toolset(
        _SampleChildAgent, key="sample", task_queue="sample-q"
    )}
    ask = tools["sample_ask"]
    sig = inspect.signature(ask)
    # subagent handle (str) + the handler's own input param name, typed as the real input model.
    assert list(sig.parameters) == ["subagent", "q"]
    assert sig.parameters["subagent"].annotation is str
    assert sig.parameters["q"].annotation is _Question
    assert sig.return_annotation is _Answer
    # The model-facing annotations (what function_param/from_callable introspects) match — so the
    # emitted schema is the child's real input model (nested, descriptions preserved) and the
    # return type is the real output model.
    assert ask.__annotations__ == {"subagent": str, "q": _Question, "return": _Answer}


def test_send_tool_docstring_carries_the_handler_description():
    tools = {t.__name__: t for t in agent.subagent_toolset(
        _SampleChildAgent, key="sample", task_queue="sample-q"
    )}
    assert "Answer a free-form question." in (tools["sample_ask"].__doc__ or "")
    assert "Summarize the conversation so far." in (tools["sample_summarize"].__doc__ or "")


def test_start_and_stop_tool_signatures():
    tools = {t.__name__: t for t in agent.subagent_toolset(
        _SampleChildAgent, key="sample", task_queue="sample-q"
    )}
    assert list(inspect.signature(tools["start_sample"]).parameters) == []
    assert inspect.signature(tools["start_sample"]).return_annotation is str
    assert list(inspect.signature(tools["stop_sample"]).parameters) == ["subagent"]


def test_toolset_requires_at_least_one_handler():
    class _NoHandlers:
        pass

    with pytest.raises(TypeError, match="no handlers under the given SubagentToolPolicy"):
        agent.subagent_toolset(_NoHandlers, key="x", task_queue="q")


def test_distinct_subagents_have_independent_gates_and_counters():
    st = _status()
    a = st.register_subagent("aaa111", "wf-a", "sample")
    b = st.register_subagent("bbb222", "wf-b", "sample")
    # Same agent_key, but independent instances/gates so different subagents run concurrently.
    assert a is not b
    a.take_ticket()
    assert a._next_ticket == 1
    assert b._next_ticket == 0


# ---------------------------------------------------------------------------
# run_subagent_turn: send to the child, then wait on its await_message update
# ---------------------------------------------------------------------------


class _FakeChild:
    """Stands in for the ``AgentClient`` the activity builds for the child agent."""

    def __init__(self, result: MessageResult) -> None:
        self.result = result
        self.sent: list[str] = []
        self.awaited: list[str] = []

    async def _submit_message(self, msg_type: str, payload: dict, *, update_id: str) -> AgentMessageReply:
        self.sent.append(msg_type)
        return AgentMessageReply(
            turn_number=4,
            turn_id="child-turn-4",
            message_id="ours",
            disposition=MessageDisposition.OPENED,
        )

    async def await_message(self, message_id: str) -> MessageResult:
        self.awaited.append(message_id)
        return self.result


class _RecordingBackend:
    """An event-stream backend that records what activities publish."""

    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    @asynccontextmanager
    async def activity_publisher(self, *, batch_interval: timedelta):
        yield self

    def publish(self, event: AgentEvent) -> None:
        self.published.append(event)


def _child_result(**fields) -> MessageResult:
    return MessageResult(message_id="ours", turn_id="child-turn-4", turn_number=4, **fields)


def _run_turn(child: _FakeChild, monkeypatch, *, heartbeat_details=()):
    """Run the activity against ``child``; return ``(result, events published to the parent)``."""
    backend = _RecordingBackend()
    monkeypatch.setattr(subagent_activities, "AgentClient", lambda _client, _workflow_id: child)
    monkeypatch.setattr(agent_workflow, "activity_backend", lambda: backend)
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_details=list(heartbeat_details))
    req = RunSubagentTurnInput(
        child_workflow_id="child-wf",
        type="ask",
        payload={},
        from_offset=3,
        handle="aaaaaa-bbbbbb",
        agent_key="sample",
        parent_stream_context=TurnStreamContext(
            turn_id="parent-turn-1", turn_number=1, agent_id="aaaaaa"
        ),
    )
    activities = SubagentActivities(client=None)  # the child client is patched out
    result = asyncio.run(env.run(activities.run_subagent_turn, req))
    return result, backend.published


def test_returns_the_childs_result_and_where_its_turn_ended(monkeypatch):
    child = _FakeChild(_child_result(output={"text": "ours"}, stream_offset=17))

    result, published = _run_turn(child, monkeypatch)

    assert child.awaited == ["ours"]
    assert result.output == {"text": "ours"}
    assert result.turn_number == 4
    assert result.consumed_offset == 17
    # The dispatch marker lands on the parent's stream, through the configured backend.
    [dispatch] = [e.event for e in published]
    assert isinstance(dispatch, SubagentMessageSent)
    assert dispatch.subagent_turn == 4
    assert dispatch.from_offset == 3


def test_a_child_handler_error_fails_the_tool_call(monkeypatch):
    child = _FakeChild(_child_result(error="child blew up"))

    with pytest.raises(ApplicationError) as excinfo:
        _run_turn(child, monkeypatch)

    assert excinfo.value.type == "SubagentTurnError"
    assert "child blew up" in str(excinfo.value)
    # The parent closes the dispatch bracket on this turn number.
    assert excinfo.value.details == ({"subagent_turn": 4},)


def test_a_retry_after_the_send_waits_without_resending(monkeypatch):
    child = _FakeChild(_child_result(output={"text": "ours"}))
    memo = _TurnProgress(sent=True, turn_id="child-turn-4", turn_number=4, message_id="ours")

    result, published = _run_turn(child, monkeypatch, heartbeat_details=[memo.model_dump()])

    assert child.sent == []
    assert published == []
    assert child.awaited == ["ours"]
    assert result.output == {"text": "ours"}
