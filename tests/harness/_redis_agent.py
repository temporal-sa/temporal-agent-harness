"""A MODEL-FREE agent whose events go to Redis, with an activity that streams reply deltas.

Kept in its own module because the workflow sandbox re-imports a workflow's defining module.
``RedisEventStream`` is deliberately imported outside ``imports_passed_through``: sharing its
module with workflow code is AgentHarnessPlugin's job, and the test relies on that.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel
from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import (
        AgentConfig,
        MidTurn,
        ReplyDelta,
        TextMessage,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner
    from temporal_agent_harness.harness.stream_context import TurnStreamContext

from temporal_agent_harness.harness.redis_event_stream import RedisEventStream


class StreamWordsInput(BaseModel):
    text: str
    context: TurnStreamContext


@activity.defn
async def stream_words(input: StreamWordsInput) -> None:
    async with AgentWorkflowRunner.publisher_from_activity(input.context) as publisher:
        for word in input.text.split():
            publisher.publish(ReplyDelta(text=word))


@agent.defn
class RedisEchoAgent:
    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=RedisEventStream(),
            approval_policy_default=ToolApprovalPolicy.dangerously_skip_all(),
        )

    @agent.accepts(mid_turn=MidTurn.ENQUEUE)
    async def echo(self, message: TextMessage) -> TextReply:
        """Stream the message back word by word from an activity, then reply with it."""
        context = self._runner.current_stream_context
        assert context is not None
        await workflow.execute_activity(
            stream_words,
            StreamWordsInput(text=message.text, context=context),
            start_to_close_timeout=timedelta(seconds=10),
        )
        return TextReply(text=message.text)
