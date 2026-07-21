"""Surface provider length-capped model responses as run stop reasons.

Background — see issue bytedance/deer-flow#4271.

Some providers stop generation because the output budget is exhausted and
surface that through ``finish_reason='length'`` while still returning a
non-empty assistant message. DeerFlow should preserve that content for
audit, but it should not silently treat the run as an uncapped clean
completion when the provider has explicitly signaled truncation.

This middleware keeps that boundary narrow:
- it only marks a run-level stop reason when the final AIMessage is capped
  by length;
- it never rewrites the assistant content or reparses XML-like text into a
  tool call;
- it ignores any response that still carries tool-call intent or malformed
  tool-call metadata, so only terminal text responses can be marked capped.

"""

from __future__ import annotations

import threading
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares._bounded_dict import BoundedDict

MODEL_LENGTH_CAPPED_STOP_REASON = "model_length_capped"
_LENGTH_FINISH_REASONS = frozenset({"length"})


def _metadata_string(message: AIMessage, field_name: str) -> str | None:
    """Read a string metadata value from LangChain's common provider fields."""
    for container_name in ("response_metadata", "additional_kwargs"):
        container = getattr(message, container_name, None) or {}
        if not isinstance(container, dict):
            continue
        value = container.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


def _is_model_length_capped(message: AIMessage) -> bool:
    finish_reason = _metadata_string(message, "finish_reason")
    return finish_reason is not None and finish_reason.lower() in _LENGTH_FINISH_REASONS


def _has_tool_call_intent_or_error(message: AIMessage) -> bool:
    if message.tool_calls or getattr(message, "invalid_tool_calls", None):
        return True
    additional_kwargs = message.additional_kwargs or {}
    return bool(additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"))


class ModelLengthFinishReasonMiddleware(AgentMiddleware[AgentState]):
    """Record ``finish_reason=length`` for terminal text responses only.

    If the last AIMessage still carries tool-call intent, this middleware
    leaves it alone and lets the normal tool-handling path decide what to do.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._stop_reason: BoundedDict[str, str] = BoundedDict(1000)

    @staticmethod
    def _get_run_id(runtime: Runtime) -> str:
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict) and "run_id" in ctx:
            return str(ctx["run_id"])
        return str(id(runtime))

    def consume_stop_reason(self, run_id: str | None) -> str | None:
        """Pop the recorded reason for integrations that collect guard caps."""
        with self._lock:
            return self._stop_reason.pop(run_id, None)

    def _apply(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        last = messages[-1]
        if _has_tool_call_intent_or_error(last):
            return None

        if not _is_model_length_capped(last):
            return None

        run_id = self._get_run_id(runtime)
        with self._lock:
            self._stop_reason[run_id] = MODEL_LENGTH_CAPPED_STOP_REASON

        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            ctx.setdefault("stop_reason", MODEL_LENGTH_CAPPED_STOP_REASON)
        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)
