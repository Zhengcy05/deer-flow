"""Unit tests for ModelLengthFinishReasonMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.model_length_finish_reason_middleware import (
    MODEL_LENGTH_CAPPED_STOP_REASON,
    ModelLengthFinishReasonMiddleware,
)


def _runtime(run_id: str = "run-1"):
    runtime = MagicMock()
    runtime.context = {"thread_id": "thread-1", "run_id": run_id}
    return runtime


def test_finish_reason_length_records_stop_reason_without_rewriting_textual_tool_call():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    msg = AIMessage(
        content=('<tool_call><invoke name="write_file"><path>/mnt/user-data/outputs/report.md</path><content># partial'),
        tool_calls=[],
        invalid_tool_calls=[],
        response_metadata={"finish_reason": "length"},
    )

    result = mw._apply({"messages": [msg]}, runtime)

    assert result is None
    assert runtime.context["stop_reason"] == MODEL_LENGTH_CAPPED_STOP_REASON
    assert msg.content.startswith('<tool_call><invoke name="write_file">')
    assert msg.tool_calls == []
    assert msg.invalid_tool_calls == []


def test_finish_reason_stop_with_tool_call_example_in_prose_passes_through():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    msg = AIMessage(
        content=('Here is an example, not a real tool call:\n\n```xml\n<tool_call><invoke name="write_file"></invoke></tool_call>\n```'),
        response_metadata={"finish_reason": "stop"},
    )

    assert mw._apply({"messages": [msg]}, runtime) is None
    assert "stop_reason" not in runtime.context


def test_additional_kwargs_finish_reason_length_records_stop_reason():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    msg = AIMessage(
        content="partial",
        additional_kwargs={"finish_reason": "length"},
    )

    assert mw._apply({"messages": [msg]}, runtime) is None
    assert runtime.context["stop_reason"] == MODEL_LENGTH_CAPPED_STOP_REASON


def test_gemini_max_tokens_finish_reason_records_stop_reason():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    msg = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "MAX_TOKENS"},
    )

    assert mw._apply({"messages": [msg]}, runtime) is None
    assert runtime.context["stop_reason"] == MODEL_LENGTH_CAPPED_STOP_REASON


def test_anthropic_max_tokens_stop_reason_records_stop_reason():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    msg = AIMessage(
        content="partial",
        response_metadata={"stop_reason": "max_tokens"},
    )

    assert mw._apply({"messages": [msg]}, runtime) is None
    assert runtime.context["stop_reason"] == MODEL_LENGTH_CAPPED_STOP_REASON


def test_finish_reason_length_with_tool_calls_passes_through():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_write_1",
                "name": "write_file",
                "args": {"path": "/mnt/user-data/outputs/report.md", "content": "# partial"},
            }
        ],
        response_metadata={"finish_reason": "length"},
    )

    assert mw._apply({"messages": [msg]}, runtime) is None
    assert "stop_reason" not in runtime.context


def test_existing_stop_reason_is_not_overwritten():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()
    runtime.context["stop_reason"] = "token_capped"
    msg = AIMessage(content="partial", response_metadata={"finish_reason": "length"})

    assert mw._apply({"messages": [msg]}, runtime) is None
    assert runtime.context["stop_reason"] == "token_capped"


def test_non_ai_last_message_passes_through():
    mw = ModelLengthFinishReasonMiddleware()
    runtime = _runtime()

    assert mw._apply({"messages": [HumanMessage(content="hello")]}, runtime) is None
    assert "stop_reason" not in runtime.context
