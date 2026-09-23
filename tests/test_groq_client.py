"""Exercises GroqLLM through the real OpenAI-compatible SDK against a mock HTTP transport (no network, no key)."""

from __future__ import annotations

import json
from typing import Any

import pytest

try:  # newer SDK releases are built on httpx2, older ones on httpx; use whichever the SDK expects
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx  # type: ignore[no-redef]

from app.analysis import GUIDE_TOOL
from app.llm import LLMError, LLMUnavailableError, GroqLLM
from app.qa import QA_TOOL


def completion(tool_calls: list[dict[str, Any]] | None, finish_reason: str = "tool_calls") -> dict[str, Any]:
    return {
        "id": "chatcmpl_test", "object": "chat.completion", "created": 0, "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def tool_call(tool_name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(arguments)}}


def make_llm(handler: Any) -> GroqLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqLLM(api_key="test-key", model="test-model", max_tokens=1234, timeout_s=5, base_url="https://api.groq.com/openai/v1", max_retries=0, http_client=client)


async def test_request_shape_and_tool_call_arguments_are_returned() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        answer = {"found": False, "answer": "n/a", "citations": []}
        return httpx.Response(200, json=completion([tool_call(QA_TOOL.name, answer)]))

    out = await make_llm(handler).structured(
        instructions="INSTR", context="CORPUS", messages=[{"role": "user", "content": "hi"}], tool=QA_TOOL
    )
    assert out == {"found": False, "answer": "n/a", "citations": []}

    body = seen["body"]
    assert seen["url"].endswith("/chat/completions") and seen["headers"]["authorization"] == "Bearer test-key"
    assert body["model"] == "test-model" and body["max_completion_tokens"] == 1234
    assert body["reasoning_effort"] == "low"
    assert body["tool_choice"] == {"type": "function", "function": {"name": QA_TOOL.name}}  # tool use is forced
    assert body["tools"][0]["function"]["parameters"]["type"] == "object"
    assert body["messages"][0] == {"role": "system", "content": "INSTR"}
    assert body["messages"][1] == {"role": "system", "content": "CORPUS"}
    assert body["messages"][2] == {"role": "user", "content": "hi"}


async def test_empty_context_omits_second_system_message() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion([tool_call(GUIDE_TOOL.name, {"answers": []})]))

    await make_llm(handler).structured(instructions="I", context="", messages=[{"role": "user", "content": "x"}], tool=GUIDE_TOOL)
    assert len(seen["body"]["messages"]) == 2  # system + user, no empty context block


@pytest.mark.parametrize(
    ("status", "exc", "fragment"),
    [(401, LLMUnavailableError, "API key"), (429, LLMError, "rate limiting"), (500, LLMError, "500")],
)
async def test_http_errors_map_to_actionable_messages(status: int, exc: type[Exception], fragment: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope", "type": "api_error"}})

    with pytest.raises(exc, match=fragment):
        await make_llm(handler).structured(instructions="i", context="c", messages=[{"role": "user", "content": "x"}], tool=QA_TOOL)


async def test_truncated_and_missing_tool_output_are_errors() -> None:
    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(None, finish_reason="length"))

    def no_tool(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(None, finish_reason="stop"))

    args = {"instructions": "i", "context": "c", "messages": [{"role": "user", "content": "x"}], "tool": QA_TOOL}
    with pytest.raises(LLMError, match="cut off"):
        await make_llm(truncated).structured(**args)
    with pytest.raises(LLMError, match="expected structured result"):
        await make_llm(no_tool).structured(**args)


async def test_malformed_tool_arguments_are_errors() -> None:
    def bad_json(request: httpx.Request) -> httpx.Response:
        call = {"id": "c1", "type": "function", "function": {"name": QA_TOOL.name, "arguments": "not json"}}
        return httpx.Response(200, json=completion([call]))

    def non_object(request: httpx.Request) -> httpx.Response:
        call = {"id": "c1", "type": "function", "function": {"name": QA_TOOL.name, "arguments": "[1, 2]"}}
        return httpx.Response(200, json=completion([call]))

    args = {"instructions": "i", "context": "c", "messages": [{"role": "user", "content": "x"}], "tool": QA_TOOL}
    with pytest.raises(LLMError, match="not valid JSON"):
        await make_llm(bad_json).structured(**args)
    with pytest.raises(LLMError, match="not a JSON object"):
        await make_llm(non_object).structured(**args)


async def test_connection_failure_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure", request=request)

    with pytest.raises(LLMError, match="Could not reach"):
        await make_llm(handler).structured(instructions="i", context="c", messages=[{"role": "user", "content": "x"}], tool=QA_TOOL)
