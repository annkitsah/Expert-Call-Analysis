"""Thin LLM layer.

Structured output uses *forced tool use*: the model must answer by calling a tool
whose input schema is generated from a Pydantic model, and the result is validated
against that model. Anything else the model might say is ignored.

Services depend on the ``LLMClient`` protocol, so tests inject a fake and the
provider can be swapped without touching business logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The model call failed or returned something unusable (HTTP 502)."""


class LLMUnavailableError(LLMError):
    """The LLM is not configured (HTTP 503)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]


class LLMClient(Protocol):
    model: str

    async def structured(
        self,
        *,
        instructions: str,
        context: str,
        messages: list[dict[str, str]],
        tool: ToolSpec,
    ) -> dict[str, Any]:
        """Return the tool-call arguments the model produced."""
        ...


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for ``model`` with ``$ref``s inlined and cosmetic titles removed."""
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            extras = {k: resolve(v) for k, v in node.items() if k != "$ref"}
            return {**target, **extras}
        # "title" as a *string* is cosmetic JSON-Schema metadata; a property named
        # "title" has a dict value and must be kept.
        return {k: resolve(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}

    resolved = resolve(schema)
    assert isinstance(resolved, dict)
    return resolved


def make_tool(name: str, description: str, model: type[BaseModel]) -> ToolSpec:
    return ToolSpec(name=name, description=description, schema=tool_schema(model))


class GroqLLM:
    """Groq Chat Completions through its OpenAI-compatible API.

    Groq exposes an OpenAI-compatible endpoint, so the existing structured tool-calling
    implementation can use the standard ``openai`` Python client with Groq's base URL.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        base_url: str = "https://api.groq.com/openai/v1",
        max_retries: int = 3,
        http_client: Any = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        # The SDK retries 429/5xx/connection errors with backoff. ``http_client`` is passed straight
        # through to the SDK so tests can exercise the real request/response handling against a mock
        # transport, without a network call or a real key.
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
            http_client=http_client,
        )

    async def structured(
        self,
        *,
        instructions: str,
        context: str,
        messages: list[dict[str, str]],
        tool: ToolSpec,
    ) -> dict[str, Any]:
        system: list[dict[str, str]] = [{"role": "system", "content": instructions}]
        if context:
            # A separate system message so instructions and the (much larger, more stable)
            # transcript corpus stay distinct; the provider can cache stable prompt prefixes.
            system.append({"role": "system", "content": context})
        function = {"name": tool.name, "description": tool.description, "parameters": tool.schema}
        payload = [{"type": "function", "function": function}]
        try:
            response = await self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                max_completion_tokens=self._max_tokens,
                reasoning_effort="low",
                messages=system + messages,
                tools=payload,
                tool_choice={"type": "function", "function": {"name": tool.name}},
            )
        except openai.AuthenticationError as exc:
            raise LLMUnavailableError("Groq rejected the API key. Check GROQ_API_KEY.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("The Groq API is rate limiting requests. Wait a moment and retry.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the Groq API. Check your network connection.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"Groq API error {exc.status_code}: {exc.message}") from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        log.info(
            "llm call tool=%s model=%s in=%s out=%s finish=%s",
            tool.name,
            self.model,
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            choice.finish_reason,
        )
        if choice.finish_reason == "length":
            raise LLMError("The model's response was cut off. Increase MAX_OUTPUT_TOKENS and retry.")
        for call in choice.message.tool_calls or []:
            if call.function.name == tool.name:
                try:
                    parsed = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise LLMError("The model's tool arguments were not valid JSON.") from exc
                if not isinstance(parsed, dict):
                    raise LLMError("The model's tool arguments were not a JSON object.")
                return parsed
        raise LLMError("The model did not return the expected structured result.")


async def structured_validated(
    llm: LLMClient,
    schema_model: type[T],
    *,
    instructions: str,
    context: str,
    messages: list[dict[str, str]],
    tool: ToolSpec,
    attempts: int = 2,
) -> T:
    """Call the model and validate against ``schema_model``; retry once on a malformed result."""
    last_error: ValidationError | None = None
    for attempt in range(1, attempts + 1):
        raw = await llm.structured(instructions=instructions, context=context, messages=messages, tool=tool)
        try:
            return schema_model.model_validate(raw)
        except ValidationError as exc:
            last_error = exc
            log.warning("tool=%s attempt=%d returned invalid structure: %s", tool.name, attempt, exc)
    raise LLMError("The model returned a malformed result twice. Please retry.") from last_error


# Backwards-compatible alias for tests or external imports that used the old class name.
OpenAILLM = GroqLLM
