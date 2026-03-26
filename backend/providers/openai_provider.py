"""
OpenAI-compatible AI Provider.

Supports: OpenAI, Azure OpenAI, Ollama, LocalAI, LM Studio, and any
service that implements the OpenAI chat completions API.

Live audio: uses OpenAI Realtime API when available, otherwise falls back
to text-only mode (the caller can supply TTS/STT externally).
"""

import os
import json
import asyncio
from typing import Any, AsyncIterator, List, Optional

from dotenv import load_dotenv

from .base import (
    BaseTextProvider,
    BaseLiveProvider,
    BaseLiveSession,
    ContentConfig,
    FunctionResponseData,
    LiveConfig,
    LiveResponse,
    StreamChunk,
    ToolCall,
    TranscriptionData,
)

load_dotenv()


def _get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Lazy-import openai and return an AsyncOpenAI client."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for the OpenAI provider. "
            "Install it with: pip install openai>=1.0.0"
        )

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    url = base_url or os.getenv("OPENAI_BASE_URL")

    kwargs = {"api_key": key}
    if url:
        kwargs["base_url"] = url

    return AsyncOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Text Provider
# ---------------------------------------------------------------------------


class OpenAITextProvider(BaseTextProvider):
    """Standard chat completion via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client = _get_openai_client(api_key, base_url)

    async def generate_content(
        self,
        contents: Any,
        model: str,
        config: Optional[ContentConfig] = None,
    ) -> Any:
        """Single-shot chat completion.

        `contents` can be:
        - a plain string (converted to a single user message)
        - a list of dicts (OpenAI messages format)
        """
        messages = self._to_messages(contents, config)
        kwargs = self._build_kwargs(model, config)

        response = await self.client.chat.completions.create(
            messages=messages,
            stream=False,
            **kwargs,
        )
        return response

    async def generate_content_stream(
        self,
        contents: Any,
        model: str,
        config: Optional[ContentConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion, yielding StreamChunk objects."""
        messages = self._to_messages(contents, config)
        kwargs = self._build_kwargs(model, config)

        stream = await self.client.chat.completions.create(
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield StreamChunk(text=delta.content, is_thought=False)

    # -- Helpers ---------------------------------------------------------------

    def _to_messages(
        self, contents: Any, config: Optional[ContentConfig] = None
    ) -> list:
        """Normalize contents into OpenAI messages list."""
        messages = []

        # System instruction
        if config and config.system_instruction:
            messages.append({"role": "system", "content": config.system_instruction})

        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            messages.extend(contents)
        else:
            messages.append({"role": "user", "content": str(contents)})

        return messages

    def _build_kwargs(self, model: str, config: Optional[ContentConfig] = None) -> dict:
        kwargs = {"model": model}
        if config:
            kwargs["temperature"] = config.temperature

            if config.tools:
                kwargs["tools"] = self._convert_tools(config.tools)
        
        if config and config.extras:
            kwargs.update(config.extras)

        return kwargs

    @staticmethod
    def _convert_tools(tools: list) -> list:
        """Best-effort conversion from Gemini-style tool dicts to OpenAI format.

        Falls through if tools are already in OpenAI format.
        """
        openai_tools = []
        for tool in tools:
            # If it's already OpenAI-formatted, pass through
            if isinstance(tool, dict) and "type" in tool and tool["type"] == "function":
                openai_tools.append(tool)
                continue

            # Gemini format: {"function_declarations": [...]}
            if isinstance(tool, dict) and "function_declarations" in tool:
                for fd in tool["function_declarations"]:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": fd["name"],
                            "description": fd.get("description", ""),
                            "parameters": _gemini_params_to_json_schema(
                                fd.get("parameters", {})
                            ),
                        },
                    })
                continue

            # Skip google_search and other non-function tools
            if isinstance(tool, dict) and "google_search" in tool:
                continue

        return openai_tools


# ---------------------------------------------------------------------------
# Live Provider (OpenAI Realtime API — text-only fallback for now)
# ---------------------------------------------------------------------------


class OpenAILiveSession(BaseLiveSession):
    """Text-only live session using OpenAI chat completions.

    This is a polling-based fallback for providers that lack a native
    real-time audio API. Audio capture / TTS is handled externally by
    the caller (AudioLoop).
    """

    def __init__(self, client, model: str, config: LiveConfig):
        self._client = client
        self._model = model
        self._config = config
        self._messages: list = []
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._tools = None

        if config.system_instruction:
            self._messages.append({
                "role": "system",
                "content": config.system_instruction,
            })

        if config.tools:
            self._tools = OpenAITextProvider._convert_tools(config.tools)

    async def send(self, input: Any, end_of_turn: bool = False):
        """Send text input and generate a response.

        For audio data dicts, this is a no-op (text-only fallback).
        """
        # Skip audio payloads and image payloads
        if isinstance(input, dict):
            if "data" in input and "mime_type" in input:
                return  # Audio/image frame — skip in text mode

        if isinstance(input, str) and input.strip():
            self._messages.append({"role": "user", "content": input})

            if end_of_turn:
                await self._generate_and_queue()

    async def _generate_and_queue(self):
        """Call OpenAI and queue the response."""
        kwargs = {"model": self._model, "messages": self._messages}
        if self._tools:
            kwargs["tools"] = self._tools

        try:
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0] if response.choices else None

            if choice:
                msg = choice.message

                # Tool calls
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        args = {}
                        if tc.function.arguments:
                            try:
                                args = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                args = {"raw": tc.function.arguments}
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            args=args,
                        ))

                    # Append assistant message to history
                    self._messages.append(msg.model_dump())

                    await self._response_queue.put(
                        LiveResponse(tool_calls=tool_calls)
                    )
                elif msg.content:
                    self._messages.append({
                        "role": "assistant",
                        "content": msg.content,
                    })
                    # No audio — send transcription-like event
                    await self._response_queue.put(
                        LiveResponse(
                            output_transcription=TranscriptionData(
                                text=msg.content, is_input=False
                            ),
                        )
                    )
        except Exception as e:
            print(f"[OpenAI Provider] Error generating response: {e}")

    def receive(self) -> "OpenAILiveResponseIterator":
        return OpenAILiveResponseIterator(self._response_queue)

    async def send_tool_response(self, function_responses: List[FunctionResponseData]):
        """Send tool results back to OpenAI and generate a follow-up."""
        for fr in function_responses:
            self._messages.append({
                "role": "tool",
                "tool_call_id": fr.id,
                "content": json.dumps(fr.response) if isinstance(fr.response, dict) else str(fr.response),
            })
        await self._generate_and_queue()


class OpenAILiveResponseIterator:
    """Async iteration over queued LiveResponse objects."""

    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self) -> LiveResponse:
        return await self._queue.get()


class OpenAILiveProvider(BaseLiveProvider):
    """Live provider using OpenAI-compatible API (text-only fallback)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client = _get_openai_client(api_key, base_url)

    async def connect(self, model: str, config: LiveConfig) -> "OpenAILiveSessionCtx":
        return OpenAILiveSessionCtx(self.client, model, config)


class OpenAILiveSessionCtx:
    """Async context manager for OpenAI live sessions."""

    def __init__(self, client, model: str, config: LiveConfig):
        self._client = client
        self._model = model
        self._config = config

    async def __aenter__(self) -> OpenAILiveSession:
        return OpenAILiveSession(self._client, self._model, self._config)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # No persistent connection to close


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemini_params_to_json_schema(params: dict) -> dict:
    """Convert Gemini-style parameter definitions to JSON Schema.

    Gemini uses {"type": "OBJECT", "properties": {...}, "required": [...]}
    OpenAI uses standard JSON Schema.
    """
    if not params:
        return {"type": "object", "properties": {}}

    type_map = {
        "STRING": "string",
        "INTEGER": "integer",
        "NUMBER": "number",
        "BOOLEAN": "boolean",
        "OBJECT": "object",
        "ARRAY": "array",
    }

    schema = {
        "type": type_map.get(params.get("type", "OBJECT"), "object"),
    }

    if "properties" in params:
        schema["properties"] = {}
        for name, prop in params["properties"].items():
            prop_schema = {
                "type": type_map.get(prop.get("type", "STRING"), "string"),
            }
            if "description" in prop:
                prop_schema["description"] = prop["description"]
            schema["properties"][name] = prop_schema

    if "required" in params:
        schema["required"] = params["required"]

    return schema
