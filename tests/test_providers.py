"""
Tests for the pluggable AI provider layer.
"""

import asyncio
from types import SimpleNamespace

import pytest

from providers.base import (
    BaseLiveProvider,
    BaseLiveSession,
    BaseTextProvider,
    ContentConfig,
    FunctionResponseData,
    LiveConfig,
)
from providers.factory import get_live_provider, get_provider_name, get_text_provider
from providers.gemini_provider import GeminiTextProvider
from providers.openai_provider import (
    OpenAILiveSession,
    OpenAITextProvider,
    _gemini_params_to_json_schema,
)


class _AsyncIterator:
    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class _FakeOpenAICompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class TestBaseProviderInterfaces:
    def test_base_text_provider_is_abstract(self):
        with pytest.raises(TypeError):
            BaseTextProvider()

    def test_base_live_provider_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLiveProvider()

    def test_base_live_session_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLiveSession()


class TestProviderFactory:
    def test_get_provider_name_defaults_to_gemini(self, monkeypatch):
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        assert get_provider_name() == "gemini"

    def test_get_provider_name_rejects_invalid_values(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "invalid")
        with pytest.raises(ValueError, match="Unknown AI_PROVIDER"):
            get_provider_name()

    def test_get_text_provider_returns_gemini(self, monkeypatch):
        import providers.gemini_provider as gemini_provider

        class FakeGeminiProvider:
            pass

        monkeypatch.setattr(gemini_provider, "GeminiTextProvider", FakeGeminiProvider)
        provider = get_text_provider("gemini")
        assert isinstance(provider, FakeGeminiProvider)

    def test_get_text_provider_returns_openai(self, monkeypatch):
        import providers.openai_provider as openai_provider

        class FakeOpenAIProvider:
            pass

        monkeypatch.setattr(openai_provider, "OpenAITextProvider", FakeOpenAIProvider)
        provider = get_text_provider("openai")
        assert isinstance(provider, FakeOpenAIProvider)

    def test_get_live_provider_returns_openai(self, monkeypatch):
        import providers.openai_provider as openai_provider

        class FakeOpenAILiveProvider:
            pass

        monkeypatch.setattr(openai_provider, "OpenAILiveProvider", FakeOpenAILiveProvider)
        provider = get_live_provider("openai")
        assert isinstance(provider, FakeOpenAILiveProvider)


class TestGeminiTextProvider:
    @pytest.mark.asyncio
    async def test_generate_content_stream_maps_thoughts_and_text(self):
        thought_part = SimpleNamespace(text="thinking", thought=True)
        text_part = SimpleNamespace(text="answer", thought=False)
        chunk = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[thought_part, text_part])
                )
            ]
        )
        fake_stream = _AsyncIterator([chunk])

        async def fake_generate_content_stream(**kwargs):
            return fake_stream

        fake_models = SimpleNamespace(
            generate_content_stream=fake_generate_content_stream
        )
        provider = object.__new__(GeminiTextProvider)
        provider.client = SimpleNamespace(aio=SimpleNamespace(models=fake_models))

        chunks = [
            stream_chunk
            async for stream_chunk in provider.generate_content_stream(
                contents="prompt",
                model="gemini-test",
                config=ContentConfig(include_thinking=True),
            )
        ]

        assert [item.text for item in chunks] == ["thinking", "answer"]
        assert [item.is_thought for item in chunks] == [True, False]


class TestOpenAITextProvider:
    @pytest.mark.asyncio
    async def test_generate_content_stream_reads_openai_delta_chunks(self):
        response = _AsyncIterator(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hello "))]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]
                ),
            ]
        )
        completions = _FakeOpenAICompletions(response)
        provider = object.__new__(OpenAITextProvider)
        provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        chunks = [
            stream_chunk
            async for stream_chunk in provider.generate_content_stream(
                contents="prompt",
                model="gpt-4o",
                config=ContentConfig(system_instruction="sys"),
            )
        ]

        assert [item.text for item in chunks] == ["hello ", "world"]
        assert completions.calls[0]["messages"][0]["role"] == "system"
        assert completions.calls[0]["stream"] is True

    def test_convert_tools_from_gemini_function_declarations(self):
        converted = OpenAITextProvider._convert_tools(
            [
                {
                    "function_declarations": [
                        {
                            "name": "lookup_weather",
                            "description": "Look up weather",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "city": {
                                        "type": "STRING",
                                        "description": "City name",
                                    }
                                },
                                "required": ["city"],
                            },
                        }
                    ]
                }
            ]
        )

        assert converted == [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ]


class TestOpenAILiveSession:
    @pytest.mark.asyncio
    async def test_text_only_live_session_returns_transcription(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello from OpenAI",
                        tool_calls=None,
                    )
                )
            ]
        )
        completions = _FakeOpenAICompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        session = OpenAILiveSession(
            client,
            "gpt-4o",
            LiveConfig(system_instruction="You are ADA."),
        )

        await session.send("Hi", end_of_turn=True)
        reply = await asyncio.wait_for(session.receive().__anext__(), timeout=1)

        assert reply.output_transcription.text == "Hello from OpenAI"
        assert completions.calls[0]["messages"][0]["role"] == "system"
        assert completions.calls[0]["messages"][1]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_send_tool_response_appends_tool_message(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Done",
                        tool_calls=None,
                    )
                )
            ]
        )
        completions = _FakeOpenAICompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        session = OpenAILiveSession(client, "gpt-4o", LiveConfig())
        session._messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1"}],
            }
        )

        await session.send_tool_response(
            [
                FunctionResponseData(
                    id="call_1",
                    name="lookup_weather",
                    response={"result": "sunny"},
                )
            ]
        )

        assert session._messages[-2]["role"] == "tool"
        assert session._messages[-2]["tool_call_id"] == "call_1"
        reply = await asyncio.wait_for(session.receive().__anext__(), timeout=1)
        assert reply.output_transcription.text == "Done"


class TestSchemaConversion:
    def test_gemini_params_to_json_schema(self):
        schema = _gemini_params_to_json_schema(
            {
                "type": "OBJECT",
                "properties": {
                    "enabled": {
                        "type": "BOOLEAN",
                        "description": "Enable the feature",
                    }
                },
                "required": ["enabled"],
            }
        )

        assert schema == {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Enable the feature",
                }
            },
            "required": ["enabled"],
        }
