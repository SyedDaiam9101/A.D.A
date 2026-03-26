"""
Gemini AI Provider — wraps the google-genai SDK.

Implements both TextProvider (for cad_agent, web_agent) and
LiveProvider (for ada.py real-time audio sessions).
"""

import os
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


class GeminiTextProvider(BaseTextProvider):
    """Text generation using the Google GenAI SDK (v1beta)."""

    def __init__(self, api_key: Optional[str] = None):
        from google import genai as _genai

        self._genai = _genai
        key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = _genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=key,
        )

    # -- Standard (single-shot) ------------------------------------------------

    async def generate_content(
        self,
        contents: Any,
        model: str,
        config: Optional[ContentConfig] = None,
    ) -> Any:
        """Single-shot generation (returns raw Gemini response)."""
        from google.genai import types

        gen_config = self._build_config(config)
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        )
        return response

    # -- Streaming -------------------------------------------------------------

    async def generate_content_stream(
        self,
        contents: Any,
        model: str,
        config: Optional[ContentConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming generation, yielding StreamChunk objects."""
        gen_config = self._build_config(config)
        stream = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=gen_config,
        )
        async for chunk in stream:
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    yield StreamChunk(
                        text=part.text,
                        is_thought=bool(getattr(part, "thought", False)),
                    )

    # -- Helpers ---------------------------------------------------------------

    def _build_config(self, config: Optional[ContentConfig] = None):
        """Convert ContentConfig → google.genai.types.GenerateContentConfig."""
        from google.genai import types

        if config is None:
            return None

        kwargs = {
            "temperature": config.temperature,
        }
        if config.system_instruction:
            kwargs["system_instruction"] = config.system_instruction
        if config.tools:
            kwargs["tools"] = config.tools
        if config.include_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

        # Pass through any Gemini-specific extras
        kwargs.update(config.extras)

        return types.GenerateContentConfig(**kwargs)


# ---------------------------------------------------------------------------
# Live / Real-time Audio
# ---------------------------------------------------------------------------


class GeminiLiveSession(BaseLiveSession):
    """Wraps a google.genai live session object."""

    def __init__(self, raw_session):
        self._session = raw_session

    async def send(self, input: Any, end_of_turn: bool = False):
        await self._session.send(input=input, end_of_turn=end_of_turn)

    def receive(self) -> AsyncIterator[LiveResponse]:
        """Returns an async iterator of LiveResponse.

        The caller should iterate like:
            turn = session.receive()
            async for response in turn:
                ...
        """
        return _GeminiLiveResponseIterator(self._session.receive())

    async def send_tool_response(self, function_responses: List[FunctionResponseData]):
        """Convert generic FunctionResponseData → google.genai FunctionResponse."""
        from google.genai import types

        gemini_responses = [
            types.FunctionResponse(
                id=fr.id,
                name=fr.name,
                response=fr.response,
            )
            for fr in function_responses
        ]
        await self._session.send_tool_response(function_responses=gemini_responses)


class _GeminiLiveResponseIterator:
    """Adapts the Gemini Live turn iterator to yield LiveResponse objects."""

    def __init__(self, turn):
        self._turn = turn

    def __aiter__(self):
        return self

    async def __anext__(self) -> LiveResponse:
        try:
            response = await self._turn.__anext__()
        except StopAsyncIteration:
            raise

        live_resp = LiveResponse(raw=response)

        # Audio data
        if data := getattr(response, "data", None):
            live_resp.audio_data = data

        # Transcription
        sc = getattr(response, "server_content", None)
        if sc:
            if it := getattr(sc, "input_transcription", None):
                if it.text:
                    live_resp.input_transcription = TranscriptionData(
                        text=it.text, is_input=True
                    )
            if ot := getattr(sc, "output_transcription", None):
                if ot.text:
                    live_resp.output_transcription = TranscriptionData(
                        text=ot.text, is_input=False
                    )

        # Tool calls
        tc = getattr(response, "tool_call", None)
        if tc and tc.function_calls:
            live_resp.tool_calls = [
                ToolCall(id=fc.id, name=fc.name, args=dict(fc.args) if fc.args else {})
                for fc in tc.function_calls
            ]

        return live_resp


class GeminiLiveProvider(BaseLiveProvider):
    """Opens real-time Gemini Live sessions."""

    def __init__(self, api_key: Optional[str] = None):
        from google import genai as _genai

        key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = _genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=key,
        )
        self._genai = _genai

    async def connect(self, model: str, config: LiveConfig) -> "GeminiLiveSessionCtx":
        """Returns an async context manager that yields a GeminiLiveSession."""
        from google.genai import types

        gemini_config = self._build_live_config(config)
        return GeminiLiveSessionCtx(self.client, model, gemini_config)

    def _build_live_config(self, config: LiveConfig):
        from google.genai import types

        kwargs = {}
        if config.response_modalities:
            kwargs["response_modalities"] = config.response_modalities
        if config.system_instruction:
            kwargs["system_instruction"] = config.system_instruction
        if config.tools:
            kwargs["tools"] = config.tools
        if config.enable_input_transcription:
            kwargs["input_audio_transcription"] = {}
        if config.enable_output_transcription:
            kwargs["output_audio_transcription"] = {}
        if config.voice_name:
            kwargs["speech_config"] = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.voice_name
                    )
                )
            )
        kwargs.update(config.extras)
        return types.LiveConnectConfig(**kwargs)


class GeminiLiveSessionCtx:
    """Async context manager wrapper around Gemini's live.connect()."""

    def __init__(self, client, model, config):
        self._client = client
        self._model = model
        self._config = config
        self._ctx = None

    async def __aenter__(self) -> GeminiLiveSession:
        self._ctx = self._client.aio.live.connect(
            model=self._model, config=self._config
        )
        raw_session = await self._ctx.__aenter__()
        return GeminiLiveSession(raw_session)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._ctx:
            return await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
