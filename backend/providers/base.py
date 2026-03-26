"""
Abstract base classes for AI providers.

Two provider types:
- BaseTextProvider: Standard request/response (for cad_agent, web_agent)
- BaseLiveProvider / BaseLiveSession: Real-time audio streaming (for ada.py AudioLoop)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


# ---------------------------------------------------------------------------
# Provider-agnostic data classes
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    text: Optional[str] = None
    is_thought: bool = False


@dataclass
class ToolCall:
    """A tool/function call from the model."""
    id: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionResponseData:
    """A function response to send back to the model."""
    id: str
    name: str
    response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentConfig:
    """Provider-agnostic generation config."""
    system_instruction: Optional[str] = None
    temperature: float = 1.0
    tools: Optional[List[Any]] = None
    include_thinking: bool = False
    # Provider-specific extras (passed through)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveConfig:
    """Configuration for a live audio session."""
    system_instruction: Optional[str] = None
    tools: Optional[List[Any]] = None
    response_modalities: Optional[List[str]] = None
    voice_name: Optional[str] = None
    enable_input_transcription: bool = True
    enable_output_transcription: bool = True
    # Provider-specific extras (passed through)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionData:
    """Transcription event from a live session."""
    text: str
    is_input: bool  # True = user speech, False = model speech


@dataclass
class LiveResponse:
    """A response event from a live session."""
    audio_data: Optional[bytes] = None
    input_transcription: Optional[TranscriptionData] = None
    output_transcription: Optional[TranscriptionData] = None
    tool_calls: Optional[List[ToolCall]] = None
    # Raw server content for provider-specific handling
    raw: Any = None


# ---------------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------------

class BaseTextProvider(ABC):
    """Provider for standard request/response text generation."""

    @abstractmethod
    async def generate_content(
        self,
        contents: Any,
        model: str,
        config: Optional[ContentConfig] = None,
    ) -> Any:
        """Single-shot text generation. Returns the full response."""
        ...

    @abstractmethod
    async def generate_content_stream(
        self,
        contents: Any,
        model: str,
        config: Optional[ContentConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming text generation with optional thinking support."""
        ...


class BaseLiveSession(ABC):
    """Represents an active real-time session with the AI model."""

    @abstractmethod
    async def send(self, input: Any, end_of_turn: bool = False):
        """Send audio/text/image data to the session."""
        ...

    @abstractmethod
    def receive(self) -> AsyncIterator[LiveResponse]:
        """Receive events from the session (audio, transcription, tool calls)."""
        ...

    @abstractmethod
    async def send_tool_response(self, function_responses: List[FunctionResponseData]):
        """Send tool/function responses back to the model."""
        ...


class BaseLiveProvider(ABC):
    """Provider for real-time audio/video streaming sessions."""

    @abstractmethod
    async def connect(self, model: str, config: LiveConfig) -> BaseLiveSession:
        """Open a live session and return it as an async context manager."""
        ...
