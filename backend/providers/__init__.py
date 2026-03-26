"""
Pluggable AI Provider Abstraction Layer for A.D.A.

Supports multiple AI backends: Gemini, OpenAI-compatible (OpenAI, Azure, Ollama, LocalAI).
"""

from .base import (
    BaseLiveProvider,
    BaseLiveSession,
    BaseTextProvider,
    FunctionResponseData,
    LiveResponse,
    ToolCall,
)
from .factory import get_live_provider, get_provider_name, get_text_provider

__all__ = [
    "BaseTextProvider",
    "BaseLiveProvider", 
    "BaseLiveSession",
    "LiveResponse",
    "ToolCall",
    "FunctionResponseData",
    "get_provider_name",
    "get_text_provider",
    "get_live_provider",
]
