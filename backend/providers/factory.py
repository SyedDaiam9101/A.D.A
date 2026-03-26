"""
Provider factory helpers.

Environment variables:
    AI_PROVIDER       = "gemini" | "openai"   (default: "gemini")
    GEMINI_API_KEY    = ...                    (used when AI_PROVIDER=gemini)
    OPENAI_API_KEY    = ...                    (used when AI_PROVIDER=openai)
    OPENAI_BASE_URL   = ...                    (optional, for Ollama/LocalAI/Azure)
"""

import os

from dotenv import load_dotenv

from .base import BaseLiveProvider, BaseTextProvider

load_dotenv()

SUPPORTED_PROVIDERS = {"gemini", "openai"}


def get_provider_name(provider_name: str = None) -> str:
    """Return the normalized provider name."""
    name = (provider_name or os.getenv("AI_PROVIDER", "gemini")).lower().strip()
    if name not in SUPPORTED_PROVIDERS:
        supported = "', '".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unknown AI_PROVIDER '{name}'. Supported: '{supported}'."
        )
    return name


def get_text_provider(provider_name: str = None) -> BaseTextProvider:
    """Return a text-generation provider instance."""
    name = get_provider_name(provider_name)

    if name == "gemini":
        from .gemini_provider import GeminiTextProvider

        return GeminiTextProvider()

    if name == "openai":
        from .openai_provider import OpenAITextProvider

        return OpenAITextProvider()

    raise AssertionError(f"Unhandled provider '{name}'")


def get_live_provider(provider_name: str = None) -> BaseLiveProvider:
    """Return a live-session provider instance."""
    name = get_provider_name(provider_name)

    if name == "gemini":
        from .gemini_provider import GeminiLiveProvider

        return GeminiLiveProvider()

    if name == "openai":
        from .openai_provider import OpenAILiveProvider

        return OpenAILiveProvider()

    raise AssertionError(f"Unhandled provider '{name}'")
