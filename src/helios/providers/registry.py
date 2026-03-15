"""Provider registry — factory for creating providers from config."""

from __future__ import annotations

from typing import Any

from helios.config.schema import ProviderConfig
from helios.providers.base import Provider


def create_provider(config: ProviderConfig) -> Any:
    """Create a provider instance from configuration."""
    provider_name = config.provider.lower()

    if provider_name == "ollama":
        from helios.providers.ollama import OllamaProvider
        return OllamaProvider(
            model=config.model,
            host=config.base_url or "http://localhost:11434",
        )

    elif provider_name == "groq":
        from helios.providers.groq import GroqProvider
        return GroqProvider(model=config.model, api_key=config.api_key)

    elif provider_name in ("openai_compatible", "openai-compatible"):
        from helios.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:1234/v1",
            api_key=config.api_key or "not-needed",
        )

    elif provider_name in ("llama_cpp", "llama-cpp", "gguf"):
        from helios.providers.llama_cpp import LlamaCppProvider
        if not config.model_path:
            raise ValueError("model_path is required for llama_cpp provider")
        return LlamaCppProvider(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Available: ollama, groq, openai_compatible, llama_cpp"
        )
