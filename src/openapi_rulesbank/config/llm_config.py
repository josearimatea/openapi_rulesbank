# src/app/config/llm_config.py
"""
LLM configuration and initialization.
Creates and exposes the global LLM instance.
Depends on settings (e.g., provider, API key).

OpenAI and OpenRouter both speak the OpenAI chat-completions protocol, so a
single ChatOpenAI client serves both: RULESBANK_PROVIDER only decides which key,
base URL and headers it is built with.
"""

from typing import Any

from langchain_openai import ChatOpenAI

from .settings import (
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT,
    MODEL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    PROVIDER,
    REASONING_EFFORT,
    TEMPERATURE,
)


def _openrouter_kwargs() -> dict[str, Any]:
    """Key, base URL, headers and reasoning payload for the OpenRouter client."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not found in .env file "
            "(or set RULESBANK_PROVIDER=openai to use OpenAI directly)."
        )

    headers = {"X-Title": OPENROUTER_APP_NAME} if OPENROUTER_APP_NAME else {}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL

    kwargs: dict[str, Any] = {
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
    }
    if headers:
        kwargs["default_headers"] = headers

    # OpenRouter takes reasoning as a body field rather than the OpenAI-style
    # top-level `reasoning_effort`, and "none" is how this project spells
    # "reasoning off" (see settings.REASONING_EFFORT).
    if REASONING_EFFORT:
        reasoning = (
            {"enabled": False}
            if REASONING_EFFORT == "none"
            else {"effort": REASONING_EFFORT}
        )
        kwargs["extra_body"] = {"reasoning": reasoning}
    return kwargs


def _openai_kwargs() -> dict[str, Any]:
    """Key and reasoning parameter for the direct-OpenAI client."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file "
            "(or set RULESBANK_PROVIDER=openrouter to use OpenRouter)."
        )

    kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY}
    # OpenAI has no "reasoning off" switch — the parameter is simply omitted for
    # models without a reasoning mode.
    if REASONING_EFFORT and REASONING_EFFORT != "none":
        kwargs["reasoning_effort"] = REASONING_EFFORT
    return kwargs


if PROVIDER == "openrouter":
    _provider_kwargs = _openrouter_kwargs()
elif PROVIDER == "openai":
    _provider_kwargs = _openai_kwargs()
else:
    raise ValueError(
        f"Unknown RULESBANK_PROVIDER={PROVIDER!r}. Use 'openai' or 'openrouter'."
    )

# LLM instance (global, reusable)
llm = ChatOpenAI(
    model=MODEL,
    temperature=TEMPERATURE,
    timeout=LLM_REQUEST_TIMEOUT,
    max_retries=LLM_MAX_RETRIES,
    **_provider_kwargs,
)
