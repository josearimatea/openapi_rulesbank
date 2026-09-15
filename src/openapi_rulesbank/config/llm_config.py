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

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
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


class UsageTracker(BaseCallbackHandler):
    """Totals the tokens and money every LLM call in this process spends.

    The pipeline invokes the model from twelve places across five nodes, so a
    per-call tally in each of them would be repetitive and easy to forget in the
    next node added. A callback on the shared client sees them all instead.

    OpenRouter returns `cost` (US dollars, already summed over prompt and
    completion) inside token_usage; OpenAI does not, and the total then stays 0.0
    while the token counts remain accurate. Cached prompt tokens are counted in
    `prompt` as the provider reports them — they are billed differently, and are
    reported separately so a run's real cost is traceable.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cost = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage:
            # Structured-output calls report usage on the generation instead.
            for gen in (response.generations or []):
                for g in gen:
                    meta = getattr(getattr(g, "message", None), "response_metadata", {}) or {}
                    usage = meta.get("token_usage") or {}
                    if usage:
                        break
                if usage:
                    break
        if not usage:
            return
        self.calls += 1
        self.prompt_tokens     += usage.get("prompt_tokens", 0) or 0
        self.completion_tokens += usage.get("completion_tokens", 0) or 0
        self.cost              += usage.get("cost", 0.0) or 0.0
        self.reasoning_tokens  += (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
        self.cached_tokens += (
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls":         self.calls,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.prompt_tokens + self.completion_tokens,
            "reasoning_tokens":  self.reasoning_tokens,
            "cached_prompt_tokens": self.cached_tokens,
            "cost_usd":          round(self.cost, 6),
        }


# One tracker per process — the run is the process, and the Builder reads it on
# every section to report the totals accumulated so far.
usage_tracker = UsageTracker()


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
    # top-level `reasoning_effort`. "none" means OMIT the parameter here, exactly
    # as it does for OpenAI below — not {"enabled": False}.
    #
    # Sending the disable flag was measurably worse. On the phase 2 anchor prompt,
    # same prompt and schema, temperature 0: with reasoning disabled the anchor came
    # out wrong in 8 of 8 runs (both "openai/gpt-5.6-luna" and its -pro variant);
    # with the flag omitted it came out right in 8 of 8. OpenAI direct also answers
    # correctly, and for the same reason — there, "none" was already omitting it.
    # The asymmetry made one route look better than the other when the only
    # difference was whether the model was allowed to reason at all.
    if REASONING_EFFORT and REASONING_EFFORT != "none":
        kwargs["extra_body"] = {"reasoning": {"effort": REASONING_EFFORT}}
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
    # Attached here rather than at each call site: every node shares this client,
    # so the tally covers the whole run without a node having to remember it.
    callbacks=[usage_tracker],
    **_provider_kwargs,
)
