# src/app/config/llm_config.py
"""
LLM configuration and initialization.
Creates and exposes the global LLM instance.
Depends on settings (e.g., API key).
"""

from langchain_openai import ChatOpenAI

from .settings import OPENAI_API_KEY, MODEL, TEMPERATURE

# LLM instance (global, reusable)
llm = ChatOpenAI(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=OPENAI_API_KEY
)