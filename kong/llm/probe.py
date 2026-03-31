"""Endpoint probing for LLM providers.

Validates connectivity and authentication before committing
to expensive operations like Ghidra startup.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import openai

from kong.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

_PROBE_DUMMY_KEY = "not-needed"


def probe_endpoint(config: LLMConfig) -> bool:
    if config.provider is LLMProvider.CUSTOM:
        return _probe_custom(config)
    if config.provider is LLMProvider.OPENAI:
        return _probe_openai(config)
    return _probe_anthropic(config)


def _probe_custom(config: LLMConfig) -> bool:
    api_key = config.api_key if config.api_key else _PROBE_DUMMY_KEY
    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.base_url)
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("Custom endpoint rejected API key")
        return False
    except openai.APIConnectionError:
        logger.warning("Could not connect to %s", config.base_url)
        return False
    except openai.APIError as e:
        logger.warning("Custom endpoint error: %s", e.message)
        return False
    except Exception as e:
        logger.warning("Could not validate %s: %s", config.base_url, e)
        return False


def _probe_openai(config: LLMConfig) -> bool:
    try:
        client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("OpenAI API key is invalid")
        return False
    except openai.APIError as e:
        logger.warning("OpenAI API error: %s", e.message)
        return False


def _probe_anthropic(config: LLMConfig) -> bool:
    try:
        from kong.llm.client import DEFAULT_BASE_URL, DEFAULT_MODEL
        base_url = config.base_url or DEFAULT_BASE_URL
        client = anthropic.Anthropic(api_key=config.api_key, base_url=base_url)
        client.messages.create(
            model=config.model or DEFAULT_MODEL,
            max_tokens=32,
            temperature=1.0,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except anthropic.AuthenticationError:
        logger.warning("Anthropic API key is invalid")
        return False
    except anthropic.APIError as e:
        logger.warning("Anthropic API error: %s", e.message)
        return False
