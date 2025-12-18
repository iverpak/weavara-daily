"""
Gemini 3.0 Flash Preview Utilities

Shared utilities for Gemini 3.0 Flash Preview migration across all executive summary phases.

Migration Notes (Dec 2025):
- New SDK: google-genai (not google-generativeai)
- Model: gemini-3-flash-preview
- Temperature: 1.0 required for reasoning (use seed for determinism)
- Thinking Level: HIGH for complex analysis
- Implicit caching: Automatic after 2,048 token prefix match

Usage:
    from modules.gemini_3_utils import (
        create_gemini_3_client,
        call_with_retry,
        extract_usage_metadata,
        FLASH_3_PRICING
    )
"""

import time
import random
import logging
from typing import Dict, Any, Optional, List, Union

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# PRICING CONSTANTS (Dec 2025)
# ------------------------------------------------------------------------------

FLASH_3_PRICING = {
    "input_per_1m": 0.50,        # $0.50 per 1M input tokens
    "cache_read_per_1m": 0.05,   # $0.05 per 1M cached tokens (90% discount)
    "output_per_1m": 3.00,       # $3.00 per 1M output tokens (includes thinking)
}


# ------------------------------------------------------------------------------
# CLIENT FACTORY
# ------------------------------------------------------------------------------

def create_gemini_3_client(api_key: str, timeout: float = 120.0):
    """
    Create a Gemini 3.0 client with appropriate timeout for HIGH thinking.

    Args:
        api_key: Google Gemini API key
        timeout: HTTP timeout in seconds (default 120s for HIGH thinking)

    Returns:
        genai.Client instance
    """
    from google import genai

    return genai.Client(
        api_key=api_key,
        http_options={'timeout': timeout}
    )


# ------------------------------------------------------------------------------
# RETRY WRAPPER
# ------------------------------------------------------------------------------

def call_with_retry(
    client,
    model: str,
    contents: List,
    config,
    max_retries: int = 3,
    ticker: str = "UNKNOWN"
) -> Optional[Any]:
    """
    Call Gemini 3.0 with smart retry logic differentiating 429 vs 503 errors.

    Args:
        client: genai.Client instance
        model: Model ID (e.g., "gemini-3-flash-preview")
        contents: List of content parts
        config: GenerateContentConfig instance
        max_retries: Maximum retry attempts (default 3)
        ticker: Ticker symbol for logging

    Returns:
        Response object on success, None on failure

    Retry Strategy:
        - 429 (rate limit): Aggressive backoff 2^n + jitter
        - 503 (server busy): Lighter backoff 1.5^n + jitter
        - 400 (invalid argument): No retry
        - Timeout: Retry with same backoff as 503
    """
    from google.genai import errors as genai_errors

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response

        except genai_errors.ClientError as e:
            error_str = str(e).lower()

            # 429: Rate limit - aggressive backoff
            if '429' in str(e) or 'rate' in error_str or 'quota' in error_str:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    LOG.warning(f"[{ticker}] 🚦 Rate limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Rate limit exceeded after {max_retries + 1} attempts")
                    return None
            else:
                # 400/403: Invalid argument or auth error - don't retry
                LOG.error(f"[{ticker}] ❌ Client error (no retry): {e}")
                return None

        except genai_errors.ServerError as e:
            # 503/500: Server overloaded - lighter backoff
            if attempt < max_retries:
                wait_time = (1.5 ** attempt) + random.uniform(0, 1)
                LOG.warning(f"[{ticker}] 🔄 Server error (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                LOG.error(f"[{ticker}] ❌ Server error after {max_retries + 1} attempts: {e}")
                return None

        except genai_errors.APIError as e:
            # Other API errors
            LOG.error(f"[{ticker}] ❌ API error: {e}")
            return None

        except Exception as e:
            # Unexpected errors - log and don't retry
            LOG.error(f"[{ticker}] ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None

    LOG.error(f"[{ticker}] ❌ Exhausted all {max_retries + 1} retry attempts")
    return None


# ------------------------------------------------------------------------------
# USAGE METADATA EXTRACTION
# ------------------------------------------------------------------------------

def extract_usage_metadata(response) -> Dict[str, Any]:
    """
    Extract token usage metadata from Gemini 3.0 response with defensive checks.

    Args:
        response: Gemini response object

    Returns:
        Dict with:
            - prompt_tokens: Input tokens
            - cached_tokens: Tokens served from cache
            - thought_tokens: Reasoning tokens (billed as output)
            - output_tokens: Final answer tokens
            - total_tokens: Grand total

    Note: All fields default to 0 if absent (handles MINIMAL thinking, streaming, etc.)
    """
    if not response or not hasattr(response, 'usage_metadata'):
        return {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "thought_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    usage = response.usage_metadata

    return {
        "prompt_tokens": getattr(usage, 'prompt_token_count', 0) or 0,
        "cached_tokens": getattr(usage, 'cached_content_token_count', 0) or 0,
        "thought_tokens": getattr(usage, 'thoughts_token_count', 0) or 0,
        "output_tokens": getattr(usage, 'candidates_token_count', 0) or 0,
        "total_tokens": getattr(usage, 'total_token_count', 0) or 0
    }


def calculate_flash_3_cost(usage: Dict[str, Any]) -> float:
    """
    Calculate cost for Gemini 3.0 Flash request.

    Args:
        usage: Dict from extract_usage_metadata()

    Returns:
        Total cost in USD

    Cost Formula:
        - Uncached input: (prompt - cached) * input_rate
        - Cached input: cached * cache_read_rate
        - Output: (thought + output) * output_rate
    """
    prompt = usage.get("prompt_tokens", 0)
    cached = usage.get("cached_tokens", 0)
    thought = usage.get("thought_tokens", 0)
    output = usage.get("output_tokens", 0)

    uncached_input = prompt - cached

    input_cost = (uncached_input / 1_000_000) * FLASH_3_PRICING["input_per_1m"]
    cache_cost = (cached / 1_000_000) * FLASH_3_PRICING["cache_read_per_1m"]
    output_cost = ((thought + output) / 1_000_000) * FLASH_3_PRICING["output_per_1m"]

    return input_cost + cache_cost + output_cost


# ------------------------------------------------------------------------------
# RESPONSE TEXT EXTRACTION
# ------------------------------------------------------------------------------

def extract_response_text(response) -> Optional[str]:
    """
    Extract text from Gemini 3.0 response, filtering out thought parts.

    Args:
        response: Gemini response object

    Returns:
        Concatenated text from non-thought parts, or None if no content
    """
    if not response or not response.candidates:
        return None

    try:
        parts = response.candidates[0].content.parts
        # Filter out thought parts (defensive check)
        text_parts = [
            part.text for part in parts
            if hasattr(part, 'text') and not getattr(part, 'thought', False)
        ]
        return "".join(text_parts) if text_parts else None
    except (IndexError, AttributeError) as e:
        LOG.error(f"Failed to extract response text: {e}")
        return None


# ------------------------------------------------------------------------------
# CONFIG BUILDER
# ------------------------------------------------------------------------------

def build_thinking_config(
    thinking_level: str = "HIGH",
    include_thoughts: bool = False,
    temperature: float = 1.0,
    max_output_tokens: int = 20000,
    seed: int = 42,
    response_mime_type: str = "application/json"
):
    """
    Build GenerateContentConfig for Gemini 3.0 Flash with thinking.

    Args:
        thinking_level: MINIMAL, LOW, MEDIUM, or HIGH (default HIGH)
        include_thoughts: Whether to include reasoning in output (default False)
        temperature: Must be 1.0 for thinking models (default 1.0)
        max_output_tokens: Max tokens in response (default 20000)
        seed: Random seed for determinism (default 42)
        response_mime_type: Output format (default "application/json")

    Returns:
        GenerateContentConfig instance
    """
    from google.genai import types

    # Map string to enum
    level_map = {
        "MINIMAL": types.ThinkingLevel.MINIMAL,
        "LOW": types.ThinkingLevel.LOW,
        "MEDIUM": types.ThinkingLevel.MEDIUM,
        "HIGH": types.ThinkingLevel.HIGH
    }
    thinking_level_enum = level_map.get(thinking_level.upper(), types.ThinkingLevel.HIGH)

    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level_enum,
            include_thoughts=include_thoughts
        ),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
        response_mime_type=response_mime_type
    )
