"""
Async retry utilities for handling LLM rate limits with exponential backoff.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")

# Attempt to import OpenAI error types, but keep optional to avoid hard deps here
try:  # pragma: no cover - best-effort import
    from openai import OpenAIError  # type: ignore
    try:
        from openai import RateLimitError  # type: ignore
    except Exception:  # pragma: no cover
        RateLimitError = None  # type: ignore
    try:
        from openai import APIStatusError  # type: ignore
    except Exception:  # pragma: no cover
        APIStatusError = None  # type: ignore
except Exception:  # pragma: no cover
    OpenAIError = None  # type: ignore
    RateLimitError = None  # type: ignore
    APIStatusError = None  # type: ignore


def is_rate_limit_error(err: Exception) -> bool:
    """Best-effort detection of OpenAI rate limit errors.

    Works with both legacy and newer OpenAI SDKs and falls back to string matching.
    """
    # Explicit OpenAI types
    if RateLimitError is not None and isinstance(err, RateLimitError):
        return True
    if APIStatusError is not None and isinstance(err, APIStatusError):
        status = getattr(err, "status_code", None) or getattr(err, "status", None)
        if status == 429:
            return True
    # Generic OpenAIError that includes response metadata
    if OpenAIError is not None and isinstance(err, OpenAIError):
        status = getattr(err, "status_code", None) or getattr(err, "status", None)
        if status == 429:
            return True

    # Fallback heuristics
    msg = str(err).lower()
    if "rate limit" in msg or "too many requests" in msg or "status code 429" in msg:
        return True

    # Also check common HTTP-like attributes
    status_attr = getattr(err, "status_code", None) or getattr(err, "status", None)
    if status_attr == 429:
        return True

    return False


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 6,
    base: float = 1.0,
    max_delay: float = 30.0,
    multiplier: float = 2.0,
    jitter: float = 0.5,
    logger: Optional[logging.Logger] = None,
    classify: Callable[[Exception], bool] = is_rate_limit_error,
    purpose: Optional[str] = None,
) -> T:
    """Retry an async operation on rate limit with exponential backoff + jitter.

    Args:
        fn: A zero-arg coroutine factory to invoke each attempt.
        attempts: Max attempts (including the first).
        base: Initial delay (seconds).
        max_delay: Max delay cap (seconds).
        multiplier: Exponential multiplier per attempt.
        jitter: Uniform random jitter (0..jitter) added to delay.
        logger: Optional logger for detailed observability.
        classify: Function deciding whether an exception is retriable (rate limit).
        purpose: Short description of the operation for logs.
    """
    _logger = logger or logging.getLogger(__name__)

    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: PERF203 - explicit retries
            retriable = classify(e)
            is_last = i >= attempts - 1
            if not retriable or is_last:
                # Final failure or non-retriable
                _logger.error(
                    "LLM call failed",
                    extra={
                        "purpose": purpose or "unknown",
                        "attempt": i + 1,
                        "attempts": attempts,
                        "retriable": retriable,
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:800],
                    },
                    exc_info=True,
                )
                raise

            # Compute backoff delay
            delay = min(max_delay, base * (multiplier ** i))
            delay += random.uniform(0, max(0.0, jitter))

            _logger.warning(
                "Rate limited; backing off",
                extra={
                    "purpose": purpose or "unknown",
                    "attempt": i + 1,
                    "attempts": attempts,
                    "next_delay_seconds": round(delay, 3),
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:300],
                },
            )
            await asyncio.sleep(delay)

    # Defensive; loop should return or raise
    raise RuntimeError("retry_async exhausted attempts without returning or raising")
