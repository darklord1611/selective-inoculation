from typing import TypeVar, Callable
from functools import wraps
import time
import random
import asyncio
import inspect

from loguru import logger

try:
    import openai
except ImportError:
    openai = None  # Optional dependency

S = TypeVar("S")
T = TypeVar("T")

def get_source_code(func: Callable) -> str:
    return inspect.getsource(func).strip()

def max_concurrency_async(max_size: int):
    """
    Decorator that limits the number of concurrent executions of an async function using a semaphore.

    Args:
        max_size: Maximum number of concurrent executions allowed

    Returns:
        Decorated async function with concurrency limiting
    """
    import asyncio

    def decorator(func):
        semaphore = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator

def timeout_async(timeout: float):
    """
    Decorator that times out an async function after a given timeout.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
        return wrapper
    return decorator

def auto_retry(exceptions: list[type[Exception]], max_retry_attempts: int = 3):
    """
    Decorator that retries function calls with exponential backoff on specified exceptions.

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        Decorated function that automatically retries on specified exceptions
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if attempt == max_retry_attempts:
                        raise e

                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def auto_retry_async(
    exceptions: list[type[Exception]],
    max_retry_attempts: int = 3,
    log_exceptions: bool = False,
):
    """
    Decorator that retries async function calls with exponential backoff on specified exceptions.

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        Decorated async function that automatically retries on specified exceptions
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if log_exceptions:
                        logger.exception(e)
                    if attempt == max_retry_attempts:
                        raise e
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)

            logger.warning(f"last attempt of {func.__name__}")
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def auto_retry_async_with_rate_limit(
    exceptions: list[type[Exception]],
    max_retry_attempts: int = 5,
    log_exceptions: bool = False,
    rate_limit_backoff_multiplier: float = 3.0,
):
    """Decorator that retries async function calls with enhanced 429 rate limit handling.

    This decorator extends auto_retry_async with specific handling for OpenAI rate limits:
    - Respects retry-after headers when present
    - Uses longer exponential backoff for 429 errors (default 3^n vs 2^n)
    - Falls back to standard retry logic for other exceptions

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 5)
        log_exceptions: Whether to log exceptions (default: False)
        rate_limit_backoff_multiplier: Base for exponential backoff on rate limits (default: 3.0)

    Returns:
        Decorated async function that automatically retries with rate-limit-aware backoff

    Example:
        @auto_retry_async_with_rate_limit([Exception], max_retry_attempts=5)
        async def sample_from_api():
            # Will retry on rate limits with longer backoff
            return await api.complete(prompt)
    """
    if openai is None:
        # Fallback to standard retry if openai not installed
        logger.warning(
            "OpenAI not installed, falling back to standard auto_retry_async"
        )
        return auto_retry_async(exceptions, max_retry_attempts, log_exceptions)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except openai.RateLimitError as e:
                    # Special handling for rate limit errors (429)
                    if attempt == max_retry_attempts:
                        logger.error(
                            f"Rate limit error after {max_retry_attempts} attempts"
                        )
                        raise e

                    # Check for retry-after header from API
                    retry_after = getattr(e, "retry_after", None)
                    if retry_after:
                        # Use server-specified retry time with small jitter
                        wait_time = float(retry_after) + random.uniform(0, 2)
                        logger.warning(
                            f"Rate limit (429) - server requested {retry_after}s wait. "
                            f"Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retry_attempts})"
                        )
                    else:
                        # Use longer exponential backoff for rate limits
                        wait_time = (
                            rate_limit_backoff_multiplier**attempt
                        ) + random.uniform(0, 5)
                        logger.warning(
                            f"Rate limit (429) - waiting {wait_time:.1f}s "
                            f"(attempt {attempt + 1}/{max_retry_attempts})"
                        )

                    await asyncio.sleep(wait_time)

                except tuple(exceptions) as e:
                    # Standard retry logic for other exceptions
                    if log_exceptions:
                        logger.exception(e)

                    if attempt == max_retry_attempts:
                        raise e

                    # Standard exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.debug(
                        f"Retrying after exception: {type(e).__name__} "
                        f"(waiting {wait_time:.1f}s, attempt {attempt + 1}/{max_retry_attempts})"
                    )
                    await asyncio.sleep(wait_time)

            logger.warning(f"last attempt of {func.__name__}")
            return await func(*args, **kwargs)

        return wrapper

    return decorator