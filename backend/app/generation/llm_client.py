import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from openai import APIStatusError, AsyncOpenAI, RateLimitError

from app.config import settings
from app.exceptions import GenerationError

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

_RETRY_DELAYS = (1.0, 2.0, 4.0)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def generate_answer(
    messages: list[dict],
    stream: bool = False,
) -> str | AsyncGenerator[str, None]:
    if stream:
        return _stream_answer(messages)
    return await _blocking_answer(messages)


async def _blocking_answer(messages: list[dict]) -> str:
    client = _get_client()
    max_attempts = len(_RETRY_DELAYS) + 1

    for attempt in range(max_attempts):
        start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            usage = response.usage
            logger.info(
                "llm model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%d",
                settings.openai_model,
                usage.prompt_tokens if usage else -1,
                usage.completion_tokens if usage else -1,
                latency_ms,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIStatusError) as exc:
            is_last = attempt == max_attempts - 1
            if is_last:
                raise GenerationError(str(exc)) from exc
            delay = _RETRY_DELAYS[attempt]
            logger.warning(
                "llm attempt=%d/%d error=%s retrying_in=%.1fs",
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            raise GenerationError(str(exc)) from exc

    raise GenerationError("exceeded max retries")


async def _stream_answer(messages: list[dict]) -> AsyncGenerator[str, None]:
    client = _get_client()
    start = time.monotonic()
    try:
        stream = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info("llm stream model=%s latency_ms=%d", settings.openai_model, latency_ms)
    except Exception as exc:
        raise GenerationError(str(exc)) from exc
