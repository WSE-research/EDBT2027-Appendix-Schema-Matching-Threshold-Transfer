"""Thin OpenRouter client (OpenAI-compatible) with hardened retry + full logging.

Returns, per call, everything the experiment logs: content, finish_reason,
token usage, real cost (from OpenRouter), latency, retry count, and the full
raw response dump.
"""
from __future__ import annotations

import random
import time

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    APIStatusError,
)

from . import config

_CLIENT: OpenAI | None = None


class TruncatedResponseError(Exception):
    """finish_reason == 'length' — response cut off."""


class EmptyResponseError(Exception):
    """Model returned empty/null content."""


class InsufficientCreditsError(Exception):
    """OpenRouter is out of credits / payment required (HTTP 402) — the signal
    for a GRACEFUL STOP of the whole experiment (the run is resumable)."""


_CREDIT_HINTS = ("insufficient", "credit", "quota", "payment", "billing",
                 "exceeded your", "negative balance")


def _is_credit_error(err: "APIStatusError") -> bool:
    if getattr(err, "status_code", 0) == 402:
        return True
    msg = str(getattr(err, "message", "") or err).lower()
    return any(h in msg for h in _CREDIT_HINTS)


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=config.api_key(),
            base_url=config.OPENROUTER_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )
    return _CLIENT


def _usage_dict(usage, price_in: float = 0.0, price_out: float = 0.0) -> dict:
    if usage is None:
        return {}
    extra = getattr(usage, "model_extra", None) or {}
    compl = getattr(usage, "completion_tokens_details", None)
    prompt = getattr(usage, "prompt_tokens_details", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    # real cost from OpenRouter if present; else estimate from per-token prices
    real = extra.get("cost", 0) or 0
    if real:
        cost, cost_source = real, "real"
    elif (pt or ct) and (price_in or price_out):
        cost = pt / 1e6 * price_in + ct / 1e6 * price_out
        cost_source = "estimated"
    else:
        cost, cost_source = 0.0, "none"
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "reasoning_tokens": (getattr(compl, "reasoning_tokens", 0) or 0) if compl else 0,
        "cached_tokens": (getattr(prompt, "cached_tokens", 0) or 0) if prompt else 0,
        "cost": round(cost, 6),
        "cost_source": cost_source,
    }


def chat(messages: list[dict], model_id: str, temperature: float,
         price_in: float = 0.0, price_out: float = 0.0) -> dict:
    """One chat completion with retry. Raises only after exhausting retries.

    A per-internal-attempt `transport_trace` (error / finish_reason / usage /
    latency for each of the up-to-MAX_RETRIES tries) is returned on success and
    attached to the raised exception on failure, so billed-but-failed calls
    (truncation/empty/rate-limit retries) are never invisible to logging/cost.
    """
    client = get_client()
    max_tokens = config.MAX_TOKENS
    last_error: Exception | None = None
    trace: list[dict] = []

    for attempt in range(config.MAX_RETRIES):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"usage": {"include": True}},
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            choice = resp.choices[0]
            content = choice.message.content if choice.message else None
            finish = choice.finish_reason
            usage = _usage_dict(resp.usage, price_in, price_out)

            if finish == "length":
                old = max_tokens
                max_tokens = min(max_tokens * 2, config.MAX_TOKENS_CEILING)
                trace.append({"attempt": attempt, "error": "truncated",
                              "finish_reason": finish, "usage": usage,
                              "latency_ms": latency_ms})
                raise TruncatedResponseError(f"finish_reason=length at max_tokens={old}")
            if not content or not content.strip():
                trace.append({"attempt": attempt, "error": "empty",
                              "finish_reason": finish, "usage": usage,
                              "latency_ms": latency_ms})
                raise EmptyResponseError("empty content")

            return {
                "content": content,
                "finish_reason": finish,
                "usage": usage,
                "latency_ms": latency_ms,
                "retries": attempt,
                "model_id": model_id,
                "raw": resp.model_dump(),
                "transport_trace": trace,
                "error": None,
            }

        except (TruncatedResponseError, EmptyResponseError) as e:
            last_error = e
            _sleep(2 ** attempt)
        except RateLimitError as e:
            last_error = e
            if _is_credit_error(e):   # credit/quota exhaustion delivered as 429
                raise InsufficientCreditsError(str(e)) from e
            trace.append({"attempt": attempt, "error": repr(e)})
            _sleep(_backoff(attempt, base=4, cap=60))
        except (APITimeoutError, APIConnectionError, InternalServerError) as e:
            last_error = e
            trace.append({"attempt": attempt, "error": repr(e)})
            _sleep(2 ** attempt)
        except APIStatusError as e:
            last_error = e
            if getattr(e, "status_code", 0) and e.status_code >= 500:
                trace.append({"attempt": attempt, "error": repr(e)})
                _sleep(2 ** attempt)
            elif _is_credit_error(e):
                # out of credits / payment required: surface a dedicated signal
                # so the runner can stop the whole experiment gracefully.
                raise InsufficientCreditsError(str(e)) from e
            else:
                raise  # other 4xx (bad model id, auth) — don't retry
    if last_error is not None:
        setattr(last_error, "transport_trace", trace)
        setattr(last_error, "transport_retries", len(trace))
        raise last_error
    raise RuntimeError("chat failed without error")


def _backoff(attempt: int, base: float = 2, cap: float = 60) -> float:
    return min(base * (2 ** attempt), cap)


def _sleep(seconds: float) -> None:
    time.sleep(seconds + random.uniform(0, 0.5))  # jitter


def list_models() -> list[str]:
    """Return served model ids (for the smoke test's fallback search)."""
    client = get_client()
    return [m.id for m in client.models.list().data]
