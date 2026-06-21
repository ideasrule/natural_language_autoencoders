"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.

    Two-phase API for overlapping work. `submit(prompts)` starts processing and
    returns an opaque handle; `await_result(handle)` blocks for the completions.
    This lets stage 2 fire off every chunk's work up front and only then wait,
    so independent jobs overlap instead of running strictly one-after-another
    (critical for the Batches API, where each job can take up to an hour). The
    one-shot `complete()` is just `await_result(submit(...))`.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...

    def submit(self, prompts: list[str]) -> object:
        """Start processing `prompts`; return a handle to pass to `await_result`.

        Default is eager/synchronous — it runs `complete()` now and carries the
        result in the handle. That's the right behavior for live providers that
        already fan out internally, so there is nothing to overlap. Batch-style
        providers override this to return as soon as the job is *queued*, and do
        the waiting in `await_result()`.
        """
        return self.complete(prompts)

    def await_result(self, handle: object) -> list[str | None]:
        """Block until the job behind `handle` (from `submit`) finishes.

        The default pairs with the eager `submit` above: the handle already *is*
        the completed result list, so just return it.
        """
        return handle


def _extract_completion(message) -> str | None:
    """Pull the explanation text out of a finished Message, or None to drop it.

    Shared by the live and batch providers so they agree on what counts as a
    usable completion. Returns None for safety refusals (no answer is coming);
    raises AssertionError for any shape we didn't expect — that's a prompt-
    template or model-config bug, not a transient failure.
    """
    assert message.stop_reason in ("end_turn", "max_tokens", "refusal"), (
        f"unexpected stop_reason={message.stop_reason!r} (want end_turn/max_tokens/refusal)"
    )
    # refusal: source text tripped safety — no answer coming, drop this row.
    # content may be [] or the refusal message; either way, no explanation.
    if message.stop_reason == "refusal":
        return None
    assert len(message.content) == 1 and message.content[0].type == "text", (
        f"expected single text block, got {[b.type for b in message.content]}"
    )
    text = message.content[0].text.strip()
    # Empty text is only acceptable on a refusal (handled above). Reaching here
    # means stop_reason is end_turn/max_tokens, so a blank completion is a bug
    # — not a droppable row. Raise rather than silently emit/drop it.
    assert text, (
        f"empty completion with stop_reason={message.stop_reason!r} "
        "(empty text is only expected on a refusal)"
    )
    return text


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        return _extract_completion(resp)

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


@dataclass
class _BatchHandle:
    """Opaque handle returned by `AnthropicBatchProvider.submit`.

    `batch_id` is None for an empty prompt list (nothing was submitted). `n` is
    the prompt count, so `await_result` can size its output slot list without
    re-deriving it.
    """
    batch_id: str | None
    n: int


class AnthropicBatchProvider(CompletionProvider):
    """Provider backed by the Anthropic Message Batches API.

    Same `prompts[i] -> completion[i]` contract as AnthropicProvider, but
    submits the whole batch as one asynchronous job (`/v1/messages/batches`)
    and polls for completion instead of firing concurrent live requests. The
    Batches API runs at 50% of standard token price and tolerates much larger
    fan-out (up to 100k requests / 256 MB per batch), at the cost of latency —
    most batches finish within an hour, with a 24h ceiling. Use this for
    bulk stage-2 generation where throughput/cost matter more than turnaround;
    use AnthropicProvider when you want completions back in seconds.

    Failure handling mirrors AnthropicProvider: a per-request result that
    refused, errored transiently, was canceled, or expired becomes None (the
    caller drops that row). Only `invalid_request` errors — malformed requests,
    i.e. a code bug — abort the whole batch by raising. Results come back in
    arbitrary order, so we key strictly by `custom_id` (the prompt index),
    never by position.

    Two-phase: `submit()` creates the batch and returns immediately with a
    handle; `await_result()` polls that batch to completion and collects the
    results. The caller (stage 2) submits every chunk's batch first, then awaits
    each — so all the batches process concurrently server-side instead of one
    chunk's hour-long batch blocking the next chunk's submission. Each
    `submit()` is one batch; `complete()` is the one-shot submit-then-await.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        poll_interval: float = 30.0,
        concurrency: int = 1, #ignored
        max_retries: int = 10,
    ):
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.poll_interval = poll_interval

    def submit(self, prompts: list[str]) -> _BatchHandle:
        if not prompts:
            return _BatchHandle(batch_id=None, n=0)

        # custom_id is the prompt's index as a string — results arrive unordered,
        # so this is the only reliable way to map a result back to its slot.
        batch = self.client.messages.batches.create(
            requests=[
                Request(
                    custom_id=str(i),
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": p}],
                    ),
                )
                for i, p in enumerate(prompts)
            ]
        )
        print(f"  [AnthropicBatchProvider] submitted batch {batch.id} ({len(prompts)} prompts)")
        return _BatchHandle(batch_id=batch.id, n=len(prompts))

    def await_result(self, handle: _BatchHandle) -> list[str | None]:
        if handle.batch_id is None:
            return []

        while True:
            batch = self.client.messages.batches.retrieve(handle.batch_id)
            if batch.processing_status == "ended":
                break
            time.sleep(self.poll_interval)

        out: list[str | None] = [None] * handle.n
        n_failed = 0
        n_refused = 0
        n_expired = 0
        for result in self.client.messages.batches.results(handle.batch_id):
            idx = int(result.custom_id)
            r = result.result
            if r.type == "succeeded":
                text = _extract_completion(r.message)
                if text is None:
                    n_refused += 1
                out[idx] = text
            elif r.type == "errored":
                # invalid_request is a malformed request — a code bug, not transient.
                # Blow up loud rather than silently dropping the whole batch's worth.
                if r.error.type == "invalid_request":
                    raise RuntimeError(
                        f"batch request {idx} failed validation: {r.error.message}"
                    )
                n_failed += 1  # server-side error — safe to have dropped, retryable upstream
            elif r.type in ("canceled", "expired"):
                n_expired += 1
            else:
                raise AssertionError(f"unexpected batch result type {r.type!r} at [{idx}]")

        if n_failed or n_refused or n_expired:
            print(
                f"  [AnthropicBatchProvider] dropped {n_refused} refused + {n_failed} errored "
                f"+ {n_expired} canceled/expired of {handle.n}"
            )
        return out

    def complete(self, prompts: list[str]) -> list[str | None]:
        return self.await_result(self.submit(prompts))

