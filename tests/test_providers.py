"""Tests for the two-phase CompletionProvider interface (submit / await_result).

The behavior under test is the submit-all-then-await split added so stage 2 can
get every chunk's batch in flight at once instead of waiting for each batch to
finish before submitting the next. These use a fake Anthropic client — no
network, no SDK calls beyond construction.

Run: python -m unittest tests.test_providers
"""
import types
import unittest
from unittest import mock

from nla.datagen import providers
from nla.datagen.providers import AnthropicBatchProvider, CompletionProvider


# --- fake Anthropic batch-result objects (just enough shape for the code) ----

def _succeeded(custom_id: int, text: str, stop_reason: str = "end_turn"):
    return types.SimpleNamespace(
        custom_id=str(custom_id),
        result=types.SimpleNamespace(
            type="succeeded",
            message=types.SimpleNamespace(
                stop_reason=stop_reason,
                content=[types.SimpleNamespace(type="text", text=text)],
            ),
        ),
    )


def _refusal(custom_id: int):
    return types.SimpleNamespace(
        custom_id=str(custom_id),
        result=types.SimpleNamespace(
            type="succeeded",
            message=types.SimpleNamespace(stop_reason="refusal", content=[]),
        ),
    )


def _errored(custom_id: int, err_type: str):
    # Mirror the real SDK shape: result.error is an ErrorResponse whose .type
    # is always "error"; the discriminating error object is nested one level
    # down at result.error.error (e.g. type="invalid_request_error").
    return types.SimpleNamespace(
        custom_id=str(custom_id),
        result=types.SimpleNamespace(
            type="errored",
            error=types.SimpleNamespace(
                type="error",
                error=types.SimpleNamespace(type=err_type, message="boom"),
            ),
        ),
    )


def _terminal(custom_id: int, kind: str):  # "canceled" or "expired"
    return types.SimpleNamespace(
        custom_id=str(custom_id),
        result=types.SimpleNamespace(type=kind),
    )


class FakeBatches:
    """Stand-in for client.messages.batches, logging the order of every call."""

    def __init__(self, results_by_id=None, statuses=None):
        self.calls: list[tuple[str, str]] = []  # ordered (op, batch_id) log
        self._results = results_by_id or {}
        # batch_id -> list of processing_status values for successive retrieves
        self._statuses = statuses or {}
        self._n = 0

    def create(self, requests):
        self._n += 1
        bid = f"batch_{self._n}"
        self.calls.append(("create", bid))
        return types.SimpleNamespace(id=bid)

    def retrieve(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        seq = self._statuses.setdefault(batch_id, ["ended"])
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        return types.SimpleNamespace(processing_status=status)

    def results(self, batch_id):
        self.calls.append(("results", batch_id))
        return iter(self._results.get(batch_id, []))


def _make_provider(fake_batches: FakeBatches) -> AnthropicBatchProvider:
    client = types.SimpleNamespace(
        messages=types.SimpleNamespace(batches=fake_batches)
    )
    with mock.patch.object(providers.anthropic, "Anthropic", return_value=client):
        return AnthropicBatchProvider(poll_interval=0)


class AnthropicBatchProviderTest(unittest.TestCase):
    def test_submits_all_chunks_before_awaiting_any(self):
        """The point of the refactor: submit() queues without polling, so many
        batches can be in flight before the first await_result()."""
        fb = FakeBatches(results_by_id={
            "batch_1": [_succeeded(0, "a")],
            "batch_2": [_succeeded(0, "b")],
            "batch_3": [_succeeded(0, "c")],
        })
        prov = _make_provider(fb)

        handles = [prov.submit([f"prompt-{i}"]) for i in range(3)]
        # Three batches created; nothing retrieved yet.
        self.assertEqual([op for op, _ in fb.calls], ["create", "create", "create"])

        results = [prov.await_result(h) for h in handles]
        self.assertEqual(results, [["a"], ["b"], ["c"]])
        # The first poll only happens after every submit — i.e. all creates
        # precede the first retrieve.
        ops = [op for op, _ in fb.calls]
        self.assertEqual(ops[:3], ["create", "create", "create"])
        self.assertNotIn("retrieve", ops[:3])
        self.assertIn("retrieve", ops[3:])

    def test_results_mapped_by_custom_id_not_position(self):
        """Batch results arrive unordered; they must be keyed by custom_id."""
        fb = FakeBatches(results_by_id={
            "batch_1": [_succeeded(2, "c"), _succeeded(0, "a"), _succeeded(1, "b")],
        })
        prov = _make_provider(fb)
        self.assertEqual(prov.complete(["x", "y", "z"]), ["a", "b", "c"])

    def test_droppable_failures_become_none(self):
        fb = FakeBatches(results_by_id={
            "batch_1": [
                _succeeded(0, "ok"),
                _refusal(1),
                _errored(2, "api_error"),  # transient server-side error
                _terminal(3, "expired"),
                _terminal(4, "canceled"),
            ],
        })
        prov = _make_provider(fb)
        self.assertEqual(
            prov.complete(["a", "b", "c", "d", "e"]),
            ["ok", None, None, None, None],
        )

    def test_invalid_request_raises(self):
        """A malformed request is a code bug — abort, don't silently drop."""
        fb = FakeBatches(results_by_id={"batch_1": [_errored(0, "invalid_request_error")]})
        prov = _make_provider(fb)
        with self.assertRaises(RuntimeError):
            prov.complete(["a"])

    def test_empty_prompts_creates_no_batch(self):
        fb = FakeBatches()
        prov = _make_provider(fb)
        handle = prov.submit([])
        self.assertIsNone(handle.batch_id)
        self.assertEqual(fb.calls, [])  # no API call at all
        self.assertEqual(prov.await_result(handle), [])

    def test_await_polls_until_ended(self):
        fb = FakeBatches(
            results_by_id={"batch_1": [_succeeded(0, "a")]},
            statuses={"batch_1": ["in_progress", "in_progress", "ended"]},
        )
        prov = _make_provider(fb)
        with mock.patch.object(providers.time, "sleep") as slept:
            self.assertEqual(prov.complete(["x"]), ["a"])
        # Slept once per not-yet-ended poll; the "ended" poll breaks without sleeping.
        self.assertEqual(slept.call_count, 2)


class DefaultSubmitTest(unittest.TestCase):
    def test_default_submit_is_eager(self):
        """Providers that don't override submit/await get the eager default:
        complete() runs at submit() time, await_result() just returns it."""
        class Upper(CompletionProvider):
            def __init__(self):
                self.complete_calls = 0

            def complete(self, prompts):
                self.complete_calls += 1
                return [p.upper() for p in prompts]

        prov = Upper()
        handle = prov.submit(["a", "b"])
        self.assertEqual(prov.complete_calls, 1)  # work done eagerly at submit
        self.assertEqual(prov.await_result(handle), ["A", "B"])
        self.assertEqual(prov.complete_calls, 1)  # await adds no extra work


if __name__ == "__main__":
    unittest.main()
