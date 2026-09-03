import json
import threading
import unittest

from research_agent.telemetry import RunTelemetry


class FakeClock:
    def __init__(self, *values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class TelemetryTests(unittest.TestCase):
    def test_provider_metrics_roll_up_to_run_snapshot(self):
        telemetry = RunTelemetry("run-123")
        provider = telemetry.provider("openalex")

        provider.record_retry(2)
        provider.record_failure()
        provider.record_results(7)
        provider.record_cache_hit(3)
        provider.record_tokens(input_tokens=120, output_tokens=45)
        provider.record_cost(0.0123456789)
        provider.set_stop_reason("provider_complete")
        telemetry.set_stop_reason("evidence_sufficient")

        snapshot = telemetry.snapshot()
        provider_snapshot = snapshot["providers"]["openalex"]

        self.assertEqual(snapshot["run_id"], "run-123")
        self.assertEqual(snapshot["retries"], 2)
        self.assertEqual(snapshot["failures"], 1)
        self.assertEqual(snapshot["result_count"], 7)
        self.assertEqual(snapshot["cache_hits"], 3)
        self.assertEqual(
            snapshot["token_usage"],
            {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
        )
        self.assertEqual(snapshot["estimated_cost_usd"], 0.01234568)
        self.assertEqual(snapshot["stop_reason"], "evidence_sufficient")
        self.assertEqual(provider_snapshot["stop_reason"], "provider_complete")

    def test_run_and_provider_timers_track_separate_latency(self):
        telemetry = RunTelemetry("timed", clock=FakeClock(10.0, 10.8, 20.0, 20.25))

        with telemetry.timer() as run_timer:
            pass
        with telemetry.provider("wikipedia").timer() as provider_timer:
            pass

        snapshot = telemetry.snapshot()
        self.assertAlmostEqual(run_timer.elapsed_seconds, 0.8)
        self.assertAlmostEqual(provider_timer.elapsed_seconds, 0.25)
        self.assertEqual(snapshot["latency_ms"], 800.0)
        self.assertEqual(snapshot["providers"]["wikipedia"]["latency_ms"], 250.0)

    def test_timer_records_failure_without_suppressing_exception(self):
        telemetry = RunTelemetry("failed", clock=FakeClock(4.0, 4.1))

        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            with telemetry.provider("crossref").timer():
                raise RuntimeError("provider unavailable")

        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["failures"], 1)
        self.assertEqual(snapshot["providers"]["crossref"]["failures"], 1)
        self.assertEqual(snapshot["providers"]["crossref"]["latency_ms"], 100.0)

    def test_snapshot_is_detached_and_json_serializable(self):
        telemetry = RunTelemetry("json")
        telemetry.provider("arxiv").record_results(2)

        first = telemetry.snapshot()
        encoded = json.dumps(first, allow_nan=False, sort_keys=True)
        first["providers"]["arxiv"]["result_count"] = 999

        self.assertIn('"run_id": "json"', encoded)
        self.assertEqual(
            telemetry.snapshot()["providers"]["arxiv"]["result_count"],
            2,
        )

    def test_updates_are_thread_safe(self):
        telemetry = RunTelemetry("concurrent")
        worker_count = 8
        iterations = 500
        barrier = threading.Barrier(worker_count)

        def update_provider(worker_id):
            provider = telemetry.provider(f"provider-{worker_id % 2}")
            barrier.wait()
            for _ in range(iterations):
                provider.record_retry()
                provider.record_results(2)
                provider.record_cache_hit()
                provider.record_tokens(input_tokens=3, output_tokens=1)
                provider.record_cost(0.0001)

        threads = [
            threading.Thread(target=update_provider, args=(worker_id,))
            for worker_id in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = telemetry.snapshot()
        events = worker_count * iterations
        self.assertEqual(snapshot["retries"], events)
        self.assertEqual(snapshot["result_count"], events * 2)
        self.assertEqual(snapshot["cache_hits"], events)
        self.assertEqual(snapshot["token_usage"]["total_tokens"], events * 4)
        self.assertAlmostEqual(snapshot["estimated_cost_usd"], events * 0.0001)
        self.assertEqual(
            sum(item["retries"] for item in snapshot["providers"].values()),
            events,
        )

    def test_invalid_values_are_rejected_without_mutating_metrics(self):
        telemetry = RunTelemetry("validation")

        invalid_calls = (
            lambda: telemetry.record_results(-1),
            lambda: telemetry.record_retry(count=True),
            lambda: telemetry.record_tokens(input_tokens=1.5),
            lambda: telemetry.record_cost(float("nan")),
            lambda: telemetry.record_latency(float("inf")),
            lambda: telemetry.provider("  "),
            lambda: telemetry.set_stop_reason(""),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["result_count"], 0)
        self.assertEqual(snapshot["retries"], 0)
        self.assertEqual(snapshot["token_usage"]["total_tokens"], 0)
        self.assertEqual(snapshot["estimated_cost_usd"], 0.0)

    def test_manual_timer_is_one_shot(self):
        telemetry = RunTelemetry("manual", clock=FakeClock(2.0, 2.5))
        timer = telemetry.provider("semantic-scholar").timer(record_failure=False)

        timer.start()
        self.assertEqual(timer.stop(), 0.5)
        with self.assertRaisesRegex(RuntimeError, "one-shot"):
            timer.start()
        with self.assertRaisesRegex(RuntimeError, "not been started"):
            timer.stop()


if __name__ == "__main__":
    unittest.main()
