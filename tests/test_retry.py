import unittest
from types import SimpleNamespace

from research_agent.retry import RetryPolicy, get_retry_after_seconds, is_retryable_error, retry_call


class StatusError(Exception):
    def __init__(self, status_code, message="provider failure"):
        super().__init__(message)
        self.status_code = status_code


class CustomTransientError(Exception):
    pass


class RetryTests(unittest.TestCase):
    def test_retries_transient_failure_with_backoff(self):
        attempts = []
        sleeps = []

        def operation():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise TimeoutError("temporary")
            return "ok"

        result = retry_call(
            operation,
            policy=RetryPolicy(max_attempts=3, initial_delay=0.5, max_delay=2, jitter=0),
            sleep=sleeps.append,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, [1, 2, 3])
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_does_not_retry_bad_input(self):
        attempts = []

        def operation():
            attempts.append(1)
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            retry_call(
                operation,
                policy=RetryPolicy(max_attempts=4, initial_delay=0, jitter=0),
                sleep=lambda _: None,
            )
        self.assertEqual(len(attempts), 1)

    def test_max_delay_caps_exponential_backoff(self):
        sleeps = []
        attempts = []

        def operation():
            attempts.append(1)
            raise TimeoutError("temporary")

        with self.assertRaises(TimeoutError):
            retry_call(
                operation,
                max_attempts=4,
                initial_delay=0.5,
                backoff_factor=3,
                max_delay=1.0,
                sleep_fn=sleeps.append,
            )

        self.assertEqual(len(attempts), 4)
        self.assertEqual(sleeps, [0.5, 1.0, 1.0])

    def test_retry_after_exception_attribute_overrides_backoff(self):
        sleeps = []
        attempts = []

        class RetryAfterError(Exception):
            retry_after = 7

        def operation():
            attempts.append(1)
            if len(attempts) == 1:
                raise RetryAfterError("rate limited")
            return "ok"

        self.assertEqual(
            retry_call(operation, initial_delay=0.25, max_delay=1, sleep=sleeps.append),
            "ok",
        )
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(get_retry_after_seconds(RetryAfterError()), 7.0)

    def test_retry_after_response_header_is_honored(self):
        sleeps = []
        attempts = []

        def operation():
            attempts.append(1)
            if len(attempts) == 1:
                error = StatusError(429, "too many requests")
                error.response = SimpleNamespace(headers={"retry-after": "3.5"})
                raise error
            return "ok"

        self.assertEqual(retry_call(operation, sleep=sleeps.append), "ok")
        self.assertEqual(sleeps, [3.5])

    def test_classifies_retryable_and_non_retryable_failures(self):
        retryable = [
            TimeoutError("timed out"),
            ConnectionError("connection reset"),
            StatusError(429),
            StatusError(503),
            RuntimeError("service temporarily unavailable"),
        ]
        non_retryable = [
            ValueError("invalid input"),
            StatusError(400, "bad request"),
            RuntimeError("unsupported parameter"),
        ]

        for error in retryable:
            with self.subTest(error=type(error).__name__):
                self.assertTrue(is_retryable_error(error))
        for error in non_retryable:
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_retryable_error(error))

    def test_custom_retry_exception_allowlist_remains_supported(self):
        attempts = []

        def operation():
            attempts.append(1)
            if len(attempts) < 2:
                raise CustomTransientError("custom failure")
            return "ok"

        self.assertEqual(
            retry_call(
                operation,
                max_attempts=2,
                retry_exceptions=(CustomTransientError,),
                sleep_fn=lambda _: None,
            ),
            "ok",
        )

    def test_max_retries_and_sleep_fn_aliases_still_work(self):
        attempts = []
        sleeps = []

        def operation():
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("temporary")
            return "ok"

        self.assertEqual(
            retry_call(
                operation,
                max_retries=1,
                initial_delay=0.25,
                sleep_fn=sleeps.append,
            ),
            "ok",
        )
        self.assertEqual(attempts, [1, 1])
        self.assertEqual(sleeps, [0.25])
