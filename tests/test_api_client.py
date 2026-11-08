"""
Tests for api_client._get's retries. No network: requests.get and time.sleep are faked.

    python tests/test_api_client.py
"""
import os
import sys
from unittest import mock

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.modules.setdefault("streamlit", None)   # not needed here; skips its warnings

import api_client  # noqa: E402

OK = {"status": "ok", "data": {"1": {}}}


def api_error(message):
    return {"status": "error", "error": {"field": "x", "message": message, "code": 407, "value": None}}


class Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def run(script):
    """_get against a scripted list of bodies or exceptions; returns (result, calls, sleeps)."""
    script, calls, sleeps, errors = list(script), [], [], []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return Resp(item)

    with mock.patch.object(api_client, "API_KEY", "test-id"), \
            mock.patch.object(api_client.requests, "get", fake_get), \
            mock.patch.object(api_client.time, "sleep", sleeps.append), \
            mock.patch.object(api_client, "_ui_error", errors.append):
        result = api_client._get("account/info/", {"account_id": 1})
    return result, len(calls), sleeps, errors


def test_rate_limit_is_retried_then_succeeds():
    result, calls, sleeps, errors = run([api_error("REQUEST_LIMIT_EXCEEDED"),
                                         api_error("REQUEST_LIMIT_EXCEEDED"), OK])
    assert result == OK and calls == 3 and sleeps == [2, 4] and errors == []


def test_rate_limit_gives_up_after_about_a_minute():
    waits = list(api_client.API_ERROR_WAITS)
    result, calls, sleeps, errors = run([api_error("REQUEST_LIMIT_EXCEEDED")] * (len(waits) + 1))
    assert result is None and calls == len(waits) + 1 and sleeps == waits
    assert errors == ["API error: REQUEST_LIMIT_EXCEEDED"]


def test_other_api_errors_are_not_retried():
    result, calls, sleeps, errors = run([api_error("INVALID_ACCOUNT_ID")])
    assert result is None and calls == 1 and sleeps == [] and errors == ["API error: INVALID_ACCOUNT_ID"]


def test_timeouts_still_get_three_tries():
    result, calls, sleeps, errors = run([requests.Timeout("slow")] * 3)
    assert result is None and calls == 3 and sleeps == [1.5, 3.0]
    assert errors and errors[0].startswith("Request failed after 3 attempts")


def test_timeout_then_success():
    result, calls, sleeps, _ = run([requests.Timeout("slow"), OK])
    assert result == OK and calls == 2 and sleeps == [1.5]


def test_missing_application_id_raises():
    with mock.patch.object(api_client, "API_KEY", None):
        try:
            api_client._get("account/info/", {"account_id": 1})
        except RuntimeError as e:
            assert "WG_APPLICATION_ID" in str(e)
        else:
            raise AssertionError("a missing application id should raise")


# ── Standalone runner (no pytest required) ──────────────────────────────────

def _collect_tests():
    mod = sys.modules[__name__]
    return [getattr(mod, name) for name in dir(mod)
            if name.startswith("test_") and callable(getattr(mod, name))]


if __name__ == "__main__":
    tests = _collect_tests()
    failures = []
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed.")
    if failures:
        print("Failures:", ", ".join(failures))
        raise SystemExit(1)
