"""A usage report that comes back blocked evicts the key locally.

Billing happens after the call, so the backend's answer to the usage report is the earliest
this server can know a key ran out; waiting for the next key distribution serves everything
in between for free.
"""

import json

import httpx

import src.usage as usage_module
from src.api_keys import KeysManager, check_api_key
from src.interfaces.usage import TextUsageFullData
from src.usage import apply_usage_report, report_usage_event_task

_REAL_ASYNC_CLIENT = httpx.AsyncClient

BLOCKED = {"reason": "no_credits", "message": "Usage window limit reached and no extra credits available."}


def _reset(keys=(), invalid=None):
    KeysManager().reset_keys(set(keys), dict(invalid or {}))


def _stub_backend(monkeypatch, handler) -> list[httpx.Request]:
    """Serve the usage report from ``handler``, and return the list of requests it saw.

    BACKEND_URL comes from the environment, so without it the post fails inside the task's
    catch-all and a test asserting "the key survived" would pass without a report ever
    happening. Assert against the returned list to keep that honest.
    """
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(usage_module.config, "BACKEND_URL", "http://backend.test")
    monkeypatch.setattr(
        usage_module.httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(_record)),
    )
    return seen


def _usage(key: str) -> TextUsageFullData:
    return TextUsageFullData(
        key=key,
        model_name="m",
        endpoint="v1/chat/completions",
        input_tokens=10,
        output_tokens=10,
        cached_tokens=0,
    )


def test_apply_report_blocks_the_key():
    _reset(keys=["k", "other"])
    apply_usage_report("k", {"invalid": BLOCKED})
    assert not KeysManager().key_exists("k")
    assert KeysManager().key_exists("other"), "only the reported key is evicted"
    resp = check_api_key("k")
    assert resp.status_code == 403
    assert json.loads(resp.body)["error"]["code"] == "no_credits"


def test_apply_report_keeps_a_usable_key():
    _reset(keys=["k"])
    apply_usage_report("k", {"invalid": None})
    assert KeysManager().key_exists("k")
    assert KeysManager().invalid_keys == {}


def test_apply_report_ignores_bodies_without_the_field():
    """Older backends answer `null`; the key must survive that."""
    _reset(keys=["k"])
    apply_usage_report("k", None)
    apply_usage_report("k", {})
    assert KeysManager().key_exists("k")


async def test_report_task_evicts_on_blocked_response(monkeypatch):
    _reset(keys=["k"])
    reports = _stub_backend(monkeypatch, lambda request: httpx.Response(200, json={"invalid": BLOCKED}))

    await report_usage_event_task(_usage("k"))

    assert len(reports) == 1
    assert not KeysManager().key_exists("k")


async def test_report_task_leaves_the_key_on_a_failed_report(monkeypatch):
    _reset(keys=["k"])
    reports = _stub_backend(monkeypatch, lambda request: httpx.Response(500, text="boom"))

    await report_usage_event_task(_usage("k"))

    assert len(reports) == 1, "the key must survive a failed report, not an unsent one"
    assert KeysManager().key_exists("k")
