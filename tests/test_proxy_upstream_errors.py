import json

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src import proxy
from src.api_keys import KeysManager
from src.config import TextModelConfig

VLLM_400 = {
    "error": {
        "message": "At most 4 image(s) may be provided in one prompt. (parameter=image)",
        "type": "BadRequestError",
        "param": "image",
        "code": 400,
    }
}


def make_client(monkeypatch, upstream: httpx.Response) -> TestClient:
    model = TextModelConfig(id="textmodel", url="http://upstream.local", allowed_paths=["v1/chat/completions"])
    monkeypatch.setitem(proxy.config.MODEL_CONFIGS, "textmodel", model)
    KeysManager().reset_keys({"k"})
    monkeypatch.setattr(proxy, "report_usage_event_task", lambda *a, **k: None)
    monkeypatch.setattr(proxy, "client", httpx.AsyncClient(transport=httpx.MockTransport(lambda request: upstream)))

    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app)


def post(tc: TestClient) -> httpx.Response:
    return tc.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "textmodel", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_openai_error_envelope_is_forwarded_verbatim(monkeypatch):
    tc = make_client(monkeypatch, httpx.Response(400, json=VLLM_400, headers={"content-type": "application/json"}))
    resp = post(tc)

    assert resp.status_code == 400
    # OpenAI SDKs read error.message/type/param; unwrapping to {"detail": ...} loses them.
    assert resp.json() == VLLM_400


@pytest.mark.parametrize("status", [400, 422, 500, 503])
def test_upstream_status_is_preserved(monkeypatch, status):
    tc = make_client(monkeypatch, httpx.Response(status, json={"error": {"message": "nope"}}))
    assert post(tc).status_code == status


def test_non_json_error_body_is_forwarded_as_is(monkeypatch):
    tc = make_client(
        monkeypatch, httpx.Response(502, text="upstream exploded", headers={"content-type": "text/plain"})
    )
    resp = post(tc)

    assert resp.status_code == 502
    assert resp.text == "upstream exploded"
    assert resp.headers["content-type"].startswith("text/plain")


def test_empty_error_body_gets_openai_shaped_placeholder(monkeypatch):
    tc = make_client(monkeypatch, httpx.Response(503, content=b""))
    resp = post(tc)

    assert resp.status_code == 503
    assert json.loads(resp.content)["error"]["code"] == 503
