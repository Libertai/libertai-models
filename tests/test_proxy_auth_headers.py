import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src import proxy
from src.api_keys import KeysManager, extract_api_key
from src.config import TextModelConfig


def test_extract_api_key_accepts_either_header():
    assert extract_api_key({"authorization": "Bearer k"}) == "k"
    assert extract_api_key({"authorization": "bearer k"}) == "k"
    assert extract_api_key({"authorization": "k"}) == "k"
    assert extract_api_key({"x-api-key": "k"}) == "k"
    assert extract_api_key({"authorization": "Bearer from-auth", "x-api-key": "from-header"}) == "from-auth"
    assert extract_api_key({}) is None
    assert extract_api_key({"x-api-key": " "}) is None


@pytest.fixture
def client(monkeypatch):
    model = TextModelConfig(id="m", url="http://upstream.local", allowed_paths=["v1/messages"])
    monkeypatch.setitem(proxy.config.MODEL_CONFIGS, "m", model)
    KeysManager().reset_keys({"k"})
    monkeypatch.setattr(proxy, "report_usage_event_task", lambda *a, **k: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    monkeypatch.setattr(proxy, "client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app)


def test_x_api_key_authenticates(client):
    # Anthropic SDKs send the key this way and never set Authorization.
    resp = client.post("/v1/messages", headers={"x-api-key": "k"}, json={"model": "m", "messages": []})
    assert resp.status_code == 200


def test_unknown_x_api_key_is_rejected(client):
    resp = client.post("/v1/messages", headers={"x-api-key": "nope"}, json={"model": "m", "messages": []})
    assert resp.status_code == 401


def test_missing_credentials_is_rejected(client):
    resp = client.post("/v1/messages", json={"model": "m", "messages": []})
    assert resp.status_code == 401
