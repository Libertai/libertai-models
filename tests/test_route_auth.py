"""Auth-gate coverage: every route must reject unknown/blocked keys before touching
model config or backend (image/tts pipelines, proxy upstream)."""
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

import src.image_routes as image_routes
import src.proxy as proxy
import src.tts_routes as tts_routes
from src.api_keys import KeysManager

BLOCKED_KEY = "blocked"
BLOCKED_MAP = {BLOCKED_KEY: {"reason": "no_credits", "message": "No credits."}}
FORBIDDEN_BODY = {"error": {"message": "No credits.", "type": "invalid_request_error", "code": "no_credits"}}
UNAUTHORIZED_BODY = {"detail": "Invalid API key"}


@pytest.fixture(autouse=True)
def _restore_keys_manager():
    km = KeysManager()
    saved_keys, saved_invalid = km.keys, km.invalid_keys
    yield
    km.reset_keys(saved_keys, saved_invalid)


def _client(router) -> TestClient:
    KeysManager().reset_keys(set(), dict(BLOCKED_MAP))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def image_client():
    return _client(image_routes.router)


@pytest.fixture
def tts_client():
    return _client(tts_routes.router)


@pytest.fixture
def proxy_client():
    return _client(proxy.router)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# /v1/images/generations


def test_images_generations_unknown_key(image_client):
    r = image_client.post(
        "/v1/images/generations", json={"model": "nope", "prompt": "hi"}, headers=_auth("unknown")
    )
    assert r.status_code == 401
    assert r.json() == UNAUTHORIZED_BODY


def test_images_generations_blocked_key(image_client):
    r = image_client.post(
        "/v1/images/generations", json={"model": "nope", "prompt": "hi"}, headers=_auth(BLOCKED_KEY)
    )
    assert r.status_code == 403
    assert r.json() == FORBIDDEN_BODY


# /v1/images/edits (multipart/form-data)


def test_images_edits_unknown_key(image_client):
    r = image_client.post(
        "/v1/images/edits",
        files={"prompt": (None, "hi"), "model": (None, "nope")},
        headers=_auth("unknown"),
    )
    assert r.status_code == 401
    assert r.json() == UNAUTHORIZED_BODY


def test_images_edits_blocked_key(image_client):
    r = image_client.post(
        "/v1/images/edits",
        files={"prompt": (None, "hi"), "model": (None, "nope")},
        headers=_auth(BLOCKED_KEY),
    )
    assert r.status_code == 403
    assert r.json() == FORBIDDEN_BODY


# /sdapi/v1/txt2img


def test_txt2img_unknown_key(image_client):
    r = image_client.post("/sdapi/v1/txt2img", json={"model": "nope", "prompt": "hi"}, headers=_auth("unknown"))
    assert r.status_code == 401
    assert r.json() == UNAUTHORIZED_BODY


def test_txt2img_blocked_key(image_client):
    r = image_client.post(
        "/sdapi/v1/txt2img", json={"model": "nope", "prompt": "hi"}, headers=_auth(BLOCKED_KEY)
    )
    assert r.status_code == 403
    assert r.json() == FORBIDDEN_BODY


# /v1/audio/speech (unknown-key case already covered by test_tts_routes.py)


def test_speech_blocked_key(tts_client):
    r = tts_client.post("/v1/audio/speech", json={"model": "nope", "input": "hi"}, headers=_auth(BLOCKED_KEY))
    assert r.status_code == 403
    assert r.json() == FORBIDDEN_BODY


# text proxy route (catch-all POST /{full_path:path})


def test_proxy_blocked_key(proxy_client):
    r = proxy_client.post("/v1/chat/completions", json={"model": "nope"}, headers=_auth(BLOCKED_KEY))
    assert r.status_code == 403
    assert r.json() == FORBIDDEN_BODY
