import json

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

import src.proxy as proxy
from src.config import TextModelConfig


@pytest.fixture
def client_and_capture(monkeypatch):
    # Register a fake vision text model whose upstream we control.
    model = TextModelConfig(
        id="visionmodel",
        url="http://upstream.local",
        allowed_paths=["v1/chat/completions", "v1/completions", "v1/messages", "v1/responses"],
    )
    monkeypatch.setitem(proxy.config.MODEL_CONFIGS, "visionmodel", model)
    # Accept any bearer token; skip real usage reporting.
    monkeypatch.setattr(proxy.keys_manager, "key_exists", lambda token: True)
    monkeypatch.setattr(proxy, "report_usage_event_task", lambda *a, **k: None)

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    monkeypatch.setattr(proxy, "client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app), captured


def test_remote_image_url_is_inlined(client_and_capture, monkeypatch):
    tc, captured = client_and_capture

    async def fake_get(url):
        return "QUJD", "image/png"

    monkeypatch.setattr("src.image_fetch.get_or_fetch", fake_get)
    resp = tc.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://ex.com/a.png"}}]}]},
    )
    assert resp.status_code == 200
    fwd = json.loads(captured["body"])
    assert fwd["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,QUJD"
    # The rewrite grows the body; the stale Content-Length must have been dropped so
    # httpx recomputed it — otherwise the upstream would read a truncated body.
    assert int(captured["headers"]["content-length"]) == len(captured["body"])


def test_anthropic_url_source_is_inlined(client_and_capture, monkeypatch):
    tc, captured = client_and_capture

    async def fake_get(url):
        return "QUJD", "image/png"

    monkeypatch.setattr("src.image_fetch.get_or_fetch", fake_get)
    resp = tc.post(
        "/v1/messages",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://ex.com/a.png"}}]}]},
    )
    assert resp.status_code == 200
    fwd = json.loads(captured["body"])
    assert fwd["messages"][0]["content"][0]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "QUJD"}
    assert int(captured["headers"]["content-length"]) == len(captured["body"])


def test_responses_input_image_is_inlined(client_and_capture, monkeypatch):
    tc, captured = client_and_capture

    async def fake_get(url):
        return "QUJD", "image/png"

    monkeypatch.setattr("src.image_fetch.get_or_fetch", fake_get)
    resp = tc.post(
        "/v1/responses",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://ex.com/a.png"}]}]},
    )
    assert resp.status_code == 200
    fwd = json.loads(captured["body"])
    assert fwd["input"][0]["content"][0]["image_url"] == "data:image/png;base64,QUJD"
    assert int(captured["headers"]["content-length"]) == len(captured["body"])


def test_failed_image_url_returns_400_and_is_not_forwarded(client_and_capture, monkeypatch):
    tc, captured = client_and_capture

    async def fake_get(url):
        raise HTTPException(status_code=400, detail=f"Failed to fetch image URL {url}: non-public")

    monkeypatch.setattr("src.image_fetch.get_or_fetch", fake_get)
    resp = tc.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://169.254.169.254/"}}]}]},
    )
    assert resp.status_code == 400
    assert "Failed to fetch image URL" in resp.json()["detail"]
    assert "url" not in captured  # request never reached the upstream


def test_unexpected_inline_error_forwards_body_unmodified(client_and_capture, monkeypatch):
    tc, captured = client_and_capture

    async def boom(body_json):
        raise RuntimeError("unexpected bug in inline")

    monkeypatch.setattr(proxy, "inline_remote_images", boom)
    resp = tc.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://ex.com/a.png"}}]}]},
    )
    # A non-HTTPException bug must NOT 500 (that would make the router fail over on
    # every replica) — the request proceeds with the body forwarded unchanged.
    assert resp.status_code == 200
    fwd = json.loads(captured["body"])
    assert fwd["messages"][0]["content"][0]["image_url"]["url"] == "https://ex.com/a.png"


def test_stream_and_inline_both_applied(client_and_capture, monkeypatch):
    tc, captured = client_and_capture

    async def fake_get(url):
        return "QUJD", "image/png"

    monkeypatch.setattr("src.image_fetch.get_or_fetch", fake_get)
    resp = tc.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "stream": True, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://ex.com/a.png"}}]}]},
    )
    assert resp.status_code == 200
    fwd = json.loads(captured["body"])
    # a single parse/serialize applied BOTH rewrites
    assert fwd["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert fwd["stream_options"]["include_usage"] is True
    assert int(captured["headers"]["content-length"]) == len(captured["body"])


def test_v1_completions_is_not_inlined(client_and_capture, monkeypatch):
    tc, captured = client_and_capture
    calls = {"n": 0}

    async def counting_inline(body_json):
        calls["n"] += 1
        return body_json, False

    monkeypatch.setattr(proxy, "inline_remote_images", counting_inline)
    resp = tc.post(
        "/v1/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "prompt": "see http://ex.com/a.png"},
    )
    assert resp.status_code == 200
    assert calls["n"] == 0  # v1/completions is not in IMAGE_INLINE_PATHS


def test_no_image_body_passes_through_unchanged(client_and_capture):
    tc, captured = client_and_capture
    resp = tc.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "visionmodel", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert json.loads(captured["body"])["messages"][0]["content"] == "hi"
