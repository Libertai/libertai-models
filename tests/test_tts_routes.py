import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api_keys import KeysManager
from src.config import AudioModelConfig
import src.tts_routes as tts_routes


@pytest.fixture
def client(monkeypatch):
    cfg = AudioModelConfig(
        id="kokoro",
        local_path="hexgrad/Kokoro-82M",
        allowed_paths=["v1/audio/speech"],
        default_voice="af_heart",
    )
    monkeypatch.setattr(tts_routes.config, "MODEL_CONFIGS", {"kokoro": cfg})
    KeysManager().reset_keys({"good"})
    app = FastAPI()
    app.include_router(tts_routes.router)
    return TestClient(app)


def _post(client, body, token="good"):
    return client.post("/v1/audio/speech", json=body, headers={"Authorization": f"Bearer {token}"})


def test_rejects_bad_api_key(client):
    r = _post(client, {"model": "kokoro", "input": "hi"}, token="bad")
    assert r.status_code == 401


def test_rejects_unknown_model(client):
    r = _post(client, {"model": "nope", "input": "hi"})
    assert r.status_code == 404


def test_rejects_empty_input(client):
    r = _post(client, {"model": "kokoro", "input": "   "})
    assert r.status_code == 400


def test_rejects_non_wav_format(client):
    r = _post(client, {"model": "kokoro", "input": "hi", "response_format": "mp3"})
    assert r.status_code == 400
    assert "wav" in r.json()["detail"].lower()


def test_rejects_out_of_range_speed(client):
    r = _post(client, {"model": "kokoro", "input": "hi", "speed": 9.0})
    assert r.status_code == 400
