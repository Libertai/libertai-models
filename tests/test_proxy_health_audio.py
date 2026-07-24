import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import proxy
from src.config import AudioModelConfig


@pytest.fixture
def client(monkeypatch):
    cfg = AudioModelConfig(
        id="kokoro",
        local_path="hexgrad/Kokoro-82M",
        allowed_paths=["v1/audio/speech"],
        default_voice="af_heart",
    )
    monkeypatch.setattr(proxy.config, "MODEL_CONFIGS", {"kokoro": cfg})
    proxy._tts_manager.register("kokoro", cfg)
    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app)


def test_health_unknown_model_404(client):
    assert client.get("/health/nope").status_code == 404


def test_health_audio_capable_returns_202_when_not_loaded(client, monkeypatch):
    # Registered but pipeline not loaded → capable
    monkeypatch.setattr(proxy._tts_manager, "is_loaded", lambda m: False)
    monkeypatch.setattr(proxy._tts_manager, "is_capable", lambda m: True)
    r = client.get("/health/kokoro")
    assert r.status_code == 202
    assert r.json()["status"] == "capable"


def test_health_audio_loaded_returns_200(client, monkeypatch):
    monkeypatch.setattr(proxy._tts_manager, "is_loaded", lambda m: True)
    r = client.get("/health/kokoro")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
