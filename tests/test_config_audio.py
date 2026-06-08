from src.config import AudioModelConfig


def test_audio_model_config_parses():
    raw = {
        "type": "audio",
        "id": "kokoro",
        "local_path": "hexgrad/Kokoro-82M",
        "allowed_paths": ["v1/audio/speech"],
        "default_voice": "af_heart",
        "lang_code": "a",
    }
    cfg = AudioModelConfig(**raw)
    assert cfg.type == "audio"
    assert cfg.id == "kokoro"
    assert cfg.default_voice == "af_heart"
    assert cfg.lang_code == "a"
    assert "v1/audio/speech" in cfg.allowed_paths


def test_audio_model_config_lang_code_defaults_to_american_english():
    cfg = AudioModelConfig(
        id="kokoro",
        local_path="hexgrad/Kokoro-82M",
        allowed_paths=["v1/audio/speech"],
        default_voice="af_heart",
    )
    assert cfg.lang_code == "a"
