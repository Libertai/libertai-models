from src.interfaces.usage import AudioUsage, AudioUsageFullData, InferenceCallType


def test_audio_usage_type_is_audio():
    u = AudioUsage(input_tokens=42)
    assert u.type == InferenceCallType.audio
    assert u.input_tokens == 42


def test_audio_usage_full_data_carries_context():
    full = AudioUsageFullData(
        key="k", model_name="kokoro", endpoint="v1/audio/speech", input_tokens=42,
    )
    payload = full.model_dump()
    assert payload["input_tokens"] == 42
    assert payload["model_name"] == "kokoro"
    assert payload["type"] == "audio"
