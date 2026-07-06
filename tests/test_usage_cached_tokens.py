import json

from src.interfaces.usage import UserContext
from src.usage import extract_usage_info, extract_usage_info_from_raw

CTX = UserContext(key="k", model_name="glm-5.2", endpoint="v1/chat/completions")


def test_non_stream_with_cached_tokens():
    data = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    }
    u = extract_usage_info(data, CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 64)


def test_non_stream_details_null():
    data = {"usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": None}}
    assert extract_usage_info(data, CTX).cached_tokens == 0


def test_non_stream_details_absent():
    data = {"usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    assert extract_usage_info(data, CTX).cached_tokens == 0


def test_stream_with_cached_tokens():
    chunk = {
        "choices": [{"delta": {}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 64},
        },
    }
    raw = (f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n").encode()
    u = extract_usage_info_from_raw(raw, CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 64)


def test_stream_without_cached_tokens():
    chunk = {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    raw = (f"data: {json.dumps(chunk)}\n\n").encode()
    assert extract_usage_info_from_raw(raw, CTX).cached_tokens == 0


ANTHROPIC_CTX = UserContext(key="k", model_name="glm-5.2", endpoint="v1/messages")


def test_anthropic_non_stream_with_cache():
    data = {"usage": {"input_tokens": 36, "output_tokens": 10, "cache_read_input_tokens": 64}}
    u = extract_usage_info(data, ANTHROPIC_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 64)


def test_anthropic_non_stream_legacy_no_cache_fields():
    data = {"usage": {"input_tokens": 100, "output_tokens": 10}}
    u = extract_usage_info(data, ANTHROPIC_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 0)


def test_anthropic_stream_with_cache():
    start = {"type": "message_start", "message": {"usage": {"input_tokens": 36, "output_tokens": 0, "cache_read_input_tokens": 64}}}
    delta = {"type": "message_delta", "usage": {"input_tokens": 36, "output_tokens": 10, "cache_read_input_tokens": 64}}
    raw = (f"event: message_start\ndata: {json.dumps(start)}\n\nevent: message_delta\ndata: {json.dumps(delta)}\n\n").encode()
    u = extract_usage_info_from_raw(raw, ANTHROPIC_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 64)


def test_anthropic_stream_legacy_no_cache_fields():
    start = {"type": "message_start", "message": {"usage": {"input_tokens": 100, "output_tokens": 0}}}
    delta = {"type": "message_delta", "usage": {"input_tokens": 100, "output_tokens": 10}}
    raw = (f"event: message_start\ndata: {json.dumps(start)}\n\nevent: message_delta\ndata: {json.dumps(delta)}\n\n").encode()
    u = extract_usage_info_from_raw(raw, ANTHROPIC_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 0)
