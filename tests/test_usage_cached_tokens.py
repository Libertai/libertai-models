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


def test_stream_usage_on_every_chunk_uses_final():
    # vLLM --enable-force-include-usage: cumulative usage on every chunk, cached details only on the last
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 0}},
        {"choices": [{"delta": {"content": "hi"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
        {
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 64}},
        },
    ]
    raw = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"
    u = extract_usage_info_from_raw(raw, CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 10, 64)


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


MSG_CTX = UserContext(key="k", model_name="glm-5.3-flash", endpoint="v1/messages")


def _sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_messages_stream_cached_prompt_not_double_counted():
    # vLLM: message_start has the whole prompt as input_tokens, message_delta splits off the cache hit
    raw = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 32428, "output_tokens": 0}}},
            {
                "type": "message_delta",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 285,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 32428,
                },
            },
        ]
    )
    u = extract_usage_info_from_raw(raw, MSG_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (32428, 285, 32428)


def test_messages_stream_partial_cache_hit():
    raw = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 58163, "output_tokens": 0}}},
            {
                "type": "message_delta",
                "usage": {"input_tokens": 9011, "output_tokens": 622, "cache_read_input_tokens": 49152},
            },
        ]
    )
    u = extract_usage_info_from_raw(raw, MSG_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (58163, 622, 49152)


def test_messages_stream_delta_without_input_keeps_start():
    # Older servers: message_delta carries only output_tokens
    raw = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 500, "output_tokens": 1}}},
            {"type": "message_delta", "usage": {"output_tokens": 42}},
        ]
    )
    u = extract_usage_info_from_raw(raw, MSG_CTX)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (500, 42, 0)
