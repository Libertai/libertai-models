import json
import re
from typing import Any, Iterator

import httpx

from src.config import config
from src.interfaces.usage import AudioUsageFullData, ImageUsageFullData, TextUsageFullData, Usage, UserContext


def _extract_cached_tokens(usage_json: dict) -> int:
    """Read the prefix-cache hit count from an OpenAI/vLLM usage object.

    vLLM only emits `usage.prompt_tokens_details.cached_tokens` when the server runs with
    `--enable-prompt-tokens-details`; otherwise the field is absent/null and we report 0.
    """
    details = usage_json.get("prompt_tokens_details")
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return 0


def _iter_json_at_key(text: str, key: str) -> Iterator[dict]:
    """Yield each `"key": {...}` value as a parsed dict. Handles nested braces."""
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*(?=\{{)')
    for match in pattern.finditer(text):
        start = match.end()
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break


async def report_usage_event_task(usage: TextUsageFullData | ImageUsageFullData | AudioUsageFullData):
    print(f"Collecting usage {usage}")
    try:
        timeout = httpx.Timeout(timeout=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            path = "api-keys/admin/usage"
            response = await client.post(f"{config.BACKEND_URL}/{path}", json=usage.model_dump())
            if response.status_code != 200:
                print(f"Error reporting usage: {response.status_code}")

    except Exception as e:
        print(f"Exception occurred during usage report {str(e)}")


def extract_usage_info_from_raw(raw_data: bytes, context: UserContext) -> Usage:
    """
    Extract token usage from raw response content that contains embedded JSON.

    Args:
        raw_data: Raw response bytes
        context: UserContext with the endpoint path

    Returns:
        Usage object with extracted token data
    """
    try:
        text = raw_data.decode("utf-8", errors="ignore")
    except Exception:
        raise ValueError("Unable to decode raw response content")

    if context.endpoint == "v1/messages":
        # Claude/Anthropic streaming: `message_start` carries initial usage (output_tokens=0
        # or 1) and `message_delta` carries the final cumulative usage. Both report
        # input_tokens, so SUMMING would double-count. Take the max across all events.
        input_tokens = 0
        output_tokens = 0
        seen = False
        for usage_json in _iter_json_at_key(text, "usage"):
            seen = True
            input_tokens = max(input_tokens, int(usage_json.get("input_tokens", 0)))
            output_tokens = max(output_tokens, int(usage_json.get("output_tokens", 0)))

        if seen:
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0)

        raise ValueError("No usage data found in Claude API streaming response")

    elif context.endpoint == "v1/responses":
        # Responses API streaming: final `response.completed` event carries usage. Object may
        # be nested (e.g. input_tokens_details), so we need a brace-aware extractor.
        found = None
        for usage_json in _iter_json_at_key(text, "usage"):
            if "input_tokens" in usage_json or "output_tokens" in usage_json:
                found = usage_json
        if found is not None:
            return Usage(
                input_tokens=int(found.get("input_tokens", 0)),
                output_tokens=int(found.get("output_tokens", 0)),
                cached_tokens=0,
            )
        raise ValueError("No usage data found in Responses streaming response")

    elif context.endpoint in ["v1/chat/completions", "v1/completions"]:
        # vLLM / OpenAI standard: final SSE chunk has {"usage": {"prompt_tokens": .., "completion_tokens": .., "total_tokens": ..}}
        # when stream_options.include_usage=true (we force-inject this in proxy).
        # May include nested prompt_tokens_details on newer vLLM, so brace-aware extract.
        for usage_json in _iter_json_at_key(text, "usage"):
            if "prompt_tokens" in usage_json or "completion_tokens" in usage_json:
                return Usage(
                    input_tokens=int(usage_json.get("prompt_tokens", 0)),
                    output_tokens=int(usage_json.get("completion_tokens", 0)),
                    cached_tokens=_extract_cached_tokens(usage_json),
                )

        # llama.cpp: usage is embedded as a "timings" object in the final chunk
        for timings_json in _iter_json_at_key(text, "timings"):
            return Usage(
                input_tokens=int(timings_json.get("cache_n", 0)) + int(timings_json.get("prompt_n", 0)),
                output_tokens=int(timings_json.get("predicted_n", 0)),
                cached_tokens=0,
            )

    elif context.endpoint == "completions":
        # Look for raw keys like: tokens_evaluated: 123, tokens_predicted: 456
        evaluated_match = re.search(r"tokens_evaluated\s*[:=]\s*(\d+)", text)
        predicted_match = re.search(r"tokens_predicted\s*[:=]\s*(\d+)", text)

        return Usage(
            input_tokens=int(evaluated_match.group(1)) if evaluated_match else 0,
            output_tokens=int(predicted_match.group(1)) if predicted_match else 0,
            cached_tokens=0,
        )

    raise ValueError("Can't extract usage metrics for this endpoint")


def extract_usage_info(data: dict[str, Any], context: UserContext) -> Usage:
    """
    Extract token cached and predicted counts from JSON response.

    Args:
        data: The JSON response from the server
        context: The user context
    """

    if context.endpoint == "v1/messages":
        # Claude API format: usage.input_tokens and usage.output_tokens
        usage_data: dict = data.get("usage", {})
        return Usage(
            input_tokens=int(usage_data.get("input_tokens", 0)),
            output_tokens=int(usage_data.get("output_tokens", 0)),
            cached_tokens=0,
        )
    elif context.endpoint == "v1/responses":
        # Responses API format: usage at top level with input_tokens, output_tokens
        usage_data = data.get("usage", {})
        return Usage(
            input_tokens=int(usage_data.get("input_tokens", 0)),
            output_tokens=int(usage_data.get("output_tokens", 0)),
            cached_tokens=0,
        )
    elif context.endpoint in ["v1/chat/completions", "v1/completions"]:
        usage_data = data.get("usage", {})
        return Usage(
            input_tokens=int(usage_data.get("prompt_tokens", 0)),
            output_tokens=int(usage_data.get("completion_tokens", 0)),
            cached_tokens=_extract_cached_tokens(usage_data),
        )
    elif context.endpoint == "v1/embeddings":
        # Embeddings are input-only: usage carries prompt_tokens (== total_tokens), no completion.
        usage_data = data.get("usage", {})
        return Usage(
            input_tokens=int(usage_data.get("prompt_tokens", 0)),
            output_tokens=0,
            cached_tokens=0,
        )
    elif context.endpoint == "completions":
        return Usage(
            input_tokens=int(data.get("tokens_evaluated", 0)),
            output_tokens=int(data.get("tokens_predicted", 0)),
            cached_tokens=0,
        )
    else:
        raise ValueError("Can't extract usage metrics")
