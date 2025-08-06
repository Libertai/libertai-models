import json
import re
from typing import Any

import httpx

from src.config import config
from src.interfaces.usage import UsageFullData, UserContext, Usage


async def report_usage_event_task(usage: UsageFullData):
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

    if context.endpoint in ["v1/chat/completions", "v1/completions"]:
        # Look for the embedded usage JSON object
        usage_match = re.search(r'"usage"\s*:\s*({.*?})', text)
        if usage_match:
            try:
                usage_json = json.loads(usage_match.group(1))
                return Usage(
                    input_tokens=int(usage_json.get("prompt_tokens", 0)),
                    output_tokens=int(usage_json.get("completion_tokens", 0)),
                    cached_tokens=0,
                )
            except Exception as e:
                raise ValueError(f"Failed to parse usage JSON: {e}")

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

    if context.endpoint in ["v1/chat/completions", "v1/completions"]:
        usage: dict = data.get("usage", {})
        return Usage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
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
