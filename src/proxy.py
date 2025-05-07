from http import HTTPStatus
from typing import Annotated, Any, Union

import aiohttp
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import config
from src.interfaces.usage import UserContext, UsageFullData, Usage
from src.tasks import report_usage_event_task

router = APIRouter(tags=["Proxy service"])
keys_manager = KeysManager()
security = HTTPBearer()


class ProxyRequest(BaseModel):
    model: str
    prefer_gpu: bool = False

    class Config:
        extra = "allow"  # Allow extra fields


async def process_response(
    response: aiohttp.ClientResponse, user_context: UserContext, background_tasks: BackgroundTasks
) -> Union[Response, StreamingResponse]:
    """
    Process the response from the upstream server and extract token information.

    Args:
        response: The response from the upstream server
        user_context: Context
        background_tasks: Tasks

    Returns:
        Either a Response or StreamingResponse object with the processed data
    """
    content_type = response.headers.get("Content-Type", "")

    # Handle JSON responses - extract token information
    if "application/json" in content_type:
        try:
            # Get response JSON to extract token counts
            response_json = await response.json()

            # Extract usage information
            try:
                if background_tasks:
                    usage_data = UsageFullData(
                        **user_context.model_dump(), **extract_usage_info(response_json, user_context).model_dump()
                    )

                    background_tasks.add_task(report_usage_event_task, usage_data)
            except Exception as e:
                print(f"Exception occurred during usage report {str(e)}")

            # Return processed JSON response
            return Response(
                content=await response.read(),
                status_code=response.status,
                headers=response.headers,
                media_type=content_type,
            )
        except Exception:
            # If JSON parsing fails, fall back to streaming
            pass

    # For non-JSON responses, use streaming
    return StreamingResponse(
        content=response.content.iter_any(),
        status_code=response.status,
        headers=dict(response.headers),
    )


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


@router.post("/{full_path:path}")
async def proxy_request(
    full_path: str,
    request: Request,
    proxy_request_data: ProxyRequest,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    background_tasks: BackgroundTasks,
):
    token = credentials.credentials
    if not keys_manager.key_exists(token):
        return HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid API key")

    if full_path not in config.MODEL_CONFIG.completion_paths:
        return HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid inference path")

    # Get model from request
    model_name = proxy_request_data.model

    user_context = UserContext(key=token, model_name=model_name, endpoint=full_path)

    # Get the original request body
    body = await request.json()

    # Forward the request to the selected server
    async with aiohttp.ClientSession() as session:
        try:
            # Forward the request to the selected server
            url = f"{config.MODEL_CONFIG.url}/{full_path}"

            async with session.request(
                method=request.method,
                url=url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                # Process and return the response
                return await process_response(response, user_context, background_tasks)
        except Exception as e:
            print("error", e)
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")
