import json
from http import HTTPStatus
from typing import Annotated, Union

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import config
from src.interfaces.usage import UserContext, UsageFullData
from src.usage import report_usage_event_task, extract_usage_info_from_raw, extract_usage_info

router = APIRouter(tags=["Proxy service"])
keys_manager = KeysManager()
security = HTTPBearer()

timeout = httpx.Timeout(timeout=600.0)  # 10 minutes


class ProxyRequest(BaseModel):
    model: str
    prefer_gpu: bool = False

    class Config:
        extra = "allow"  # Allow extra fields


async def process_response(
    response: httpx.Response, user_context: UserContext, background_tasks: BackgroundTasks
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
            response_bytes = response.content
            response_json = json.loads(response_bytes)

            # Extract usage info and queue background task
            if background_tasks:
                try:
                    usage_data = UsageFullData(
                        **user_context.model_dump(),
                        **extract_usage_info(response_json, user_context).model_dump(),
                    )
                    background_tasks.add_task(report_usage_event_task, usage_data)
                except Exception as e:
                    print(f"Exception occurred during usage report {str(e)}")

            return Response(
                content=response_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type,
            )

        except Exception as e:
            print(f"Failed to parse JSON response: {e}")

    # All other (streaming or binary) responses
    try:
        # Read full content for usage extraction
        response_bytes = response.content

        if background_tasks:
            try:
                usage_data = UsageFullData(
                    **user_context.model_dump(),
                    **extract_usage_info_from_raw(response_bytes, user_context).model_dump(),
                )
                background_tasks.add_task(report_usage_event_task, usage_data)
            except Exception as e:
                print(f"Streaming usage extraction failed: {e}")

        return Response(
            content=response_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=content_type,
        )

    except Exception as e:
        print(f"Unhandled response type: {e}")
        return Response(status_code=500, content="Error handling upstream response")


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

    # Get the original request body & headers
    headers = dict(request.headers)
    body = await request.body()

    # Clean up headers
    headers.pop("host", None)

    # Forward the request
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Forward the request to the selected server
            url = f"{config.MODEL_CONFIG.url}/{full_path}"

            response: httpx.Response = await client.post(
                url, content=body, headers=headers, params=request.query_params
            )
        except Exception as e:
            print("error", e)
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")

    return await process_response(response, user_context, background_tasks)
