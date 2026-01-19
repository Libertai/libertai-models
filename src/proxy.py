import json
from http import HTTPStatus
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import TextModelConfig, config
from src.interfaces.usage import UserContext, UsageFullData
from src.usage import report_usage_event_task, extract_usage_info_from_raw, extract_usage_info

router = APIRouter(tags=["Proxy service"])
keys_manager = KeysManager()
security = HTTPBearer()

timeout = httpx.Timeout(timeout=600.0)  # 10 minutes


class ProxyRequest(BaseModel):
    model: str

    class Config:
        extra = "allow"  # Allow extra fields


client = httpx.AsyncClient(timeout=timeout)


@router.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


@router.get("/metrics/{model_name}")
async def proxy_metrics(request: Request, model_name: str):
    if model_name not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{model_name}' not found")

    model_config = config.MODEL_CONFIGS[model_name]
    if not isinstance(model_config, TextModelConfig):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Metrics only available for text models")

    url = f"{model_config.url}/metrics"

    # Get the original request headers
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        response: httpx.Response = await client.get(url, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("Content-Type", ""),
    )


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
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid API key")

    # Get model from request
    model_name = proxy_request_data.model

    if model_name not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{model_name}' not found")

    model_config = config.MODEL_CONFIGS[model_name]

    if not isinstance(model_config, TextModelConfig):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="This endpoint is only for text models")

    if full_path not in model_config.allowed_paths:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid inference path")

    user_context = UserContext(key=token, model_name=model_name, endpoint=full_path)

    # Get the original request body & headers
    headers = dict(request.headers)
    body = await request.body()

    # Clean up headers
    headers.pop("host", None)

    url = f"{model_config.url}/{full_path}"

    try:
        req = client.build_request("POST", url, content=body, headers=headers, params=request.query_params)
        response = await client.send(req, stream=True)
        response.raise_for_status()

        is_streaming_response = response.headers.get("content-type", "") == "text/event-stream"

        if is_streaming_response:

            async def generate_chunks():
                full_response_buffer = b""  # Buffer to collect all chunks
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        full_response_buffer += chunk
                finally:
                    await response.aclose()
                    if background_tasks:
                        try:
                            # Pass the full accumulated buffer to the extraction function
                            usage_data = UsageFullData(
                                **user_context.model_dump(),
                                **extract_usage_info_from_raw(full_response_buffer, user_context).model_dump(),
                            )
                            background_tasks.add_task(report_usage_event_task, usage_data)
                        except Exception as e:
                            print(f"Streaming usage extraction failed: {e}")

            return StreamingResponse(
                content=generate_chunks(),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("Content-Type", ""),
            )
        else:
            response_bytes = await response.aread()
            await response.aclose()

            response_json = json.loads(response_bytes)

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
                media_type=response.headers.get("Content-Type", ""),
            )

    except Exception as e:
        print(f"Error forwarding request: {e}")
        raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")
