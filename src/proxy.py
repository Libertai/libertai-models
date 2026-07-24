import json
import logging
from http import HTTPStatus
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import check_api_key
from src.config import (
    AudioModelConfig,
    EmbeddingModelConfig,
    ImageEditModelConfig,
    ImageModelConfig,
    TextModelConfig,
    config,
)
from src.image_fetch import IMAGE_INLINE_PATHS, aclose_client, inline_remote_images
from src.image_generation import ImageModelManager
from src.interfaces.usage import TextUsageFullData, UserContext
from src.tts_generation import TTSModelManager
from src.usage import extract_usage_info, extract_usage_info_from_raw, report_usage_event_task

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Proxy service"])
security = HTTPBearer()

timeout = httpx.Timeout(timeout=600.0)  # 10 minutes
limits = httpx.Limits(
    max_connections=500,
    max_keepalive_connections=100,
    keepalive_expiry=30.0,  # Recycle idle connections after 30s
)


class ProxyRequest(BaseModel):
    model: str

    class Config:
        extra = "allow"  # Allow extra fields


client = httpx.AsyncClient(timeout=timeout, limits=limits)

# OpenAI completion paths that get stream_options.include_usage injected (llama.cpp
# ignores the field). Kept separate from IMAGE_INLINE_PATHS on purpose.
STREAM_USAGE_PATHS = ("v1/chat/completions", "v1/completions")

# Register image model configs with the manager
_image_manager = ImageModelManager()
for _model_id, _model_config in config.MODEL_CONFIGS.items():
    if isinstance(_model_config, (ImageModelConfig, ImageEditModelConfig)):
        _image_manager.register(_model_id, _model_config)

# Audio model configs are registered in tts_routes; this is the same singleton instance
_tts_manager = TTSModelManager()


@router.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    await aclose_client()


@router.get("/health/{model_name}")
async def proxy_health(request: Request, model_name: str):
    if model_name not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{model_name}' not configured")

    model_config = config.MODEL_CONFIGS[model_name]

    # For image models, check if the in-process pipeline is loaded
    if isinstance(model_config, (ImageModelConfig, ImageEditModelConfig)):
        if _image_manager.is_loaded(model_name):
            return Response(
                content=json.dumps({"status": "ok"}).encode(),
                status_code=200,
                media_type="application/json",
            )
        elif _image_manager.is_capable(model_name):
            return Response(
                content=json.dumps({"status": "capable", "detail": "Model not loaded but can be loaded on demand"}).encode(),
                status_code=202,
                media_type="application/json",
            )
        else:
            return Response(
                content=json.dumps({"status": "error", "detail": "Image model not registered"}).encode(),
                status_code=503,
                media_type="application/json",
            )

    # For audio models, check if the in-process TTS pipeline is loaded
    if isinstance(model_config, AudioModelConfig):
        if _tts_manager.is_loaded(model_name):
            return Response(
                content=json.dumps({"status": "ok"}).encode(),
                status_code=200,
                media_type="application/json",
            )
        elif _tts_manager.is_capable(model_name):
            return Response(
                content=json.dumps(
                    {"status": "capable", "detail": "Model not loaded but can be loaded on demand"}
                ).encode(),
                status_code=202,
                media_type="application/json",
            )
        else:
            return Response(
                content=json.dumps({"status": "error", "detail": "Audio model not registered"}).encode(),
                status_code=503,
                media_type="application/json",
            )

    # For text and embedding models, proxy to the llamacpp /health endpoint
    url = f"{model_config.url}/health"

    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        response: httpx.Response = await client.get(url, headers=headers)
    except Exception as e:
        logger.exception("proxy health forward to %s failed", url)
        raise HTTPException(status_code=500, detail=f"Error forwarding request: {type(e).__name__}: {e}")

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
    if (key_error := check_api_key(token)) is not None:
        return key_error

    # Get model from request
    model_name = proxy_request_data.model

    if model_name not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{model_name}' not found")

    model_config = config.MODEL_CONFIGS[model_name]

    if not isinstance(model_config, (TextModelConfig, EmbeddingModelConfig)):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="This endpoint is only for text and embedding models"
        )

    if full_path not in model_config.allowed_paths:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid inference path")

    payment_payload = request.headers.get("x-payment") or None
    payment_requirements = request.headers.get("x-payment-requirements") or None

    user_context = UserContext(
        key=token,
        model_name=model_name,
        endpoint=full_path,
        payment_payload=payment_payload,
        payment_requirements=payment_requirements,
    )

    # Get the original request body & headers
    headers = dict(request.headers)
    body = await request.body()

    # Clean up headers
    headers.pop("host", None)

    # Two body rewrites can apply: inline remote image URLs (fetch + base64) so vLLM
    # never fetches them itself, and inject stream_options.include_usage so streaming
    # usage accounting works. Parse once (FastAPI already parsed the body for the
    # pydantic model, so request.json() is cached), mutate, re-serialize once.
    want_inline = full_path in IMAGE_INLINE_PATHS
    want_stream_usage = full_path in STREAM_USAGE_PATHS
    if want_inline or want_stream_usage:
        try:
            body_json = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body_json = None
        body_changed = False
        if isinstance(body_json, dict):
            # Fetch/validation failures surface as 400; an unexpected bug forwards the
            # body unmodified rather than 500 (a 500 would make the api router fail over
            # on every replica).
            if want_inline:
                try:
                    body_json, inlined = await inline_remote_images(body_json)
                # Load-bearing: re-raise fetch 400s so they reach the client. Without
                # this they'd be caught by the except Exception below and the body
                # forwarded unmodified (defeating the SSRF/validation guards).
                except HTTPException:
                    raise
                except Exception:
                    logger.exception("image inline failed; forwarding request unmodified")
                    inlined = False
                body_changed = body_changed or inlined
            # stream_options injection is llama.cpp-safe (it ignores the field) and
            # stays gated to the OpenAI completion paths only.
            if want_stream_usage and body_json.get("stream") is True:
                stream_options = body_json.get("stream_options") or {}
                if not stream_options.get("include_usage"):
                    stream_options["include_usage"] = True
                    body_json["stream_options"] = stream_options
                    body_changed = True
            if body_changed:
                body = json.dumps(body_json).encode()
                headers.pop("content-length", None)

    url = f"{model_config.url}/{full_path}"

    try:
        req = client.build_request("POST", url, content=body, headers=headers, params=request.query_params)
        response = await client.send(req, stream=True)

        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            # Try to forward the original error from llama.cpp
            try:
                error_json = json.loads(error_body)
                detail = error_json.get("error", {}).get("message", error_body.decode(errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = (
                    error_body.decode(errors="replace") if error_body else f"Upstream returned {response.status_code}"
                )
            raise HTTPException(status_code=response.status_code, detail=detail)

        is_streaming_response = "text/event-stream" in response.headers.get("content-type", "")

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
                            usage_data = TextUsageFullData(
                                **user_context.model_dump(),
                                **extract_usage_info_from_raw(full_response_buffer, user_context).model_dump(),
                            )
                            background_tasks.add_task(report_usage_event_task, usage_data)
                        except Exception:
                            logger.exception("streaming usage extraction failed")

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
                    usage_data = TextUsageFullData(
                        **user_context.model_dump(),
                        **extract_usage_info(response_json, user_context).model_dump(),
                    )
                    background_tasks.add_task(report_usage_event_task, usage_data)
                except Exception:
                    logger.exception("usage report failed")

            return Response(
                content=response_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("Content-Type", ""),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("proxy forward to %s failed", url)
        raise HTTPException(status_code=500, detail=f"Error forwarding request: {type(e).__name__}: {e}")
