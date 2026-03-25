import asyncio
import io
import random
import time
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import ImageEditModelConfig, ImageModelConfig, config
from src.image_generation import ImageModelManager, edit_image, generate_image
from src.interfaces.usage import ImageUsage, ImageUsageFullData, UserContext
from src.usage import report_usage_event_task

router = APIRouter(tags=["Image Generation"])
security = HTTPBearer()
keys_manager = KeysManager()
image_manager = ImageModelManager()

# Register image model configs
for _model_id, _model_config in config.MODEL_CONFIGS.items():
    if isinstance(_model_config, (ImageModelConfig, ImageEditModelConfig)):
        image_manager.register(_model_id, _model_config)

# Constants
MAX_DIMENSION = 2048
MAX_STEPS = 50
MIN_STEPS = 1
DEFAULT_Z_IMAGE_TURBO_STEPS = 9
MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_IMAGE_PIXELS = 2048 * 2048  # ~4 megapixels


# Helper functions
def validate_prompt(prompt: str) -> None:
    """Validate prompt is not empty or whitespace"""
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Prompt cannot be empty")


def validate_dimensions(width: int, height: int) -> None:
    """Validate image dimensions are positive and within limits"""
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Width and height must be positive")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Width and height must not exceed {MAX_DIMENSION}px"
        )


def validate_steps(steps: int) -> None:
    """Validate number of inference steps is within reasonable range"""
    if steps < MIN_STEPS or steps > MAX_STEPS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Steps must be between {MIN_STEPS} and {MAX_STEPS}"
        )


def validate_cfg_scale(cfg_scale: float) -> None:
    """Validate CFG scale is non-negative"""
    if cfg_scale < 0:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="cfg_scale must be non-negative")


def validate_model_and_endpoint(model_name: str, endpoint: str, token: str) -> ImageModelConfig | ImageEditModelConfig:
    """Validate API key, model existence, model type, and endpoint path"""
    # Auth validation
    if not keys_manager.key_exists(token):
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid API key")

    # Model existence
    if model_name not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{model_name}' not found")

    # Model type
    model_config = config.MODEL_CONFIGS[model_name]
    if not isinstance(model_config, (ImageModelConfig, ImageEditModelConfig)):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Model '{model_name}' is not an image model")

    # Endpoint path
    if endpoint not in model_config.allowed_paths:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid endpoint for this model")

    return model_config


MAX_IMAGES = 4


def track_usage(
    token: str,
    model_name: str,
    endpoint: str,
    background_tasks: BackgroundTasks,
    image_count: int = 1,
    payment_payload: str | None = None,
    payment_requirements: str | None = None,
) -> None:
    """Track image generation usage"""
    try:
        user_context = UserContext(
            key=token,
            model_name=model_name,
            endpoint=endpoint,
            payment_payload=payment_payload,
            payment_requirements=payment_requirements,
        )
        usage_data = ImageUsageFullData(
            **user_context.model_dump(),
            **ImageUsage(image_count=image_count).model_dump(),
        )
        background_tasks.add_task(report_usage_event_task, usage_data)
    except Exception as e:
        print(f"Exception occurred during usage report: {str(e)}")


# OpenAI format models
class OpenAIImageRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"  # Only b64_json supported
    remove_background: bool = False  # Remove background with rembg

    class Config:
        extra = "allow"


MAX_INPUT_IMAGES = 4


class ImageData(BaseModel):
    b64_json: str


class OpenAIImageResponse(BaseModel):
    created: int
    data: list[ImageData]


# A1111 format models
class A1111Request(BaseModel):
    model: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = DEFAULT_Z_IMAGE_TURBO_STEPS
    cfg_scale: float = 0.0  # Turbo model doesn't need CFG
    seed: int = -1
    remove_background: bool = False  # Remove background with rembg

    class Config:
        extra = "allow"


class A1111Response(BaseModel):
    images: list[str]  # Base64 encoded PNGs
    parameters: dict
    info: str


@router.post("/v1/images/generations", response_model=OpenAIImageResponse)
async def generate_image_openai(
    request: OpenAIImageRequest,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    background_tasks: BackgroundTasks,
    raw_request: Request,
):
    """OpenAI-compatible image generation endpoint"""
    token = credentials.credentials
    payment_payload = raw_request.headers.get("x-payment") or None
    payment_requirements = raw_request.headers.get("x-payment-requirements") or None

    # Validate model and endpoint
    validate_model_and_endpoint(request.model, "v1/images/generations", token)

    # Validate parameters
    if request.n < 1 or request.n > MAX_IMAGES:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"n must be between 1 and {MAX_IMAGES}")

    if request.response_format != "b64_json":
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Only b64_json format is supported")

    # Parse and validate size
    try:
        parts = request.size.split("x")
        if len(parts) != 2:
            raise ValueError("Invalid format")
        width, height = map(int, parts)
    except ValueError:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid size format. Use WIDTHxHEIGHT (e.g., 1024x1024)"
        )

    validate_prompt(request.prompt)
    validate_dimensions(width, height)

    try:
        def _blocking_generate():
            pipeline = image_manager.acquire(request.model)
            try:
                results = []
                for _ in range(request.n):
                    image_b64 = generate_image(
                        pipeline=pipeline,
                        prompt=request.prompt,
                        width=width,
                        height=height,
                        steps=DEFAULT_Z_IMAGE_TURBO_STEPS,
                        guidance_scale=0.0,
                        remove_background=request.remove_background,
                    )
                    results.append(ImageData(b64_json=image_b64))
                return results
            finally:
                image_manager.release(request.model)

        loop = asyncio.get_running_loop()
        images = await loop.run_in_executor(None, _blocking_generate)

        # Track usage
        track_usage(
            token,
            request.model,
            "v1/images/generations",
            background_tasks,
            image_count=request.n,
            payment_payload=payment_payload,
            payment_requirements=payment_requirements,
        )

        return OpenAIImageResponse(
            created=int(time.time()),
            data=images,
        )

    except RuntimeError as e:
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@router.post("/v1/images/edits", response_model=OpenAIImageResponse)
async def edit_image_openai(
    raw_request: Request,
    background_tasks: BackgroundTasks,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    """OpenAI-compatible image editing endpoint (multipart/form-data)"""
    token = credentials.credentials

    # Parse multipart form data
    form = await raw_request.form()

    prompt = form.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="prompt is required")

    model_name = form.get("model")
    if not model_name or not isinstance(model_name, str):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="model is required")

    n_raw = form.get("n", "1")
    try:
        n = int(str(n_raw))
    except (ValueError, TypeError):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="n must be an integer")

    seed_raw = form.get("seed", "-1")
    try:
        seed = int(str(seed_raw))
    except (ValueError, TypeError):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="seed must be an integer")

    response_format = form.get("response_format", "b64_json")
    if response_format != "b64_json":
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Only b64_json format is supported")

    # Validate
    validate_prompt(prompt)
    validate_model_and_endpoint(model_name, "v1/images/edits", token)

    if n < 1 or n > MAX_IMAGES:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"n must be between 1 and {MAX_IMAGES}")

    # Extract images from form (supports multiple files under "image" key)
    from PIL import Image as PILImage

    # Guard against decompression bombs
    PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    input_images: list[PILImage.Image] = []
    for item in form.getlist("image"):
        if hasattr(item, 'read'):  # UploadFile
            # Read at most MAX_IMAGE_FILE_SIZE + 1 to detect oversized uploads without loading full file
            contents = await item.read(MAX_IMAGE_FILE_SIZE + 1)
            if len(contents) > MAX_IMAGE_FILE_SIZE:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Image file too large (max {MAX_IMAGE_FILE_SIZE // (1024 * 1024)}MB)"
                )
            try:
                raw_img = PILImage.open(io.BytesIO(contents))
                raw_img.verify()  # Check headers without full decode
                # Re-open after verify (verify leaves file in unusable state)
                img = PILImage.open(io.BytesIO(contents)).convert("RGB")
            except Exception:
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid image file")
            input_images.append(img)

    if not input_images:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="At least one image is required")

    if len(input_images) > MAX_INPUT_IMAGES:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Maximum {MAX_INPUT_IMAGES} input images allowed"
        )

    payment_payload = raw_request.headers.get("x-payment") or None
    payment_requirements = raw_request.headers.get("x-payment-requirements") or None

    try:
        def _blocking_edit():
            pipeline = image_manager.acquire(model_name)
            try:
                return edit_image(
                    pipeline=pipeline,
                    images=input_images,
                    prompt=prompt,
                    num_images=n,
                    seed=seed,
                )
            finally:
                image_manager.release(model_name)

        loop = asyncio.get_running_loop()
        result_images = await loop.run_in_executor(None, _blocking_edit)

        images = [ImageData(b64_json=b64) for b64 in result_images]

        track_usage(
            token,
            model_name,
            "v1/images/edits",
            background_tasks,
            image_count=n,
            payment_payload=payment_payload,
            payment_requirements=payment_requirements,
        )

        return OpenAIImageResponse(
            created=int(time.time()),
            data=images,
        )

    except RuntimeError as e:
        print(f"Error editing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error editing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error editing image: {str(e)}")


@router.post("/sdapi/v1/txt2img", response_model=A1111Response)
async def generate_image_a1111(
    request: A1111Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    background_tasks: BackgroundTasks,
    raw_request: Request,
):
    """AUTOMATIC1111-compatible txt2img endpoint"""
    token = credentials.credentials
    payment_payload = raw_request.headers.get("x-payment") or None
    payment_requirements = raw_request.headers.get("x-payment-requirements") or None

    # Validate model and endpoint
    validate_model_and_endpoint(request.model, "sdapi/v1/txt2img", token)

    # Validate input parameters
    validate_prompt(request.prompt)
    validate_dimensions(request.width, request.height)
    validate_steps(request.steps)
    validate_cfg_scale(request.cfg_scale)

    # Create seed if none is passed
    seed = request.seed if request.seed >= 0 else random.randint(0, 2147483647)

    try:
        def _blocking_generate():
            pipeline = image_manager.acquire(request.model)
            try:
                return generate_image(
                    pipeline=pipeline,
                    prompt=request.prompt,
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    seed=seed,
                    remove_background=request.remove_background,
                )
            finally:
                image_manager.release(request.model)

        loop = asyncio.get_running_loop()
        image_b64 = await loop.run_in_executor(None, _blocking_generate)

        # Track usage
        track_usage(
            token,
            request.model,
            "sdapi/v1/txt2img",
            background_tasks,
            payment_payload=payment_payload,
            payment_requirements=payment_requirements,
        )

        return A1111Response(
            images=[image_b64],
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "seed": seed,
                "remove_background": request.remove_background,
            },
            info="Generated with Z-Image-Turbo",
        )

    except RuntimeError as e:
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
