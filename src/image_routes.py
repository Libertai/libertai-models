import time
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import ImageModelConfig, config
from src.image_generation import ImagePipelineManager, generate_image
from src.interfaces.usage import Usage, UserContext, UsageFullData
from src.usage import report_usage_event_task

router = APIRouter(tags=["Image Generation"])
security = HTTPBearer()
keys_manager = KeysManager()
pipeline_manager = ImagePipelineManager()

# Constants
MAX_DIMENSION = 2048
MAX_STEPS = 50
MIN_STEPS = 1
DEFAULT_Z_IMAGE_TURBO_STEPS = 9


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


def validate_model_and_endpoint(model_name: str, endpoint: str, token: str) -> ImageModelConfig:
    """Validate API key, model existence, model type, and endpoint path"""
    # Auth validation
    if not keys_manager.key_exists(token):
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid API key")

    # Model existence
    if model_name not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{model_name}' not found")

    # Model type
    model_config = config.MODEL_CONFIGS[model_name]
    if not isinstance(model_config, ImageModelConfig):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Model '{model_name}' is not an image model")

    # Endpoint path
    if endpoint not in model_config.allowed_paths:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid endpoint for this model")

    return model_config


def track_usage(token: str, model_name: str, endpoint: str, background_tasks: BackgroundTasks) -> None:
    """Track image generation usage (0 tokens)"""
    try:
        user_context = UserContext(key=token, model_name=model_name, endpoint=endpoint)
        usage_data = UsageFullData(
            **user_context.model_dump(),
            **Usage(input_tokens=0, output_tokens=0, cached_tokens=0).model_dump(),
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

    class Config:
        extra = "allow"


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
    sampler_name: str = "Euler a"  # Ignored, just for API compat
    seed: int = -1

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
):
    """OpenAI-compatible image generation endpoint"""
    token = credentials.credentials

    # Validate model and endpoint
    model_config = validate_model_and_endpoint(request.model, "v1/images/generations", token)

    # Validate parameters
    if request.n != 1:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Only n=1 is supported")

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
        # Load pipeline if not already loaded
        pipeline_manager.load_pipeline(model_config.local_path)
        pipeline = pipeline_manager.get_pipeline()

        # Generate image
        image_b64 = generate_image(
            pipeline=pipeline,
            prompt=request.prompt,
            width=width,
            height=height,
            steps=DEFAULT_Z_IMAGE_TURBO_STEPS,  # Z-Image-Turbo optimal
            guidance_scale=0.0,  # Turbo doesn't need CFG
        )

        # Track usage
        track_usage(token, request.model, "v1/images/generations", background_tasks)

        return OpenAIImageResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=image_b64)],
        )

    except RuntimeError as e:
        # RuntimeError raised from our code (CUDA OOM, model loading failures)
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@router.post("/sdapi/v1/txt2img", response_model=A1111Response)
async def generate_image_a1111(
    request: A1111Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    background_tasks: BackgroundTasks,
):
    """AUTOMATIC1111-compatible txt2img endpoint"""
    token = credentials.credentials

    # Validate model and endpoint
    model_config = validate_model_and_endpoint(request.model, "sdapi/v1/txt2img", token)

    # Validate input parameters
    validate_prompt(request.prompt)
    validate_dimensions(request.width, request.height)
    validate_steps(request.steps)
    validate_cfg_scale(request.cfg_scale)

    try:
        # Load pipeline if not already loaded
        pipeline_manager.load_pipeline(model_config.local_path)
        pipeline = pipeline_manager.get_pipeline()

        # Generate image (note: Z-Image doesn't support negative prompts natively)
        image_b64 = generate_image(
            pipeline=pipeline,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance_scale=request.cfg_scale,
            seed=request.seed,
        )

        # Track usage
        track_usage(token, request.model, "sdapi/v1/txt2img", background_tasks)

        return A1111Response(
            images=[image_b64],
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "seed": request.seed,
            },
            info="Generated with Z-Image-Turbo",
        )

    except RuntimeError as e:
        # RuntimeError raised from our code (CUDA OOM, model loading failures)
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
