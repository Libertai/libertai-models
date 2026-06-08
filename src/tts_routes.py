import asyncio
from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import AudioModelConfig, config
from src.interfaces.usage import AudioUsage, AudioUsageFullData, UserContext
from src.tts_generation import TTSModelManager, synthesize_wav
from src.usage import report_usage_event_task

router = APIRouter(tags=["Audio"])
security = HTTPBearer()
keys_manager = KeysManager()
tts_manager = TTSModelManager()

# Register audio model configs at import
for _model_id, _model_config in config.MODEL_CONFIGS.items():
    if isinstance(_model_config, AudioModelConfig):
        tts_manager.register(_model_id, _model_config)

MIN_SPEED = 0.25
MAX_SPEED = 4.0
MAX_INPUT_CHARS = 8192


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0


def _validate(body: SpeechRequest, token: str) -> AudioModelConfig:
    if not keys_manager.key_exists(token):
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid API key")
    if body.model not in config.MODEL_CONFIGS:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model '{body.model}' not found")
    model_config = config.MODEL_CONFIGS[body.model]
    if not isinstance(model_config, AudioModelConfig):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Model '{body.model}' is not an audio model")
    if "v1/audio/speech" not in model_config.allowed_paths:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid endpoint for this model")
    if not body.input or not body.input.strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="input cannot be empty")
    if len(body.input) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"input exceeds {MAX_INPUT_CHARS} characters")
    if body.response_format != "wav":
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"response_format '{body.response_format}' not supported; only 'wav' is available",
        )
    if not (MIN_SPEED <= body.speed <= MAX_SPEED):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"speed must be between {MIN_SPEED} and {MAX_SPEED}"
        )
    return model_config


def _track_usage(token: str, model_name: str, character_count: int, background_tasks: BackgroundTasks) -> None:
    try:
        user_context = UserContext(key=token, model_name=model_name, endpoint="v1/audio/speech")
        usage_data = AudioUsageFullData(
            **user_context.model_dump(),
            **AudioUsage(input_tokens=character_count).model_dump(),
        )
        background_tasks.add_task(report_usage_event_task, usage_data)
    except Exception as e:
        print(f"Exception occurred during usage report: {str(e)}")


@router.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Response:
    token = credentials.credentials
    model_config = _validate(body, token)
    voice = body.voice or model_config.default_voice

    def _blocking() -> bytes:
        pipeline, inference_lock = tts_manager.acquire(body.model)
        try:
            with inference_lock:
                return synthesize_wav(pipeline, body.input, voice, body.speed)
        finally:
            tts_manager.release(body.model)

    loop = asyncio.get_running_loop()
    try:
        wav_bytes = await loop.run_in_executor(None, _blocking)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"TTS synthesis failed: {str(e)}")

    _track_usage(token, body.model, len(body.input), background_tasks)
    return Response(content=wav_bytes, media_type="audio/wav")
