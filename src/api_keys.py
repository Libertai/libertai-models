from http import HTTPStatus
from typing import ClassVar

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import config
from src.cryptography import verify_signed_payload


class KeysManager:
    _instance = None
    keys: ClassVar[set[str]] = set()
    # key -> {"reason": str, "message": str} for real-but-unusable keys (limits/credits/disabled)
    invalid_keys: ClassVar[dict[str, dict]] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def reset_keys(self, new_keys: set[str], invalid_keys: dict[str, dict] | None = None):
        # singleton: reassign via the class, not self, since keys/invalid_keys are ClassVars
        KeysManager.keys = new_keys
        KeysManager.invalid_keys = invalid_keys or {}

    def key_exists(self, key):
        return key in self.keys


def apply_key_payload(decrypted_data: dict) -> int:
    """Update the KeysManager from a decrypted distribution payload; returns valid-key count."""
    keys = decrypted_data.get("keys", [])
    invalid_keys = decrypted_data.get("invalid_keys") or {}
    KeysManager().reset_keys(set(keys), dict(invalid_keys))
    return len(keys)


def check_api_key(token: str) -> JSONResponse | None:
    """None if the key is usable; otherwise the error response to return.

    Blocked keys get an OpenAI-shaped 403 (`error.message` is what openai-node
    displays); unknown keys keep the legacy 401 body.
    """
    keys_manager = KeysManager()
    if keys_manager.key_exists(token):
        return None
    invalid_info = keys_manager.invalid_keys.get(token)
    if invalid_info is not None:
        return JSONResponse(
            status_code=HTTPStatus.FORBIDDEN,
            content={
                "error": {
                    "message": invalid_info.get("message") or "This API key is currently not usable.",
                    "type": "invalid_request_error",
                    "code": invalid_info.get("reason") or "forbidden",
                }
            },
        )
    return JSONResponse(status_code=HTTPStatus.UNAUTHORIZED, content={"detail": "Invalid API key"})


router = APIRouter(tags=["LibertAI"], prefix="/libertai")


class EncryptedApiKeysPayload(BaseModel):
    encrypted_payload: dict[str, str]


@router.post("/api-keys")
async def receive_api_keys(payload: EncryptedApiKeysPayload):
    """
    Endpoint for receiving encrypted API keys from the backend.
    The backend will call this endpoint with encrypted/signed keys.
    """
    try:
        decrypted_data = verify_signed_payload(payload.encrypted_payload, config.API_PUBLIC_KEY)
        keys_received = apply_key_payload(decrypted_data)
        return {"status": "success", "keys_received": keys_received}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing encrypted keys: {e!s}")
