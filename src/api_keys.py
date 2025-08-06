from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import config
from src.cryptography import verify_signed_payload


class KeysManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KeysManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # Check if already initialized
            self.keys = set()

    def add_keys(self, keys):
        self.keys.update(keys)

    def key_exists(self, key):
        return key in self.keys


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
        # Verify and decrypt the payload using the public key
        decrypted_data = verify_signed_payload(payload.encrypted_payload, config.API_PUBLIC_KEY)

        # Extract the keys from the decrypted data
        keys = decrypted_data.get("keys", [])

        # Update the KeysManager with the new keys
        keys_manager = KeysManager()
        keys_manager.add_keys(set(keys))

        return {"status": "success", "keys_received": len(keys)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing encrypted keys: {str(e)}")
