import base64
import json
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey


def verify_signed_payload(encrypted_payload: dict[str, str], public_key_b64: str) -> dict[str, Any]:
    public_key_pem = base64.b64decode(public_key_b64.encode()).decode()
    public_key: RSAPublicKey = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())  # type: ignore

    data = base64.b64decode(encrypted_payload["data"].encode())
    signature = base64.b64decode(encrypted_payload["signature"].encode())

    # Verify signature with public key
    try:
        public_key.verify(
            signature,
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
    except Exception as e:
        raise ValueError(f"Signature verification failed: {e}")

    # Return the decrypted data
    return json.loads(data.decode())
